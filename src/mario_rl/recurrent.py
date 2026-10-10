"""In-context meta-RL (RL^2; Duan et al. 2016, Wang et al. 2016) for goal B.

A GRU policy plays *trials*: several attempts at the same level (``EnvConfig.trial_episodes``),
with its memory kept across the attempts and reset only between trials. The value function
treats the whole trial as one episode, so information gathered in an early attempt that pays off
in a later one is rewarded. At test time the agent should improve over the attempts at an
unseen level without any weight update: learning from its own previous deaths.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict
from multiprocessing import connection as mp_connection
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import EnvConfig, observation_spec
from .levels import parse_levels
from .models import _encoder, _init
from .obs import allocate, stack, store, to_torch
from .plr import LevelSampler
from .workers import WorkerPool


class RecurrentActorCritic(nn.Module):
    """Separate actor / critic encoders, each followed by a GRU cell; state shape (B, 2, hidden)."""

    def __init__(self, obs_type: str, obs_shape, n_actions: int, hidden: int = 256, arch: str = "default"):
        super().__init__()
        self.obs_type, self.n_actions, self.hidden, self.arch = obs_type, n_actions, hidden, arch
        self.obs_shape = (
            {k: tuple(v) for k, v in obs_shape.items()} if isinstance(obs_shape, dict) else tuple(obs_shape)
        )
        self.actor_encoder = _encoder(obs_type, self.obs_shape, arch)
        self.critic_encoder = _encoder(obs_type, self.obs_shape, arch)
        self.actor_rnn = nn.GRU(self.actor_encoder.out_dim, hidden)  # (T, B, F); cuDNN for sequences
        self.critic_rnn = nn.GRU(self.critic_encoder.out_dim, hidden)
        self.policy = _init(nn.Linear(hidden, n_actions), gain=0.01)
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, 2, self.hidden, device=device)

    def _features(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_encoder.preprocess(obs)
        return self.actor_encoder(x), self.critic_encoder(x)

    def step(self, obs, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: (logits, value, new state)."""
        fa, fc = self._features(obs)
        ha = self.actor_rnn(fa.unsqueeze(0), state[:, 0].unsqueeze(0).contiguous())[0][0]
        hc = self.critic_rnn(fc.unsqueeze(0), state[:, 1].unsqueeze(0).contiguous())[0][0]
        return self.policy(ha), self.value(hc).squeeze(-1), torch.stack([ha, hc], 1)

    def forward_seq(
        self, obs, state0: torch.Tensor, resets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (T, B, ...) per key, state0: (B, 2, H), resets: (T, B) with 1 = clear the state before step t.

        Each env's sequence is cut at its resets into pieces; the first piece starts from state0, later ones
        from zeros. All pieces run as one packed cuDNN GRU call (identical to stepping with resets)."""
        T, B = resets.shape
        flat = (
            {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
            if isinstance(obs, dict)
            else obs.flatten(0, 1)
        )
        fa, fc = self._features(flat)
        fa, fc = fa.view(T, B, -1), fc.view(T, B, -1)
        starts = resets.detach().cpu().numpy() > 0.5
        pieces = []  # (env, start, end)
        for b in range(B):
            cuts = [0, *[t for t in range(1, T) if starts[t, b]], T]
            pieces += [(b, s0, s1) for s0, s1 in itertools.pairwise(cuts)]
        lengths = torch.tensor([s1 - s0 for _, s0, s1 in pieces])
        idx_t = torch.cat([torch.arange(s0, s1) for _, s0, s1 in pieces])
        idx_b = torch.cat([torch.full((s1 - s0,), b) for b, s0, s1 in pieces])
        first = torch.tensor([s0 == 0 for _, s0, _ in pieces], device=state0.device).view(1, -1, 1).float()
        envs = torch.tensor([b for b, _, _ in pieces], device=state0.device)

        def run(rnn, feats, h_env):
            seqs = torch.split(feats[idx_t, idx_b], lengths.tolist())
            packed = nn.utils.rnn.pack_sequence(seqs, enforce_sorted=False)
            h0 = (h_env[envs].unsqueeze(0) * first).contiguous()  # pieces after a reset start from zeros
            out, _ = rnn(packed, h0)
            out, _ = nn.utils.rnn.pad_packed_sequence(out)  # (max_len, pieces, H)
            full = feats.new_zeros(T, B, out.shape[-1])
            for j, (b, s0, s1) in enumerate(pieces):
                full[s0:s1, b] = out[: s1 - s0, j]
            return full

        ha_seq = run(self.actor_rnn, fa, state0[:, 0])
        hc_seq = run(self.critic_rnn, fc, state0[:, 1])
        return self.policy(ha_seq), self.value(hc_seq).squeeze(-1)

    def parameter_groups(self) -> list[list[nn.Parameter]]:
        actor = [*self.actor_encoder.parameters(), *self.actor_rnn.parameters(), *self.policy.parameters()]
        critic = [*self.critic_encoder.parameters(), *self.critic_rnn.parameters(), *self.value.parameters()]
        return [actor, critic]

    def spec(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "recurrent": True,
            "hidden": self.hidden,
            "arch": self.arch,
        }


def _sample_step(policy: RecurrentActorCritic, obs, state):
    logits, value, new_state = policy.step(obs, state)
    log_probs = torch.log_softmax(logits, -1)
    u = torch.rand_like(log_probs).clamp_(1e-10, 1.0)  # Gumbel-max sampling
    action = (log_probs - (-u.log()).log()).argmax(-1)
    packed = torch.stack([action.float(), log_probs.gather(-1, action[:, None]).squeeze(-1), value])
    return packed, new_state


class RecurrentActor:
    """Batched action selection with per-environment memory; CUDA graphs per batch size on a GPU."""

    def __init__(self, policy: RecurrentActorCritic):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.use_graphs = self.device.type == "cuda"
        self.graphs: dict = {}

    def _dev(self, items):
        obs = to_torch(stack(items))
        return (
            {k: v.to(self.device) for k, v in obs.items()} if isinstance(obs, dict) else obs.to(self.device)
        )

    def _graph(self, obs, state):
        b = state.shape[0]
        if b not in self.graphs:
            s_obs = {k: v.clone() for k, v in obs.items()} if isinstance(obs, dict) else obs.clone()
            s_state = state.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        _sample_step(self.policy, s_obs, s_state)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = _sample_step(self.policy, s_obs, s_state)
            self.graphs[b] = (graph, s_obs, s_state, out)
        return self.graphs[b]

    @torch.no_grad()
    def act(self, items, state: torch.Tensor, greedy: bool = False):
        """(actions, log-probs, values, new state) for a list of observations and their memory (B, 2, H)."""
        obs = self._dev(items)
        if self.use_graphs and not greedy:
            graph, s_obs, s_state, (packed, new_state) = self._graph(obs, state)
            if isinstance(obs, dict):
                for k in obs:
                    s_obs[k].copy_(obs[k])
            else:
                s_obs.copy_(obs)
            s_state.copy_(state)
            graph.replay()
            packed, new_state = packed.clone(), new_state.clone()
        elif greedy:
            logits, value, new_state = self.policy.step(obs, state)
            log_probs = torch.log_softmax(logits, -1)
            action = logits.argmax(-1)
            packed = torch.stack([action.float(), log_probs.gather(-1, action[:, None]).squeeze(-1), value])
        else:
            packed, new_state = _sample_step(self.policy, obs, state)
        out = packed.cpu().numpy()
        return out[0].astype(np.int64), out[1], out[2], new_state

    @torch.no_grad()
    def values(self, items, state: torch.Tensor) -> np.ndarray:
        return self.policy.step(self._dev(items), state)[1].cpu().numpy()


def rollout_recurrent(
    pool: WorkerPool, actor: RecurrentActor, num_steps: int, state: torch.Tensor, weights=None
):
    """One segment of ``num_steps`` per worker with asynchronous dynamic batching (whichever workers are ready
    act together). ``state`` (n, 2, H) is the memory carried in, cleared where a trial ended, and updated in
    place. Returns the per-worker buffers, the memory at the segment start, next values and episodes."""
    n = len(pool.conns)
    for conn in pool.conns:
        conn.send(("begin", weights))
    obs, level = map(list, zip(*pool._gather(), strict=True))
    state0 = state.clone()
    buf = {
        "obs": [allocate(o, num_steps) for o in obs],
        "actions": np.empty((n, num_steps), np.int64),
        "logprobs": np.empty((n, num_steps), np.float32),
        "values": np.empty((n, num_steps), np.float32),
        "rewards": np.empty((n, num_steps), np.float32),
        "bootstrap": np.zeros((n, num_steps), np.float32),
        "dones": np.zeros((n, num_steps), bool),
        "resets": np.zeros((n, num_steps), np.float32),  # 1 = memory cleared right before this step
        "level_ids": np.empty((n, num_steps), np.int16),
    }
    episodes: list[dict] = []
    truncated = []  # (worker, step, final obs, memory) -> valued at the end
    t = [0] * n
    index = {conn: i for i, conn in enumerate(pool.conns)}
    ready, waiting = list(range(n)), set()
    while ready or waiting:
        if ready:
            a, lp, v, new_state = actor.act([obs[i] for i in ready], state[ready])
            state[ready] = new_state
            for j, i in enumerate(ready):
                store(buf["obs"][i], t[i], obs[i])
                buf["level_ids"][i, t[i]] = level[i]
                buf["actions"][i, t[i]], buf["logprobs"][i, t[i]], buf["values"][i, t[i]] = a[j], lp[j], v[j]
                pool.conns[i].send(("step", int(a[j])))
            waiting.update(ready)
            ready = []
        for conn in mp_connection.wait([pool.conns[i] for i in waiting]):
            i = index[conn]
            status, payload = conn.recv()
            if status == "error":
                pool.close(force=True)
                raise RuntimeError(payload)
            o, r, done, final, episode, lv = payload
            if final is not None:
                truncated.append((i, t[i], final, state[i : i + 1].clone()))
            obs[i], level[i], buf["rewards"][i, t[i]], buf["dones"][i, t[i]] = o, lv, r, done
            if episode is not None:
                episodes.append(episode)
            if done:
                state[i].zero_()  # a new trial starts: forget
                if t[i] + 1 < num_steps:
                    buf["resets"][i, t[i] + 1] = 1.0
            waiting.discard(i)
            t[i] += 1
            if t[i] < num_steps:
                ready.append(i)
    for i, step, final, mem in truncated:
        buf["bootstrap"][i, step] = actor.values([final], mem)[0]
    next_values = actor.values(obs, state)
    return buf, state0, next_values, episodes


def _gae(rewards, values, dones, next_value, gamma, lam):
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def ppo_update_recurrent(model, optimizer, cfg, obs, actions, logprobs, returns, adv, resets, state0):
    """obs: per key (n, T, ...) tensors; the rest (n, T); minibatches are groups of whole worker sequences."""
    n = actions.shape[0]
    per_mb = max(1, n // cfg.num_minibatches)
    stats = {"pg": [], "v": [], "ent": [], "kl": [], "clip": []}
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, per_mb):
            idx = perm[start : start + per_mb]
            seq = (
                {k: v[idx].transpose(0, 1) for k, v in obs.items()}
                if isinstance(obs, dict)
                else obs[idx].transpose(0, 1)
            )
            logits, values = model.forward_seq(seq, state0[idx], resets[idx].transpose(0, 1))
            dist = Categorical(logits=logits)
            act = actions[idx].transpose(0, 1)
            logratio = dist.log_prob(act) - logprobs[idx].transpose(0, 1)
            ratio = logratio.exp()
            a = adv[idx].transpose(0, 1)
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            v_loss = 0.5 * (values - returns[idx].transpose(0, 1)).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg - cfg.ent_coef * ent + cfg.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            for group in model.parameter_groups():
                nn.utils.clip_grad_norm_(group, cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats["kl"].append(((ratio - 1) - logratio).mean().item())
                stats["clip"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
            stats["pg"].append(pg.item())
            stats["v"].append(v_loss.item())
            stats["ent"].append(ent.item())
    return {
        "loss/policy": float(np.mean(stats["pg"])),
        "loss/value": float(np.mean(stats["v"])),
        "entropy": float(np.mean(stats["ent"])),
        "approx_kl": float(np.mean(stats["kl"])),
        "clipfrac": float(np.mean(stats["clip"])),
    }


def evaluate_trials(
    pool: WorkerPool, actor: RecurrentActor, levels, trials_per_level: int, attempts: int, greedy=False
):
    """Play ``trials_per_level`` trials of ``attempts`` attempts on each level, no learning. Returns a list of
    episode summaries with ``attempt`` set; memory is kept across the attempts of a trial."""
    jobs = [lv for lv in levels for _ in range(trials_per_level)]
    n = len(pool.conns)
    results: list[dict] = []
    queue = list(jobs)
    active: dict[int, dict] = {}  # worker -> {"obs", "attempts_left"}
    state = actor.policy.initial_state(n, actor.device)

    def start(i):
        level = queue.pop(0)
        pool.conns[i].send(("reset_level", {"level": level, "attempt": 0}))
        status, o = pool.conns[i].recv()
        if status == "error":
            raise RuntimeError(o)
        state[i].zero_()
        active[i] = {"obs": o, "done_attempts": 0}

    for i in range(min(n, len(queue))):
        start(i)
    while active:
        ids = sorted(active)
        a, _, _, new_state = actor.act([active[i]["obs"] for i in ids], state[ids], greedy=greedy)
        for i, ai in zip(ids, a, strict=True):
            pool.conns[i].send(("step", int(ai)))
        state[ids] = new_state
        for i in ids:
            status, payload = pool.conns[i].recv()
            if status == "error":
                raise RuntimeError(payload)
            o, _r, done, _final, episode, _lv = payload
            active[i]["obs"] = o
            if episode is not None:
                results.append(episode)
                active[i]["done_attempts"] += 1
            if done or active[i]["done_attempts"] >= attempts:
                del active[i]
                if queue:
                    start(i)
    return results


def summarize_trials(results: list[dict], attempts: int) -> dict:
    out = {}
    for k in range(attempts):
        eps = [e for e in results if e.get("attempt") == k]
        if eps:
            out[f"attempt{k + 1}/progress"] = float(np.mean([e["progress"] for e in eps]))
            out[f"attempt{k + 1}/completion"] = float(np.mean([e["flag_get"] for e in eps]))
    return out


def train_recurrent(cfg, env_cfg: EnvConfig) -> Path:
    """PPO with a recurrent policy over trials (requires env_cfg.trial_episodes > 0)."""
    import json
    from datetime import datetime

    from .ppo import Logger, RewardNormalizer, resolve_device, save_checkpoint

    if env_cfg.trial_episodes < 1:
        raise ValueError("recurrent training needs EnvConfig.trial_episodes >= 1")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads or 1)
    name = cfg.run_name or f"rl2-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(cfg.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"ppo": asdict(cfg), "env": env_cfg.to_dict()}, indent=2))
    obs_type, obs_shape, n_actions = observation_spec(env_cfg)
    device = resolve_device(cfg.device)
    model = RecurrentActorCritic(obs_type, obs_shape, n_actions, arch=cfg.arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    train_levels = parse_levels(env_cfg.levels)
    eval_levels = parse_levels(cfg.eval_levels)
    sampler = LevelSampler(len(train_levels), cfg.level_sampler if cfg.level_sampler != "plr" else "uniform")
    normalizer = RewardNormalizer(cfg.num_envs, cfg.gamma)
    batch = cfg.num_envs * cfg.num_steps
    num_updates = cfg.total_steps // batch
    logger = Logger(run_dir)
    # Workers only emulate here (actions are picked in this process); their CPU model copy is unused.
    from .models import ActorCritic

    dummy = ActorCritic(obs_type, obs_shape, n_actions)
    dummy.share_memory()
    pool = WorkerPool(cfg.num_envs, env_cfg, dummy, cfg.seed)
    actor = RecurrentActor(model)
    state = model.initial_state(cfg.num_envs, device)
    next_eval = cfg.eval_interval or None
    next_save = cfg.save_interval
    global_step = 0
    attempts = env_cfg.trial_episodes
    print(f"run dir: {run_dir} (recurrent PPO over trials of {attempts} attempts, on {device})", flush=True)
    try:
        for update in range(1, num_updates + 1):
            t0 = time.time()
            if cfg.anneal_lr:
                optimizer.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1) / num_updates)
            weights = sampler.weights() if cfg.level_sampler == "completion" else None
            buf, state0, next_values, episodes = rollout_recurrent(pool, actor, cfg.num_steps, state, weights)
            global_step += batch

            class _Ro:  # adapter for RewardNormalizer
                def __init__(self, r, d):
                    self.rewards, self.dones = r, d

            ros = [_Ro(buf["rewards"][i], buf["dones"][i]) for i in range(cfg.num_envs)]
            rewards = normalizer(ros) if cfg.norm_reward else [r.rewards for r in ros]
            adv = np.stack(
                [
                    _gae(rewards[i] + cfg.gamma * buf["bootstrap"][i], buf["values"][i], buf["dones"][i],
                         next_values[i], cfg.gamma, cfg.gae_lambda)
                    for i in range(cfg.num_envs)
                ]
            )  # fmt: skip
            obs_t = (
                to_torch({k: np.stack([o[k] for o in buf["obs"]]) for k in buf["obs"][0]})
                if isinstance(buf["obs"][0], dict)
                else to_torch(np.stack(buf["obs"]))
            )
            obs_t = (
                {k: v.to(device) for k, v in obs_t.items()} if isinstance(obs_t, dict) else obs_t.to(device)
            )
            T = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
            returns = adv + buf["values"]
            stats = ppo_update_recurrent(
                model, optimizer, cfg, obs_t, T(buf["actions"]), T(buf["logprobs"]), T(returns), T(adv),
                T(buf["resets"]), state0,
            )  # fmt: skip
            if cfg.level_sampler == "completion":
                for ep in episodes:
                    if ep.get("attempt", 0) == 0 and ep.get("level_id", -1) >= 0:
                        sampler.update_completion(ep["level_id"], bool(ep["flag_get"]))
            elapsed = time.time() - t0
            metrics = {**stats, "lr": optimizer.param_groups[0]["lr"], "sps": batch / elapsed}
            if episodes:
                metrics["train/return"] = float(np.mean([e["r"] for e in episodes]))
                metrics["train/progress"] = float(np.mean([e["progress"] for e in episodes]))
                metrics["train/completion"] = float(np.mean([e["flag_get"] for e in episodes]))
                metrics.update({f"train/{k}": v for k, v in summarize_trials(episodes, attempts).items()})
            logger.log(global_step, metrics)
            if update % 10 == 0 or update == 1:
                a1, ak = (
                    metrics.get("train/attempt1/progress", float("nan")),
                    metrics.get(f"train/attempt{attempts}/progress", float("nan")),
                )
                print(
                    f"step {global_step:>9,d} | sps {metrics['sps']:6.0f} | progress attempt1 {a1:6.1%} "
                    f"attempt{attempts} {ak:6.1%} | ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f}",
                    flush=True,
                )
            if next_eval is not None and global_step >= next_eval:
                next_eval += cfg.eval_interval
                record = {"step": global_step}
                for group, levels in (("test", eval_levels), ("train", train_levels)):
                    s = summarize_trials(
                        evaluate_trials(pool, actor, levels, cfg.eval_episodes, attempts), attempts
                    )
                    record[group] = s
                    logger.log(global_step, {f"eval_{group}/{k}": v for k, v in s.items()})
                with open(run_dir / "eval.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
                state.zero_()  # evaluation used the workers: restart training trials cleanly
                print(f"eval @ {global_step:,d}: {record['test']}", flush=True)
            if global_step >= next_save or update == num_updates:
                next_save += cfg.save_interval
                for path in ([run_dir / f"ckpt_{global_step}.pt"] if update != num_updates else []) + [
                    run_dir / "latest.pt"
                ]:
                    save_checkpoint(
                        path, model, optimizer, sampler, normalizer, cfg, env_cfg, global_step, update
                    )
    finally:
        pool.close()
        logger.close()
    return run_dir


def load_recurrent(path) -> tuple[RecurrentActorCritic, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ckpt["model_spec"]
    model = RecurrentActorCritic(
        spec["obs_type"], spec["obs_shape"], spec["n_actions"], spec["hidden"], spec["arch"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
