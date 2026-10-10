"""Proximal Policy Optimization over many Super Mario Bros. levels."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .augment import random_shift
from .env import EnvConfig, observation_spec
from .evaluate import format_table, summarize
from .inference import BatchedActor
from .levels import parse_levels
from .models import ActorCritic, flatten_obs
from .obs import batch_size, concat, take, to_device, to_torch
from .plr import LevelSampler
from .workers import Rollout, WorkerPool


@dataclass
class PPOConfig:
    run_name: str | None = None
    out_dir: str = "runs"
    seed: int = 1
    total_steps: int = 5_000_000
    num_envs: int = 4
    num_steps: int = 512
    num_minibatches: int = 4
    update_epochs: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5  # applied separately to the actor and the critic
    # DrAC: keep the policy and value unchanged under random horizontal grid shifts of up to
    # aug_shift cells (0 disables), weighted by aug_coef. The PPO loss itself uses the original obs.
    aug_shift: int = 0
    aug_coef: float = 0.1
    # Auxiliary forward model (obs="objects"): the actor's encoder also predicts the next observation
    # from (observation, action), weighted by aux_coef (0 disables). "Learn how the world works".
    aux_coef: float = 0.0
    # Reptile meta-learning (Nichol et al., 2018): every task = one training level for meta_inner_steps
    # env steps of PPO from the current meta-weights theta (fresh Adam moments); then
    # theta += meta_step_size * (adapted - theta). 0 disables. Checkpoints and evaluations hold theta.
    meta_inner_steps: int = 0
    meta_step_size: float = 0.3
    shared_encoder: bool = False
    arch: str = "default"  # encoder: "default" or "impala" (tiles2 only), see models.ARCHS
    norm_reward: bool = True  # divide rewards by the running std of discounted returns
    level_sampler: str = "uniform"  # "uniform", "plr" or "completion"
    plr_temperature: float = 0.1
    plr_staleness: float = 0.1
    eval_interval: int = 250_000  # env steps between evaluations (0 disables)
    eval_levels: str = "test"
    eval_on_train: bool = True
    eval_episodes: int = 2
    eval_greedy: bool = False
    save_interval: int = 500_000
    torch_threads: int = 0  # 0 = all cores
    # Where the PPO update runs: "auto" (cuda if available), "cpu" or "cuda". Rollout workers always
    # act on CPU with the shared-memory copy of the policy.
    device: str = "auto"
    # Who picks actions during rollouts: "worker" (each worker runs the CPU copy of the policy),
    # "batched" (the learner steps all workers and runs one batched forward pass on its device),
    # or "auto" (batched when the learner is on a GPU).
    inference: str = "auto"
    resume: str | None = None
    # Start a new run from the weights of this checkpoint (fresh optimizer, sampler and step
    # counter), e.g. to fine-tune a pretrained policy on a few new levels.
    init_from: str | None = None


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """GAE for one trajectory segment; ``dones[t]`` means the episode ended after step t."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def plr_scores(adv: np.ndarray, dones: np.ndarray, level_ids: np.ndarray):
    """Yield (level_id, mean |advantage|) for each episode segment in a rollout."""
    start = 0
    for t in range(len(adv)):
        if dones[t] or t == len(adv) - 1:
            yield int(level_ids[start]), float(np.abs(adv[start : t + 1]).mean())
            start = t + 1


class RunningMeanStd:
    """Running variance of a stream of scalars (parallel Welford update)."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        b_mean, b_var, b_count = float(np.mean(x)), float(np.var(x)), len(x)
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


class RewardNormalizer:
    """Scale rewards by the std of the discounted return, tracked per env stream."""

    def __init__(self, num_envs: int, gamma: float):
        self.gamma = gamma
        self.returns = np.zeros(num_envs)
        self.rms = RunningMeanStd()

    def __call__(self, rollouts) -> list[np.ndarray]:
        streams = []
        for i, ro in enumerate(rollouts):
            ret = np.empty(len(ro.rewards))
            acc = self.returns[i]
            for t, (r, done) in enumerate(zip(ro.rewards, ro.dones, strict=True)):
                acc = acc * self.gamma + r
                ret[t] = acc
                if done:
                    acc = 0.0
            self.returns[i] = acc
            streams.append(ret)
        self.rms.update(np.concatenate(streams))
        scale = np.sqrt(self.rms.var + 1e-8)
        return [(ro.rewards / scale).astype(np.float32) for ro in rollouts]

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var + 1e-8))


class Logger:
    def __init__(self, run_dir: Path):
        self.file = open(run_dir / "metrics.jsonl", "a")  # noqa: SIM115 (closed in close())
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(run_dir / "tb"))
        except ImportError:
            self.tb = None

    def log(self, step: int, metrics: dict) -> None:
        self.file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.file.flush()
        if self.tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(key, value, step)

    def close(self) -> None:
        self.file.close()
        if self.tb is not None:
            self.tb.close()


def save_checkpoint(path: Path, model, optimizer, sampler, normalizer, cfg, env_cfg, step, update) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_spec": model.spec(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "reward_rms": normalizer.rms.state_dict(),
            "ppo_config": asdict(cfg),
            "env_config": env_cfg.to_dict(),
            "global_step": step,
            "update": update,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_policy(path: str | Path) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ckpt["model_spec"]
    model = ActorCritic(
        spec["obs_type"],
        spec["obs_shape"],
        spec["n_actions"],
        spec.get("shared", False),
        spec.get("arch", "default"),
        spec.get("aux", False),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def train(cfg: PPOConfig, env_cfg: EnvConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads or os.cpu_count() or 1)

    ckpt = None
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        run_dir = Path(cfg.resume).resolve().parent
    else:
        name = cfg.run_name or f"{env_cfg.obs}-{cfg.level_sampler}-{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = Path(cfg.out_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps({"ppo": asdict(cfg), "env": env_cfg.to_dict()}, indent=2)
        )

    obs_type, obs_shape, n_actions = observation_spec(env_cfg)
    model = ActorCritic(
        obs_type, obs_shape, n_actions, shared=cfg.shared_encoder, arch=cfg.arch, aux=cfg.aux_coef > 0
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    elif cfg.init_from:
        source = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if source["model_spec"] != model.spec():
            raise ValueError(f"--init-from {cfg.init_from}: model {source['model_spec']} != {model.spec()}")
        model.load_state_dict(source["model"])
    model.share_memory()  # workers read the learner's weights directly
    # On a GPU the update runs on a separate copy whose weights are copied back after every update.
    device = resolve_device(cfg.device)
    learner = model if device.type == "cpu" else copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=cfg.lr, eps=1e-5)
    train_levels = parse_levels(env_cfg.levels)
    sampler = LevelSampler(len(train_levels), cfg.level_sampler, cfg.plr_temperature, cfg.plr_staleness)
    normalizer = RewardNormalizer(cfg.num_envs, cfg.gamma)
    global_step, start_update = 0, 1
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        sampler.load_state_dict(ckpt["sampler"])
        if "reward_rms" in ckpt:
            normalizer.rms.load_state_dict(ckpt["reward_rms"])
        global_step, start_update = ckpt["global_step"], ckpt["update"] + 1

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_steps // batch_size
    eval_levels = parse_levels(cfg.eval_levels)
    logger = Logger(run_dir)
    pool = WorkerPool(cfg.num_envs, env_cfg, model, cfg.seed)
    batched = cfg.inference == "batched" or (cfg.inference == "auto" and device.type != "cpu")
    actor = BatchedActor(learner) if batched else None
    print(f"run dir: {run_dir} (PPO update on {device}, {'batched' if batched else 'in-worker'} inference)")
    print(f"training on {len(train_levels)} levels: {' '.join(map(str, train_levels))}")
    print(f"held-out eval levels: {' '.join(map(str, eval_levels))}")

    next_eval = (global_step // cfg.eval_interval + 1) * cfg.eval_interval if cfg.eval_interval else None
    next_save = (global_step // cfg.save_interval + 1) * cfg.save_interval
    last_eval_step = None

    def eval_and_save(update: int) -> None:
        nonlocal next_eval, next_save, last_eval_step
        if next_eval is not None and global_step >= next_eval:
            next_eval += cfg.eval_interval
            logger.log(
                global_step, run_evaluation(pool, cfg, train_levels, eval_levels, run_dir, global_step)
            )
            last_eval_step = global_step
        if global_step >= next_save:
            next_save += cfg.save_interval
            for path in (run_dir / f"ckpt_{global_step}.pt", run_dir / "latest.pt"):
                save_checkpoint(
                    path, model, optimizer, sampler, normalizer, cfg, env_cfg, global_step, update
                )

    meta = cfg.meta_inner_steps > 0
    meta_k = max(1, cfg.meta_inner_steps // batch_size)  # PPO updates per task
    theta: list[torch.Tensor] | None = None
    task_weights = None
    meta_rng = np.random.default_rng(cfg.seed)

    def meta_outer_step() -> None:
        """theta += eps * (adapted - theta); the learner (and the workers' copy) then hold theta."""
        if theta is None:
            return
        with torch.no_grad():
            for param, start in zip(learner.parameters(), theta, strict=True):
                param.copy_(start + cfg.meta_step_size * (param - start))
        _sync(model, learner)

    try:
        for update in range(start_update, num_updates + 1):
            t0 = time.time()
            if cfg.anneal_lr:
                optimizer.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1) / num_updates)
            if meta and (update - start_update) % meta_k == 0:  # task boundary
                meta_outer_step()
                if theta is not None:
                    eval_and_save(update - 1)  # evaluations and checkpoints see theta, not adapted weights
                theta = [param.detach().clone() for param in learner.parameters()]
                optimizer.state.clear()  # fresh Adam moments for the new task
                task_weights = np.eye(len(train_levels))[meta_rng.integers(len(train_levels))]
                pool.reset_all(task_weights)
            plr_weights = sampler.weights() if cfg.level_sampler != "uniform" else None
            weights = task_weights if meta else plr_weights
            if batched:
                rollouts: list[Rollout] = pool.rollout_batched(actor, cfg.num_steps, weights)
            else:
                rollouts = pool.rollout(cfg.num_steps, weights)
            t_rollout = time.time() - t0
            global_step += batch_size

            rewards = normalizer(rollouts) if cfg.norm_reward else [ro.rewards for ro in rollouts]
            advantages = []
            for ro, rew in zip(rollouts, rewards, strict=True):
                # truncated episodes bootstrap from V(final obs), already in value units
                rew = rew + cfg.gamma * ro.bootstrap
                adv = compute_gae(rew, ro.values, ro.dones, ro.next_value, cfg.gamma, cfg.gae_lambda)
                advantages.append(adv)
                if cfg.level_sampler == "plr":
                    for level_id, score in plr_scores(adv, ro.dones, ro.level_ids):
                        sampler.update(level_id, score)
            b_obs = to_device(to_torch(concat([ro.obs for ro in rollouts])), device)
            b_target = b_tmask = None
            if cfg.aux_coef:
                b_target, b_tmask = _next_obs_targets(rollouts, device)
            b_actions = torch.from_numpy(np.concatenate([ro.actions for ro in rollouts])).to(device)
            b_logprobs = torch.from_numpy(np.concatenate([ro.logprobs for ro in rollouts])).to(device)
            b_values = torch.from_numpy(np.concatenate([ro.values for ro in rollouts]))
            b_adv = torch.from_numpy(np.concatenate(advantages))
            b_returns = b_adv + b_values

            stats = _ppo_update(
                learner,
                optimizer,
                cfg,
                b_obs,
                b_actions,
                b_logprobs,
                b_returns.to(device),
                b_adv.to(device),
                minibatch_size,
                b_target,
                b_tmask,
            )
            _sync(model, learner)
            y_pred, y_true = b_values.numpy(), b_returns.numpy()
            var_y = np.var(y_true)
            stats["explained_variance"] = float(1 - np.var(y_true - y_pred) / var_y) if var_y > 0 else 0.0

            episodes = [ep for ro in rollouts for ep in ro.episodes]
            if cfg.level_sampler == "completion":
                for ep in episodes:
                    if not ep.get("mid_start") and ep.get("level_id", -1) >= 0:
                        sampler.update_completion(ep["level_id"], bool(ep["flag_get"]))
            elapsed = time.time() - t0
            metrics = {
                **stats,
                "lr": optimizer.param_groups[0]["lr"],
                "sps": batch_size / elapsed,
                "rollout_sps": batch_size / t_rollout,
                "reward_scale": normalizer.scale,
            }
            # progress / completion only count episodes that started at the beginning of a level
            full = [ep for ep in episodes if not ep.get("mid_start")]
            if episodes:
                metrics["train/episodes"] = len(episodes)
                metrics["train/mid_start_frac"] = 1.0 - len(full) / len(episodes)
            if full:
                metrics["train/return"] = float(np.mean([e["r"] for e in full]))
                metrics["train/progress"] = float(np.mean([e["progress"] for e in full]))
                metrics["train/completion"] = float(np.mean([e["flag_get"] for e in full]))
                metrics["train/length"] = float(np.mean([e["l"] for e in full]))
            logger.log(global_step, metrics)
            if update % 10 == 0 or update == start_update:
                print(
                    f"step {global_step:>9,d} | sps {metrics['sps']:6.0f} | "
                    f"return {metrics.get('train/return', float('nan')):7.2f} | "
                    f"progress {metrics.get('train/progress', float('nan')):6.1%} | "
                    f"done {metrics.get('train/completion', float('nan')):5.1%} | "
                    f"ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f}",
                    flush=True,
                )

            if not meta:
                eval_and_save(update)
        if meta:
            meta_outer_step()  # the final weights are meta-weights too
        save_checkpoint(
            run_dir / "latest.pt",
            model,
            optimizer,
            sampler,
            normalizer,
            cfg,
            env_cfg,
            global_step,
            num_updates,
        )
        if cfg.eval_interval and last_eval_step != global_step:
            logger.log(
                global_step, run_evaluation(pool, cfg, train_levels, eval_levels, run_dir, global_step)
            )
    finally:
        pool.close()
        logger.close()
    return run_dir


def _sync(model: ActorCritic, learner: ActorCritic) -> None:
    """Copy the learner's weights in place into the shared CPU model the workers act with."""
    if learner is not model:
        with torch.no_grad():
            for dst, src in zip(model.state_dict().values(), learner.state_dict().values(), strict=True):
                dst.copy_(src)


def _next_obs_targets(rollouts, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened next observation for every step, and a mask that is 0 where there is none in the same
    episode (the episode ended after the step, or it is the last step of the segment)."""
    targets, masks = [], []
    for ro in rollouts:
        flat = flatten_obs(to_torch(ro.obs))
        nxt = torch.zeros_like(flat)
        nxt[:-1] = flat[1:]
        mask = torch.from_numpy(~ro.dones).float()
        mask[-1] = 0.0
        targets.append(nxt)
        masks.append(mask)
    return torch.cat(targets).to(device), torch.cat(masks).to(device)


def _ppo_update(
    model,
    optimizer,
    cfg,
    b_obs,
    b_actions,
    b_logprobs,
    b_returns,
    b_adv,
    minibatch_size,
    b_target=None,
    b_tmask=None,
):
    n = batch_size(b_obs)
    clipfracs, approx_kls, pg_losses, v_losses, entropies = [], [], [], [], []
    aug_pi_losses, aug_v_losses, aux_losses = [], [], []
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=b_actions.device)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb_obs = take(b_obs, idx)
            logits, new_value = model(mb_obs)
            dist = Categorical(logits=logits)
            new_logprob, entropy = dist.log_prob(b_actions[idx]), dist.entropy()
            logratio = new_logprob - b_logprobs[idx]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kls.append(((ratio - 1) - logratio).mean().item())
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
            adv = b_adv[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            v_loss = 0.5 * (new_value - b_returns[idx]).pow(2).mean()
            ent = entropy.mean()
            loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
            if cfg.aug_shift:
                aug_logits, aug_value = model(random_shift(mb_obs, cfg.aug_shift))
                log_p = torch.log_softmax(logits.detach(), -1)
                aug_pi = (log_p.exp() * (log_p - torch.log_softmax(aug_logits, -1))).sum(-1).mean()
                aug_v = (aug_value - new_value.detach()).pow(2).mean()
                loss = loss + cfg.aug_coef * (aug_pi + aug_v)
                aug_pi_losses.append(aug_pi.item())
                aug_v_losses.append(aug_v.item())
            if b_target is not None:
                err = (model.predict_next(mb_obs, b_actions[idx]) - b_target[idx]).pow(2).mean(-1)
                m = b_tmask[idx]
                aux = (err * m).sum() / m.sum().clamp(min=1.0)
                loss = loss + cfg.aux_coef * aux
                aux_losses.append(aux.item())
            optimizer.zero_grad()
            loss.backward()
            for group in model.parameter_groups():
                nn.utils.clip_grad_norm_(group, cfg.max_grad_norm)
            optimizer.step()
            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent.item())
    stats = {
        "loss/policy": float(np.mean(pg_losses)),
        "loss/value": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "clipfrac": float(np.mean(clipfracs)),
    }
    if aug_pi_losses:
        stats["loss/aug_policy_kl"] = float(np.mean(aug_pi_losses))
        stats["loss/aug_value"] = float(np.mean(aug_v_losses))
    if aux_losses:
        stats["loss/aux_forward"] = float(np.mean(aux_losses))
    return stats


def run_evaluation(pool, cfg, train_levels, eval_levels, run_dir: Path, step: int) -> dict:
    t0 = time.time()
    metrics: dict = {}
    record = {"step": step}
    groups = [("test", eval_levels)]
    if cfg.eval_on_train:
        groups.append(("train", train_levels))
    for name, levels in groups:
        summary = summarize(pool.evaluate(levels, cfg.eval_episodes, cfg.eval_greedy))
        metrics[f"eval_{name}/progress"] = summary["progress"]
        metrics[f"eval_{name}/completion"] = summary["completion"]
        record[name] = summary
        print(format_table(summary, f"== eval on {name} levels @ {step:,d} steps =="), flush=True)
    with open(run_dir / "eval.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"evaluation took {time.time() - t0:.0f}s", flush=True)
    return metrics
