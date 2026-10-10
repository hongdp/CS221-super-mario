"""Rollout workers: one emulator per process.

stable-retro allows a single emulator per process. Two ways to pick actions:

- ``WorkerPool.rollout``: each worker holds a reference to the learner's
  shared-memory CPU policy and collects a whole rollout segment locally, so the
  learner exchanges one message per worker per PPO iteration. Best without a GPU,
  where per-step IPC (Gymnasium's AsyncVectorEnv) costs more than the emulator.
- ``WorkerPool.rollout_batched``: the learner steps all workers in lock-step and
  runs one batched forward pass per step on its own device (GPU). Workers only
  emulate, so a larger network no longer slows down the CPU side.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from .env import EnvConfig, MarioEnv
from .inference import BatchedActor
from .models import ActorCritic
from .obs import allocate, store, to_torch


@dataclass
class Rollout:
    obs: Any  # (T, *obs_shape) array, or a dict of such arrays
    actions: np.ndarray  # (T,) int64
    logprobs: np.ndarray  # (T,) float32
    values: np.ndarray  # (T,) float32
    rewards: np.ndarray  # (T,) float32, raw environment rewards
    bootstrap: np.ndarray  # (T,) float32, V(final obs) where an episode was truncated, else 0
    dones: np.ndarray  # (T,) bool, episode ended after step t
    level_ids: np.ndarray  # (T,) int16
    next_value: float
    episodes: list[dict[str, Any]] = field(default_factory=list)


def _level_id(env: MarioEnv) -> int:
    """Index of the current level in the training levels; -1 for a level outside them (evaluation)."""
    return env.levels.index(env.level) if env.level in env.levels else -1


def _run_episode(env: MarioEnv, model: ActorCritic, level, greedy: bool, gen) -> dict:
    obs, _ = env.reset(options={"level": level})
    while True:
        action, _, _ = model.act(to_torch(obs, batch=True), greedy=greedy, generator=gen)
        obs, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return {"level": info["level"], **info["episode"]}


def _worker(index: int, conn, env_config: dict, model: ActorCritic, seed: int):
    torch.set_num_threads(1)
    try:
        env = MarioEnv(EnvConfig(**env_config))
        gen = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        conn.send(("ready", None))
        while True:
            command, payload = conn.recv()
            if command == "rollout":
                num_steps, weights = payload
                if weights is not None:
                    env.set_level_weights(weights)
                buf_obs = allocate(obs, num_steps)
                actions = np.empty(num_steps, dtype=np.int64)
                logprobs = np.empty(num_steps, dtype=np.float32)
                values = np.empty(num_steps, dtype=np.float32)
                rewards = np.empty(num_steps, dtype=np.float32)
                bootstrap = np.zeros(num_steps, dtype=np.float32)
                dones = np.empty(num_steps, dtype=bool)
                level_ids = np.empty(num_steps, dtype=np.int16)
                episodes = []
                with torch.inference_mode():
                    for t in range(num_steps):
                        level_ids[t] = _level_id(env)
                        store(buf_obs, t, obs)
                        action, logprob, value = model.act(to_torch(obs, batch=True), generator=gen)
                        obs, reward, terminated, truncated, info = env.step(int(action))
                        if truncated and not terminated:
                            # Time-limit / stuck truncation is not a real terminal
                            # state: bootstrap from the value of the final observation.
                            bootstrap[t] = float(model.get_value(to_torch(obs, batch=True)))
                        actions[t] = int(action)
                        logprobs[t] = float(logprob)
                        values[t] = float(value)
                        rewards[t] = reward
                        dones[t] = terminated or truncated
                        if dones[t]:
                            episodes.append({"level": info["level"], **info["episode"]})
                            obs, _ = env.reset()
                    next_value = float(model.get_value(to_torch(obs, batch=True)))
                conn.send(
                    (
                        "ok",
                        Rollout(
                            buf_obs, actions, logprobs, values, rewards, bootstrap, dones, level_ids,
                            next_value, episodes,
                        ),
                    )
                )  # fmt: skip
            elif command == "reset_level":  # trial evaluation: explicit level (and attempt), returns the obs
                obs, _ = env.reset(options=payload)
                conn.send(("ok", obs))
            elif command == "reset_to":  # switch level now (meta-learning tasks): set weights, new episode
                env.set_level_weights(payload)
                obs, _ = env.reset()
                conn.send(("ok", None))
            elif command == "begin":  # start a learner-driven (batched) rollout segment
                if payload is not None:
                    env.set_level_weights(payload)
                conn.send(("ok", (obs, _level_id(env))))
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(payload)
                # "done" for GAE (and recurrent-state resets) is the end of a trial; without trial mode that is
                # every episode end. Time-limit / stuck truncation bootstraps from the final observation's value.
                done = info["trial_end"]
                final_obs = obs if done and truncated and not terminated else None
                episode = None
                if terminated or truncated:
                    episode = {"level": info["level"], **info["episode"]}
                    obs, _ = (
                        env.reset()
                    )  # in trial mode: the next attempt of the same level until the trial ends
                level_id = _level_id(env)
                conn.send(("ok", (obs, float(reward), done, final_obs, episode, level_id)))
            elif command == "evaluate":
                jobs, greedy = payload
                with torch.inference_mode():
                    results = [_run_episode(env, model, level, greedy, gen) for level in jobs]
                obs, _ = env.reset()  # resume training on a fresh episode
                conn.send(("ok", results))
            elif command == "close":
                env.close()
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command {command!r}")
    except Exception:
        conn.send(("error", f"worker {index}:\n{traceback.format_exc()}"))


class WorkerPool:
    def __init__(self, num_workers: int, env_config: EnvConfig, model: ActorCritic, seed: int):
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        for i in range(num_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(i, child, env_config.to_dict(), model, seed + 1000 * i),
                daemon=True,
            )
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self._gather()

    def _gather(self) -> list:
        results = []
        for conn in self.conns:
            status, payload = conn.recv()
            if status == "error":
                self.close(force=True)
                raise RuntimeError(payload)
            results.append(payload)
        return results

    def rollout(self, num_steps: int, level_weights: np.ndarray | None = None) -> list[Rollout]:
        for conn in self.conns:
            conn.send(("rollout", (num_steps, level_weights)))
        return self._gather()

    def reset_all(self, level_weights: np.ndarray) -> None:
        """Start a fresh episode in every worker, sampling levels from ``level_weights`` from now on."""
        for conn in self.conns:
            conn.send(("reset_to", level_weights))
        self._gather()

    def rollout_batched(
        self, actor: BatchedActor, num_steps: int, level_weights: np.ndarray | None = None
    ) -> list[Rollout]:
        """Collect one segment per worker, picking actions on the learner's device.

        Asynchronous dynamic batching: each iteration runs one forward pass for whichever workers
        have reported a new observation, so a slow step (reset, cut-scene) in one worker does not
        stall the others. The policy is fixed for the whole segment, so this is exactly on-policy.
        """
        n = len(self.conns)
        for conn in self.conns:
            conn.send(("begin", level_weights))
        obs, level = map(list, zip(*self._gather(), strict=True))
        buf_obs = [allocate(o, num_steps) for o in obs]
        actions = np.empty((n, num_steps), dtype=np.int64)
        logprobs = np.empty((n, num_steps), dtype=np.float32)
        values = np.empty((n, num_steps), dtype=np.float32)
        rewards = np.empty((n, num_steps), dtype=np.float32)
        bootstrap = np.zeros((n, num_steps), dtype=np.float32)
        dones = np.empty((n, num_steps), dtype=bool)
        level_ids = np.empty((n, num_steps), dtype=np.int16)
        episodes: list[list[dict]] = [[] for _ in range(n)]
        truncated: list[tuple[int, int, Any]] = []  # (worker, step, final obs), valued at the end
        t = [0] * n
        index = {conn: i for i, conn in enumerate(self.conns)}
        ready, waiting = list(range(n)), set()
        while ready or waiting:
            if ready:
                a, lp, v = actor.act([obs[i] for i in ready])
                for j, i in enumerate(ready):
                    store(buf_obs[i], t[i], obs[i])
                    level_ids[i, t[i]] = level[i]
                    actions[i, t[i]], logprobs[i, t[i]], values[i, t[i]] = a[j], lp[j], v[j]
                    self.conns[i].send(("step", int(a[j])))
                waiting.update(ready)
                ready = []
            for conn in mp_connection.wait([self.conns[i] for i in waiting]):
                i = index[conn]
                status, payload = conn.recv()
                if status == "error":
                    self.close(force=True)
                    raise RuntimeError(payload)
                obs[i], rewards[i, t[i]], dones[i, t[i]], final, episode, level[i] = payload
                if final is not None:
                    truncated.append((i, t[i], final))
                if episode is not None:
                    episodes[i].append(episode)
                waiting.discard(i)
                t[i] += 1
                if t[i] < num_steps:
                    ready.append(i)
        if truncated:
            for (i, step, _), v in zip(truncated, actor.values([f for _, _, f in truncated]), strict=True):
                bootstrap[i, step] = v
        next_values = actor.values(obs)
        return [
            Rollout(
                buf_obs[i], actions[i], logprobs[i], values[i], rewards[i], bootstrap[i], dones[i],
                level_ids[i], float(next_values[i]), episodes[i],
            )
            for i in range(n)
        ]  # fmt: skip

    def evaluate(self, levels, episodes_per_level: int, greedy: bool = False) -> list[dict]:
        jobs = [lv for lv in levels for _ in range(episodes_per_level)]
        shards = [jobs[i :: len(self.conns)] for i in range(len(self.conns))]
        for conn, shard in zip(self.conns, shards, strict=True):
            conn.send(("evaluate", (shard, greedy)))
        return [r for shard in self._gather() for r in shard]

    def close(self, force: bool = False) -> None:
        if not force:
            for conn in self.conns:
                try:
                    conn.send(("close", None))
                    conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.conns, self.procs = [], []
