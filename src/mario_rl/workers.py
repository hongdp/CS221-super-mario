"""Rollout workers: one emulator per process, policy inference inside the worker.

stable-retro allows a single emulator per process, and per-step IPC with a
central learner (Gymnasium's AsyncVectorEnv) costs more than the emulator
itself on small CPU machines. Each worker therefore holds a reference to the
learner's shared-memory policy and collects a whole rollout segment locally;
the learner only exchanges one message per worker per PPO iteration.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from .env import EnvConfig, MarioEnv
from .models import ActorCritic


@dataclass
class Rollout:
    obs: np.ndarray  # (T, *obs_shape) uint8
    actions: np.ndarray  # (T,) int64
    logprobs: np.ndarray  # (T,) float32
    values: np.ndarray  # (T,) float32
    rewards: np.ndarray  # (T,) float32, truncations already bootstrapped
    dones: np.ndarray  # (T,) bool, episode ended after step t
    level_ids: np.ndarray  # (T,) int16
    next_value: float
    episodes: list[dict[str, Any]] = field(default_factory=list)


def _run_episode(env: MarioEnv, model: ActorCritic, level, greedy: bool, gen) -> dict:
    obs, _ = env.reset(options={"level": level})
    while True:
        action, _, _ = model.act(torch.from_numpy(obs)[None], greedy=greedy, generator=gen)
        obs, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return {"level": info["level"], **info["episode"]}


def _worker(index: int, conn, env_config: dict, model: ActorCritic, seed: int, gamma: float):
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
                buf_obs = np.empty((num_steps, *obs.shape), dtype=np.uint8)
                actions = np.empty(num_steps, dtype=np.int64)
                logprobs = np.empty(num_steps, dtype=np.float32)
                values = np.empty(num_steps, dtype=np.float32)
                rewards = np.empty(num_steps, dtype=np.float32)
                dones = np.empty(num_steps, dtype=bool)
                level_ids = np.empty(num_steps, dtype=np.int16)
                episodes = []
                with torch.inference_mode():
                    for t in range(num_steps):
                        level_ids[t] = env.levels.index(env.level)
                        buf_obs[t] = obs
                        action, logprob, value = model.act(torch.from_numpy(obs)[None], generator=gen)
                        obs, reward, terminated, truncated, info = env.step(int(action))
                        if truncated and not terminated:
                            # Time-limit / stuck truncation is not a real terminal
                            # state: bootstrap from the value of the final observation.
                            reward += gamma * float(model.get_value(torch.from_numpy(obs)[None]))
                        actions[t] = int(action)
                        logprobs[t] = float(logprob)
                        values[t] = float(value)
                        rewards[t] = reward
                        dones[t] = terminated or truncated
                        if dones[t]:
                            episodes.append({"level": info["level"], **info["episode"]})
                            obs, _ = env.reset()
                    next_value = float(model.get_value(torch.from_numpy(obs)[None]))
                conn.send(
                    (
                        "ok",
                        Rollout(
                            buf_obs, actions, logprobs, values, rewards, dones, level_ids,
                            next_value, episodes,
                        ),
                    )
                )  # fmt: skip
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
    def __init__(self, num_workers: int, env_config: EnvConfig, model: ActorCritic, seed: int, gamma: float):
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        for i in range(num_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(i, child, env_config.to_dict(), model, seed + 1000 * i, gamma),
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
