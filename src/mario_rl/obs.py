"""Helpers that treat array and dict observations uniformly."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

Obs = Any  # np.ndarray | dict[str, np.ndarray]


def to_torch(obs: Obs, batch: bool = False):
    """numpy -> torch without copying; ``batch=True`` adds a leading batch dimension."""
    if isinstance(obs, dict):
        return {k: to_torch(v, batch) for k, v in obs.items()}
    tensor = torch.from_numpy(np.asarray(obs))
    return tensor[None] if batch else tensor


def allocate(template: Obs, n: int) -> Obs:
    if isinstance(template, dict):
        return {k: allocate(v, n) for k, v in template.items()}
    return np.empty((n, *template.shape), dtype=template.dtype)


def store(buffer: Obs, index: int, obs: Obs) -> None:
    if isinstance(buffer, dict):
        for k in buffer:
            buffer[k][index] = obs[k]
    else:
        buffer[index] = obs


def concat(items: list[Obs]) -> Obs:
    if isinstance(items[0], dict):
        return {k: np.concatenate([o[k] for o in items]) for k in items[0]}
    return np.concatenate(items)


def take(obs, index):
    """Index the batch dimension of an array/tensor or of every entry of a dict."""
    if isinstance(obs, dict):
        return {k: v[index] for k, v in obs.items()}
    return obs[index]


def batch_size(obs) -> int:
    return len(next(iter(obs.values()))) if isinstance(obs, dict) else len(obs)
