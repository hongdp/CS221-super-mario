"""Level sampling: uniform or Prioritized Level Replay (Jiang et al., 2021).

PLR replays levels where the agent's value predictions are most wrong (large
average |GAE|), mixed with a staleness term so every level is revisited.
"""

from __future__ import annotations

import numpy as np


class LevelSampler:
    def __init__(
        self,
        num_levels: int,
        strategy: str = "uniform",
        temperature: float = 0.1,
        staleness_coef: float = 0.1,
    ):
        if strategy not in ("uniform", "plr"):
            raise ValueError(f"unknown level sampling strategy {strategy!r}")
        self.num_levels = num_levels
        self.strategy = strategy
        self.temperature = temperature
        self.staleness_coef = staleness_coef
        self.scores = np.zeros(num_levels)
        self.seen = np.zeros(num_levels, dtype=bool)
        self.last_update = np.zeros(num_levels)
        self.updates = 0

    def update(self, level_id: int, score: float) -> None:
        self.updates += 1
        self.scores[level_id] = score
        self.seen[level_id] = True
        self.last_update[level_id] = self.updates

    def weights(self) -> np.ndarray:
        n = self.num_levels
        if self.strategy == "uniform":
            return np.full(n, 1.0 / n)
        if not self.seen.all():
            # Visit every level once before prioritizing.
            unseen = (~self.seen).astype(np.float64)
            return unseen / unseen.sum()
        order = np.argsort(-self.scores, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        score_w = (1.0 / ranks) ** (1.0 / self.temperature)
        score_w /= score_w.sum()
        staleness = self.updates - self.last_update
        stale_w = staleness / staleness.sum() if staleness.sum() > 0 else np.full(n, 1.0 / n)
        return (1 - self.staleness_coef) * score_w + self.staleness_coef * stale_w

    def state_dict(self) -> dict:
        return {
            "scores": self.scores.tolist(),
            "seen": self.seen.tolist(),
            "last_update": self.last_update.tolist(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.scores = np.asarray(state["scores"], dtype=np.float64)
        self.seen = np.asarray(state["seen"], dtype=bool)
        self.last_update = np.asarray(state["last_update"], dtype=np.float64)
        self.updates = int(state["updates"])
