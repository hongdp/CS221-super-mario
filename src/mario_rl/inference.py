"""Batched action selection on the learner's device, with CUDA graphs on a GPU.

A batch of a few observations is far too small to keep a GPU busy: an eager forward pass is
dominated by kernel-launch overhead (~0.7 ms for TileEncoderV2, ~2.2 ms for the IMPALA encoder).
Replaying a captured CUDA graph cuts that by about 4x. One graph is captured lazily per batch
size. The graphs read the parameters in place, so they stay valid across optimizer steps.
"""

from __future__ import annotations

import numpy as np
import torch

from .models import ActorCritic
from .obs import Obs, stack, to_torch


def _sample(policy: ActorCritic, obs) -> torch.Tensor:
    """(3, B) tensor of [action, log-prob, value]: a single device-to-host copy per call."""
    logits, value = policy(obs)
    log_probs = torch.log_softmax(logits, -1)
    u = torch.rand_like(log_probs).clamp_(1e-10, 1.0)  # Gumbel-max sampling, as in ActorCritic.act
    action = (log_probs - (-u.log()).log()).argmax(-1)
    return torch.stack([action.float(), log_probs.gather(-1, action[:, None]).squeeze(-1), value])


class BatchedActor:
    def __init__(self, policy: ActorCritic):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.use_graphs = self.device.type == "cuda"
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, Obs, torch.Tensor]] = {}

    def _to_device(self, obs: Obs):
        return (
            {k: v.to(self.device) for k, v in obs.items()} if isinstance(obs, dict) else obs.to(self.device)
        )

    def _graph(self, example):
        batch = len(next(iter(example.values()))) if isinstance(example, dict) else len(example)
        if batch not in self.graphs:
            static = self._to_device(example)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                with torch.cuda.stream(stream):  # warm-up outside the capture
                    for _ in range(2):
                        _sample(self.policy, static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = _sample(self.policy, static)
            self.graphs[batch] = (graph, static, out)
        return self.graphs[batch]

    def act(self, items: list[Obs]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions for a list of single observations: (actions, log-probs, values)."""
        obs = to_torch(stack(items))
        if self.use_graphs:
            graph, static, out = self._graph(obs)
            if isinstance(static, dict):
                for k in static:
                    static[k].copy_(obs[k], non_blocking=True)
            else:
                static.copy_(obs, non_blocking=True)
            graph.replay()
            result = out.cpu().numpy()
        else:
            with torch.no_grad():
                result = _sample(self.policy, self._to_device(obs)).cpu().numpy()
        return result[0].astype(np.int64), result[1], result[2]

    def values(self, items: list[Obs]) -> np.ndarray:
        with torch.no_grad():
            return self.policy.get_value(self._to_device(to_torch(stack(items)))).cpu().numpy()
