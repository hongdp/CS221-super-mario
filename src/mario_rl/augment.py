"""Observation augmentation for regularizing the policy (DrAC, Raileanu et al. 2021)."""

from __future__ import annotations

import torch


def random_shift(obs, max_shift: int):
    """Shift every sample's grid horizontally by its own random offset in [-max_shift, max_shift].

    Works on (B, stack, H, W) tensors and on dict observations (only ``grid`` is shifted; Mario's
    state vector is translation invariant). All frames of a stack move together, and the exposed
    edge repeats the border column, so no fake pit or wall appears at the screen edge.
    """
    if isinstance(obs, dict):
        return {**obs, "grid": random_shift(obs["grid"], max_shift)}
    b, w = obs.shape[0], obs.shape[-1]
    shift = torch.randint(-max_shift, max_shift + 1, (b, 1), device=obs.device)
    cols = (torch.arange(w, device=obs.device)[None] - shift).clamp_(0, w - 1)  # (B, W)
    index = cols.view(b, *([1] * (obs.dim() - 2)), w).expand_as(obs)
    return obs.gather(-1, index)
