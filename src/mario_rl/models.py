"""Actor-critic networks for tile-grid and pixel observations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .tiles import NUM_TILE_TYPES


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class TileEncoder(nn.Module):
    """One-hot the categorical tile grid, then a small CNN.

    Input: (B, stack, 13, 16) uint8 with values in [0, NUM_TILE_TYPES).
    """

    def __init__(self, obs_shape: tuple[int, ...], hidden: int = 256):
        super().__init__()
        stack, rows, cols = obs_shape
        self.in_channels = stack * NUM_TILE_TYPES
        # Deliberately small (~1.8M MACs/sample): training runs on CPUs.
        self.net = nn.Sequential(
            _init(nn.Conv2d(self.in_channels, 16, 3, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, self.in_channels, rows, cols)).shape[1]
        self.fc = nn.Sequential(_init(nn.Linear(n_flat, hidden)), nn.ReLU())
        self.out_dim = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b, stack, rows, cols = obs.shape
        onehot = nn.functional.one_hot(obs.long(), NUM_TILE_TYPES)  # B,S,R,C,T
        x = onehot.permute(0, 1, 4, 2, 3).reshape(b, stack * NUM_TILE_TYPES, rows, cols)
        return self.fc(self.net(x.float()))


class PixelEncoder(nn.Module):
    """Nature-DQN CNN for (B, stack, 84, 84) uint8 grayscale frames."""

    def __init__(self, obs_shape: tuple[int, ...], hidden: int = 512):
        super().__init__()
        stack = obs_shape[0]
        self.net = nn.Sequential(
            _init(nn.Conv2d(stack, 32, 8, stride=4)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, *obs_shape)).shape[1]
        self.fc = nn.Sequential(_init(nn.Linear(n_flat, hidden)), nn.ReLU())
        self.out_dim = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(obs.float() / 255.0))


def _encoder(obs_type: str, obs_shape: tuple[int, ...]) -> nn.Module:
    if obs_type == "tiles":
        return TileEncoder(obs_shape)
    if obs_type == "pixels":
        return PixelEncoder(obs_shape)
    raise ValueError(f"unknown observation type {obs_type!r}")


class ActorCritic(nn.Module):
    """Policy and value networks.

    By default the actor and critic have separate encoders: with a shared trunk
    the (much larger) value-loss gradients dominate the features and the
    policy struggles to condition on the state; decoupling also helps
    generalization (Raileanu & Fergus, 2021, "Decoupling Value and Policy for
    Generalization in RL"). ``shared=True`` restores a single trunk.
    """

    def __init__(self, obs_type: str, obs_shape: tuple[int, ...], n_actions: int, shared: bool = False):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.shared = shared
        self.actor_encoder = _encoder(obs_type, self.obs_shape)
        self.critic_encoder = None if shared else _encoder(obs_type, self.obs_shape)
        self.policy = _init(nn.Linear(self.actor_encoder.out_dim, n_actions), gain=0.01)
        self.value = _init(nn.Linear(self.actor_encoder.out_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.actor_encoder(obs)
        h_v = h if self.critic_encoder is None else self.critic_encoder(obs)
        return self.policy(h), self.value(h_v).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        encoder = self.actor_encoder if self.critic_encoder is None else self.critic_encoder
        return self.value(encoder(obs)).squeeze(-1)

    def parameter_groups(self) -> list[list[nn.Parameter]]:
        """Parameter groups whose gradients are clipped independently."""
        if self.critic_encoder is None:
            return [list(self.parameters())]
        actor = [*self.actor_encoder.parameters(), *self.policy.parameters()]
        critic = [*self.critic_encoder.parameters(), *self.value.parameters()]
        return [actor, critic]

    def act(
        self, obs: torch.Tensor, greedy: bool = False, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) actions; returns (action, log-prob, value)."""
        logits, value = self(obs)
        log_probs = torch.log_softmax(logits, -1)
        if greedy:
            action = logits.argmax(-1)
        else:
            # Gumbel-max trick: exact categorical sampling, much cheaper than multinomial.
            u = torch.rand(log_probs.shape, generator=generator).clamp_(1e-10, 1.0)
            action = (log_probs - (-u.log()).log()).argmax(-1)
        return action, log_probs.gather(-1, action[:, None]).squeeze(-1), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def spec(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "shared": self.shared,
        }
