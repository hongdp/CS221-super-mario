"""Super Mario Bros. reinforcement learning across many levels."""

import gymnasium as gym

from .env import EnvConfig, MarioEnv
from .levels import ALL_LEVELS, TEST_LEVELS, TRAIN_LEVELS, Level

__all__ = ["ALL_LEVELS", "TEST_LEVELS", "TRAIN_LEVELS", "EnvConfig", "Level", "MarioEnv"]
__version__ = "2.0.0"

gym.register(id="MarioRL/SuperMarioBros-v0", entry_point="mario_rl.env:MarioEnv")
