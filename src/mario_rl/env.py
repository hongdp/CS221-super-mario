"""Gymnasium environment for multi-level Super Mario Bros."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np

from . import smb
from .emulator import DOWN, LEFT, RIGHT, UP, A, B
from .levels import TRAIN_LEVELS, Level, parse_levels
from .tiles import COLS, NUM_TILE_TYPES, ROWS, tile_grid

ACTION_SETS: dict[str, tuple[int, ...]] = {
    "right_only": (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B),
    # Same as gym-super-mario-bros' SIMPLE_MOVEMENT.
    "simple": (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT),
    # Same as gym-super-mario-bros' COMPLEX_MOVEMENT.
    "complex": (
        0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT,
        LEFT | A, LEFT | B, LEFT | A | B, DOWN, UP,
    ),
}  # fmt: skip

PIXEL_SIZE = 84


@dataclass
class EnvConfig:
    levels: tuple[str, ...] = tuple(str(lv) for lv in TRAIN_LEVELS)
    obs: Literal["tiles", "pixels"] = "tiles"
    actions: str = "simple"
    frame_skip: int = 4
    frame_stack: int = 4
    noop_max: int = 30  # random idle frames after reset, varies enemy timing
    max_episode_steps: int = 3000
    stuck_steps: int = 250  # truncate when the furthest x has not improved for this long
    # reward = scale * (dx - time_penalty * clock_ticks + death/flag terms)
    reward_scale: float = 0.1
    time_penalty: float = 1.0
    death_penalty: float = 25.0
    flag_reward: float = 50.0
    rom_path: str | None = None
    level_weights: tuple[float, ...] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_spec(config: EnvConfig) -> tuple[str, tuple[int, ...], int]:
    """(obs type, obs shape, number of actions) without starting an emulator."""
    if config.obs == "tiles":
        shape = (config.frame_stack, ROWS, COLS)
    elif config.obs == "pixels":
        shape = (config.frame_stack, PIXEL_SIZE, PIXEL_SIZE)
    else:
        raise ValueError(f"unknown observation type {config.obs!r}")
    if config.actions not in ACTION_SETS:
        raise ValueError(f"unknown action set {config.actions!r}; choose from {sorted(ACTION_SETS)}")
    return config.obs, shape, len(ACTION_SETS[config.actions])


class MarioEnv(gym.Env):
    """One emulator that plays a (weighted) random level from ``levels`` each episode.

    Observations are a stack of the last ``frame_stack`` frames, either as the
    symbolic tile grid (``(stack, 13, 16)`` with values in ``[0, 5)``) or as
    grayscale ``84x84`` pixels.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, **kwargs):
        config = config or EnvConfig()
        for key, value in kwargs.items():
            if not hasattr(config, key):
                raise TypeError(f"unknown EnvConfig field {key!r}")
            setattr(config, key, value)
        self.config = config
        self.render_mode = render_mode
        self.levels: tuple[Level, ...] = parse_levels(config.levels)
        obs_type, shape, n_actions = observation_spec(config)
        self._buttons = ACTION_SETS[config.actions]
        self.action_space = gym.spaces.Discrete(n_actions)
        high = NUM_TILE_TYPES - 1 if obs_type == "tiles" else 255
        self.observation_space = gym.spaces.Box(0, high, shape, np.uint8)
        self._frames = np.zeros(shape, dtype=np.uint8)
        self._weights = self._normalise(config.level_weights)
        self._game = smb.SMBGame(config.rom_path)
        self.level = self.levels[0]
        self._reset_episode_state()

    # --- level sampling ------------------------------------------------------
    def _normalise(self, weights) -> np.ndarray:
        if weights is None:
            return np.full(len(self.levels), 1.0 / len(self.levels))
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.levels),) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("level weights must be non-negative, one per level")
        return w / w.sum()

    def set_level_weights(self, weights) -> None:
        """Change the level sampling distribution (used by prioritized level replay)."""
        self._weights = self._normalise(weights)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "level" in options:
            level = options["level"]
            self.level = level if isinstance(level, Level) else Level.parse(level)
        else:
            idx = self.np_random.choice(len(self.levels), p=self._weights)
            self.level = self.levels[idx]
        self._game.load(self.level)
        for _ in range(int(self.np_random.integers(0, self.config.noop_max + 1))):
            self._game.frame(0)
        ram = self._game.ram()
        self._reset_episode_state(ram)
        frame = self._observe(ram)
        self._frames[:] = frame
        return self._frames.copy(), self._info(ram)

    def step(self, action: int):
        cfg = self.config
        game = self._game
        buttons = self._buttons[int(action)]
        dx_total = 0
        ticks = 0
        dead = flag = False
        ram = None
        for _ in range(cfg.frame_skip):
            game.frame(buttons)
            ram = game.ram()
            if smb.is_busy(ram) and not smb.flag_get(ram) and not smb.is_dead(ram):
                game.skip_cutscenes()  # pipes, vines, area changes
                ram = game.ram()
                self._x_last = smb.x_pos(ram)
            x = smb.x_pos(ram)
            dx = x - self._x_last
            if abs(dx) <= 8:  # ignore teleports (pipes, sub-areas)
                dx_total += dx
            self._x_last = x
            now = smb.game_time(ram)
            if now < self._time_last:
                ticks += self._time_last - now
            self._time_last = now
            dead = smb.is_dead(ram)
            flag = smb.flag_get(ram)
            if dead or flag:
                break

        self._steps += 1
        x = smb.x_pos(ram)
        if x > self._max_x:
            self._max_x = x
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        reward = dx_total - cfg.time_penalty * ticks
        if dead:
            reward -= cfg.death_penalty
        if flag:
            reward += cfg.flag_reward
            self._flag = True
        reward *= cfg.reward_scale
        self._return += reward

        terminated = dead or flag
        truncated = not terminated and (
            self._steps >= cfg.max_episode_steps or self._steps_since_progress >= cfg.stuck_steps
        )
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = self._observe(ram)
        info = self._info(ram)
        if terminated or truncated:
            info["episode"] = {
                "r": self._return,
                "l": self._steps,
                "progress": info["progress"],
                "flag_get": self._flag,
                "level_id": self.levels.index(self.level) if self.level in self.levels else -1,
            }
        return self._frames.copy(), float(reward), terminated, truncated, info

    def render(self):
        return self._game.emu.screen.copy()

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    # --- helpers ---------------------------------------------------------------
    def _reset_episode_state(self, ram: np.ndarray | None = None) -> None:
        self._steps = 0
        self._return = 0.0
        self._flag = False
        self._steps_since_progress = 0
        self._x_last = smb.x_pos(ram) if ram is not None else 0
        self._max_x = self._x_last
        self._time_last = smb.game_time(ram) if ram is not None else 0

    def _observe(self, ram: np.ndarray) -> np.ndarray:
        if self.config.obs == "tiles":
            return tile_grid(ram)
        return _downsample(self._game.emu.screen)

    def _info(self, ram: np.ndarray) -> dict[str, Any]:
        x = smb.x_pos(ram)
        progress = 1.0 if self._flag else min(self._max_x / self.level.length, 1.0)
        return {
            "level": str(self.level),
            "x_pos": x,
            "max_x": self._max_x,
            "progress": progress,
            "flag_get": self._flag,
            "time": smb.game_time(ram),
        }


_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _downsample(screen: np.ndarray) -> np.ndarray:
    gray = screen.astype(np.float32) @ _GRAY  # (224, 240)
    h, w = gray.shape
    pooled = gray.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))  # (112, 120)
    rows = np.linspace(0, pooled.shape[0] - 1, PIXEL_SIZE).round().astype(int)
    cols = np.linspace(0, pooled.shape[1] - 1, PIXEL_SIZE).round().astype(int)
    return pooled[np.ix_(rows, cols)].astype(np.uint8)
