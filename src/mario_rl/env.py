"""Gymnasium environment for multi-level Super Mario Bros."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np

from . import smb
from .emulator import DOWN, LEFT, RIGHT, UP, A, B
from .levels import TRAIN_LEVELS, Level, parse_levels
from .objects import ENTITY_FEATURES, TERRAIN_FEATURES, EntityTracker, terrain_profile
from .tiles import (
    COLS,
    COLS2,
    EMPTY2,
    HAZARD2,
    MARIO2,
    NUM_FEATURES,
    NUM_TILE_TYPES,
    NUM_TILE_TYPES2,
    PLATFORM2,
    ROWS,
    ROWS2,
    SOLID2,
    STOMPABLE2,
    mario_features,
    tile_grid,
    tile_grid_v2,
)

ACTION_SETS: dict[str, tuple[int, ...]] = {
    "right_only": (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B),
    # Same as gym-super-mario-bros' SIMPLE_MOVEMENT.
    "simple": (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT),
    # simple + DOWN: needed to enter the pipes of the 8-4 maze.
    "simple_down": (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT, DOWN),
    # Same as gym-super-mario-bros' COMPLEX_MOVEMENT.
    "complex": (
        0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT,
        LEFT | A, LEFT | B, LEFT | A | B, DOWN, UP,
    ),
}  # fmt: skip

PIXEL_SIZE = 84
CEILING_ROUTE_LEVELS = frozenset({"1-2", "4-2"})  # ceiling runs lead into a warp zone
MAZE_LEVELS = frozenset({"4-4", "7-4", "8-4"})  # wrong paths loop back
_NOISE_BLOCK = 0x61  # hard block metatile: solid in the game and in the grid, cannot be hit


@dataclass
class EnvConfig:
    levels: tuple[str, ...] = tuple(str(lv) for lv in TRAIN_LEVELS)
    # tiles: 13x16 grid, 16px cells (2017 representation); tiles2: 26x32 grid of 8px cells built
    # from collision boxes with stompable/hazard classes, plus a Mario state vector; pixels: 84x84.
    # objects: Mario-relative terrain profile + nearby-entity list + state vector (see objects.py).
    obs: Literal["tiles", "tiles2", "pixels", "objects"] = "tiles2"
    actions: str = "simple"
    frame_skip: int = 4
    frame_stack: int = 4
    noop_max: int = 30  # random idle frames after reset, varies enemy timing
    # Probability that a frame repeats the previous frame's buttons instead of the
    # chosen ones (Machado et al., 2018); breaks memorized open-loop action sequences.
    sticky_prob: float = 0.0
    max_episode_steps: int = 3000
    stuck_steps: int = 250  # truncate when the furthest x has not improved for this long
    # reward = scale * (dx - time_penalty * clock_ticks + death/flag terms)
    reward_scale: float = 0.1
    time_penalty: float = 1.0
    death_penalty: float = 25.0
    flag_reward: float = 50.0
    # Egocentric tiles2 window: keep only the columns from ego_back cells behind Mario to ego_ahead
    # cells ahead of him (full height). 0/0 keeps the whole screen. Less of the level is visible, so
    # the network has less layout to memorize and must act on what is near Mario.
    ego_back: int = 0
    ego_ahead: int = 0
    # objects observation: terrain columns (16 px) behind / ahead of Mario, and entity slots
    obj_back: int = 1
    obj_ahead: int = 7
    obj_slots: int = 12
    # Mid-level starts: training resets (those without an explicit level option, so never
    # evaluation) load a random archived state of the chosen level with this probability. The
    # archive holds one state per (area, mid_start_bucket px) reached on the ground. A candidate
    # is kept only once Mario has survived mid_start_confirm more agent steps after it.
    # In 1-2 and 4-2 Mario can run on top of the ceiling past the exit pipe into the warp zone.
    # It pays more x than the exit, but it is a dead end here (warp pipes need DOWN and lead to
    # other worlds), so agents learn to get stuck there. When enabled, in those levels running
    # above the screen earns no progress reward, and staying up there (ceiling_steps net agent
    # steps) ends the episode like a death.
    block_ceiling_route: bool = True
    # Swimming strokes trigger on a new A press, and the frame skip turns consecutive A actions
    # into one long hold (one stroke), so Mario sinks unless the agent alternates A and no-A.
    # When enabled, A is released on the last frame of every agent step while swimming.
    swim_tap: bool = True
    ceiling_steps: int = 8
    mid_start_prob: float = 0.0
    mid_start_levels: tuple[str, ...] = ()  # restrict mid-level starts to these levels (empty = all)
    # Maze castles teleport Mario back on a wrong path, and walking the loop again would be paid
    # again. When enabled, progress there is paid only for new ground: x beyond the furthest x
    # reached in the current area (correct pipe warps forward count, loops do not).
    maze_new_ground_reward: bool = True
    # Procedural level perturbation for training (goal B): as each 16 px column scrolls into view,
    # with this probability add a solid block, either a floating platform 3-4 blocks above the
    # ground or a 1-block step on it. The grid and the game's collisions both read the block
    # buffer, so the agent sees and collides with the new block. Only empty cells with headroom
    # are used. Like mid-level starts, this never applies to resets with an explicit level (evaluation).
    terrain_noise: float = 0.0
    mid_start_bucket: int = 256
    mid_start_confirm: int = 30
    rom_path: str | None = None
    level_weights: tuple[float, ...] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_spec(config: EnvConfig) -> tuple[str, Any, int]:
    """(obs type, obs shape, number of actions) without starting an emulator.

    The shape is a tuple, or a dict of tuples for dict observations (``tiles2``).
    """
    if config.obs == "tiles":
        shape = (config.frame_stack, ROWS, COLS)
    elif config.obs == "tiles2":
        cols = config.ego_back + config.ego_ahead if config.ego_ahead else COLS2
        shape = {"grid": (config.frame_stack, ROWS2, cols), "vec": (NUM_FEATURES,)}
    elif config.obs == "pixels":
        shape = (config.frame_stack, PIXEL_SIZE, PIXEL_SIZE)
    elif config.obs == "objects":
        shape = {
            "terrain": (config.obj_back + config.obj_ahead, TERRAIN_FEATURES),
            "ents": (config.obj_slots, ENTITY_FEATURES),
            "vec": (NUM_FEATURES,),
        }
    else:
        raise ValueError(f"unknown observation type {config.obs!r}")
    if config.actions not in ACTION_SETS:
        raise ValueError(f"unknown action set {config.actions!r}; choose from {sorted(ACTION_SETS)}")
    return config.obs, shape, len(ACTION_SETS[config.actions])


class MarioEnv(gym.Env):
    """One emulator that plays a (weighted) random level from ``levels`` each episode.

    Observations stack the last ``frame_stack`` frames: the symbolic tile grid
    (``tiles``: ``(stack, 13, 16)``), the 8px grid plus Mario's state vector
    (``tiles2``: ``{"grid": (stack, 26, 32), "vec": (10,)}``) or grayscale
    ``84x84`` pixels.
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
        if config.ego_ahead and obs_type != "tiles2":
            raise ValueError("ego window (ego_back/ego_ahead) is implemented for obs='tiles2' only")
        self._buttons = ACTION_SETS[config.actions]
        self.action_space = gym.spaces.Discrete(n_actions)
        if obs_type == "tiles2":
            self.observation_space = gym.spaces.Dict(
                {
                    "grid": gym.spaces.Box(0, NUM_TILE_TYPES2 - 1, shape["grid"], np.uint8),
                    "vec": gym.spaces.Box(-10.0, 10.0, shape["vec"], np.float32),
                }
            )
            frame_shape = (config.frame_stack, ROWS2, COLS2)  # full screen; cropped in _obs if ego
        elif obs_type == "objects":
            self.observation_space = gym.spaces.Dict(
                {
                    k: gym.spaces.Box(-10.0, 10.0, v, np.float32) for k, v in shape.items()
                }  # features are clipped
            )
            frame_shape = (config.frame_stack, ROWS2, COLS2)  # tiles2 grid kept for analysis tools
            self._entities = EntityTracker(config.obj_slots)
        else:
            high = NUM_TILE_TYPES - 1 if obs_type == "tiles" else 255
            self.observation_space = gym.spaces.Box(0, high, shape, np.uint8)
            frame_shape = shape
        self._frames = np.zeros(frame_shape, dtype=np.uint8)
        self._weights = self._normalise(config.level_weights)
        self._archive: dict[Level, dict[tuple[int, int], bytes]] = {}  # mid-level start states
        self._pending: tuple[tuple[int, int], bytes, int] | None = None
        self._mid_start = False
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
        self._mid_start = False
        archive = self._archive.get(self.level)
        if (
            "level" not in options
            and archive
            and self._mid_start_allowed()
            and self.np_random.random() < self.config.mid_start_prob
        ):
            keys = list(archive)
            self._game.emu.set_state(archive[keys[int(self.np_random.integers(len(keys)))]])
            self._mid_start = True
        self._pending = None
        self._noise_active = self.config.terrain_noise > 0 and "level" not in options
        self._noise_col: int | None = None
        # Idle frames vary enemy timing at a level start. After a mid-level load they could carry
        # Mario (e.g. on a moving lift) into a situation the archived state never survived.
        noops = 0 if self._mid_start else int(self.np_random.integers(0, self.config.noop_max + 1))
        for _ in range(noops):
            self._game.frame(0)
        ram = self._game.ram()
        self._reset_episode_state(ram)
        self._frames[:] = self._observe(ram)
        return self._obs(ram), self._info(ram)

    def step(self, action: int):
        cfg = self.config
        game = self._game
        buttons = self._buttons[int(action)]
        dx_total = 0
        ticks = 0
        dead = flag = False
        ram = None
        ceiling_level = cfg.block_ceiling_route and str(self.level) in CEILING_ROUTE_LEVELS
        maze_level = cfg.maze_new_ground_reward and str(self.level) in MAZE_LEVELS
        for k in range(cfg.frame_skip):
            if cfg.sticky_prob and self.np_random.random() < cfg.sticky_prob:
                pressed = self._last_buttons
            else:
                pressed = buttons
            self._last_buttons = pressed
            if cfg.swim_tap and k == cfg.frame_skip - 1 and ram is not None and ram[smb.SWIMMING]:
                pressed &= ~A  # end the press, so the next A action is a fresh stroke
            game.frame(pressed)
            ram = game.ram()
            if smb.is_busy(ram) and not smb.flag_get(ram) and not smb.is_dead(ram):
                game.skip_cutscenes()  # pipes, vines, area changes
                ram = game.ram()
                self._x_last = smb.x_pos(ram)
            x = smb.x_pos(ram)
            dx = x - self._x_last
            # ignore teleports (pipes, sub-areas); on a ceiling-route level, running above the
            # screen earns nothing, so the dead-end ceiling route never outpays the exit pipe
            if abs(dx) <= 8 and not maze_level and not (ceiling_level and ram[smb.PLAYER_Y_VIEWPORT] == 0):
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
        new_ground = 0
        if maze_level:
            area = int(ram[smb.AREA])
            best = self._area_max.setdefault(area, x)  # entering an area pays nothing by itself
            new_ground = min(max(x - best, 0), 512)  # cap: a pipe warp pays well, a glitch cannot explode
            self._area_max[area] = max(best, x)
            dx_total = new_ground
        if self.config.mid_start_prob and self._mid_start_allowed():
            self._update_archive(ram, x, alive=not dead)
        if self._noise_active and not dead and not flag:
            self._perturb_terrain(ram, x)
        if x > self._max_x or new_ground > 0:
            self._max_x = max(self._max_x, x)
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        if ceiling_level:
            above = ram[smb.PLAYER_Y_VIEWPORT] == 0 and not dead and not flag
            # bobbing back onto the screen while running on the ceiling only decrements the count
            self._steps_above = self._steps_above + 1 if above else max(self._steps_above - 1, 0)
            if self._steps_above >= cfg.ceiling_steps:
                dead = True  # out of bounds on the ceiling: treated as a death

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
                "mid_start": self._mid_start,
            }
        return self._obs(ram), float(reward), terminated, truncated, info

    def render(self):
        return self._game.emu.screen.copy()

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    # --- helpers ---------------------------------------------------------------
    def _reset_episode_state(self, ram: np.ndarray | None = None) -> None:
        if hasattr(self, "_entities"):
            self._entities.reset()
        self._steps_above = 0
        self._area_max: dict[int, int] = {}
        self._last_buttons = 0
        self._steps = 0
        self._return = 0.0
        self._flag = False
        self._steps_since_progress = 0
        self._x_last = smb.x_pos(ram) if ram is not None else 0
        self._max_x = self._x_last
        self._time_last = smb.game_time(ram) if ram is not None else 0

    def _observe(self, ram: np.ndarray) -> np.ndarray:
        """The newest frame of the stack."""
        if self.config.obs == "tiles":
            return tile_grid(ram)
        if self.config.obs in ("tiles2", "objects"):
            return tile_grid_v2(ram)
        return _downsample(self._game.emu.screen)

    def _obs(self, ram: np.ndarray):
        if self.config.obs == "tiles2":
            grid = self._frames.copy() if not self.config.ego_ahead else self._ego_window(ram)
            return {"grid": grid, "vec": mario_features(ram)}
        if self.config.obs == "objects":  # called exactly once per reset / step (entity velocities)
            return {
                "terrain": terrain_profile(ram, self.config.obj_back, self.config.obj_ahead),
                "ents": self._entities(ram),
                "vec": mario_features(ram),
            }
        return self._frames.copy()

    def _perturb_terrain(self, ram: np.ndarray, x: int) -> None:
        """Maybe add a block to every column that has entered the right edge of the screen."""
        if smb.is_busy(ram):
            return
        right = (smb.screen_left(ram) + 255) // 16  # rightmost visible column: already rendered
        if self._noise_col is None or not right - 16 <= self._noise_col <= right + 1:
            self._noise_col = right + 1  # first step or an area change: start at the next column
            return
        last = min(right, (self.level.length - 160) // 16)  # leave the flagpole / axe area alone
        emu = self._game.emu
        while self._noise_col <= last:
            col = self._noise_col
            self._noise_col += 1
            if abs(col * 16 - x) < 48 or self.np_random.random() >= self.config.terrain_noise:
                continue
            base = smb.BLOCK_BUFFER + ((col // 16) & 1) * 0xD0 + (col % 16)
            column = [int(ram[base + 16 * r]) for r in range(13)]
            if column[12] == 0:
                continue  # a pit: leave it
            ground = 12
            while ground > 0 and column[ground - 1] != 0:
                ground -= 1  # top of the solid run that starts at the bottom row
            if self.np_random.random() < 0.5:
                row = ground - int(self.np_random.integers(3, 5))  # floating platform
                free = range(max(row - 1, 0), ground)
            else:
                row = ground - 1  # 1-block step
                free = range(max(ground - 4, 0), ground)
            if row >= 1 and all(column[r] == 0 for r in free):
                emu.write(base + 16 * row, _NOISE_BLOCK)

    def _mid_start_allowed(self) -> bool:
        return not self.config.mid_start_levels or str(self.level) in self.config.mid_start_levels

    def _update_archive(self, ram: np.ndarray, x: int, alive: bool) -> None:
        """Confirm or drop the pending candidate, then propose a new one for an unseen bucket."""
        if self._pending is not None:
            if not alive:
                self._pending = None
            else:
                key, state, left = self._pending
                self._pending = (key, state, left - 1) if left > 1 else None
                if left <= 1:
                    self._archive.setdefault(self.level, {}).setdefault(key, state)
            return
        key = (int(ram[smb.AREA]), x // self.config.mid_start_bucket)
        if (
            alive
            and key[1] > 0
            and key not in self._archive.get(self.level, {})
            and ram[smb.PLAYER_FLOAT_STATE] == 0  # standing or running on the ground
            and ram[smb.PLAYER_Y_VIEWPORT] == 1
            and not smb.is_busy(ram)
            and _safe_start(ram)
        ):
            self._pending = (key, self._game.emu.get_state(), self.config.mid_start_confirm)

    def _ego_window(self, ram: np.ndarray) -> np.ndarray:
        """Crop every stacked frame to the columns around Mario's current position."""
        back, ahead = self.config.ego_back, self.config.ego_ahead
        col = int(np.clip(smb.x_pos(ram) - smb.screen_left(ram), 0, 255)) // 8  # Mario's left cell
        stack, rows, cols = self._frames.shape
        padded = np.empty((stack, rows, back + cols + ahead), dtype=self._frames.dtype)
        padded[..., :back] = SOLID2  # the left screen edge is a wall in SMB
        padded[..., back : back + cols] = self._frames
        padded[..., back + cols :] = EMPTY2  # beyond the right edge: not yet visible
        return padded[..., col : col + back + ahead]  # window [col - back, col + ahead) shifted by back

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


def _safe_start(ram: np.ndarray) -> bool:
    """No lift/platform around Mario and no enemy or hazard right next to him.

    States on lifts are fragile to restart from: a lift's timing relative to Mario is easy to lose.
    """
    grid = tile_grid_v2(ram)
    _, cols = np.nonzero(grid == MARIO2)
    if not len(cols):
        return False
    c = int(cols.min())
    if (grid[:, max(c - 4, 0) : c + 12] == PLATFORM2).any():
        return False
    near = grid[:, max(c - 4, 0) : c + 6]
    return not ((near == STOMPABLE2) | (near == HAZARD2)).any()


_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _downsample(screen: np.ndarray) -> np.ndarray:
    gray = screen.astype(np.float32) @ _GRAY  # (224, 240)
    h, w = gray.shape
    pooled = gray.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))  # (112, 120)
    rows = np.linspace(0, pooled.shape[0] - 1, PIXEL_SIZE).round().astype(int)
    cols = np.linspace(0, pooled.shape[1] - 1, PIXEL_SIZE).round().astype(int)
    return pooled[np.ix_(rows, cols)].astype(np.uint8)
