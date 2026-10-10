"""Object-centric observation: a Mario-relative terrain profile plus a list of nearby entities.

Instead of a grid of cells (whose layout is easy to memorize), the agent gets
- terrain: for each 16 px column from ``back`` columns behind Mario to ``ahead`` columns in front,
  whether the column is a pit, and the 3 walkable surfaces nearest to Mario's feet (height
  relative to his feet, in 16 px blocks);
- entities: the ``slots`` nearest objects (enemies, lifts, firebar segments, hammers) with their
  class, position and size relative to Mario, and their velocity since the previous step.
Mario's own state stays in ``vec`` (see ``tiles.mario_features``).
"""

from __future__ import annotations

import numpy as np

from . import smb
from .tiles import (
    _FIREBAR_TILE,
    _FIREBAR_TYPES,
    _HAMMER_TILES,
    _HAZARD_TYPES,
    _IGNORED_TYPES,
    _METATILE_CLASS,
    _PLATFORM_TYPES,
    SOLID,
)

TERRAIN_FEATURES = 7  # pit, then (dy, valid) for the 3 surfaces nearest to Mario's feet
ENTITY_FEATURES = 10  # present, dx, dy, w, h, vx, vy (relative to Mario, scaled), class one-hot: stompable / hazard / platform
STOMPABLE, HAZARD, PLATFORM = range(3)


def _solid_column(ram: np.ndarray, x: int) -> np.ndarray:
    """13 bools (top to bottom) for the 16 px column containing level x (must be in the block buffer)."""
    page, col = (x >> 8) & 1, (x & 0xFF) >> 4
    base = smb.BLOCK_BUFFER + page * 0xD0 + col
    return _METATILE_CLASS[ram[base : base + 13 * 16 : 16]] == SOLID


def terrain_profile(ram: np.ndarray, back: int, ahead: int) -> np.ndarray:
    """(back + ahead, TERRAIN_FEATURES) float32, columns ordered from behind Mario to ahead of him."""
    out = np.zeros((back + ahead, TERRAIN_FEATURES), dtype=np.float32)
    _x0, y0, _x1, y1 = (int(v) for v in ram[smb.PLAYER_BOX : smb.PLAYER_BOX + 4])
    mario_x = smb.x_pos(ram) + 8  # Mario's centre in level coordinates
    feet_row = (y1 - 32) // 16 if ram[smb.PLAYER_Y_VIEWPORT] == 1 and y1 > y0 else 11
    for i, k in enumerate(range(-back, ahead)):
        x = mario_x + 16 * k
        if x < 0:
            continue  # left of the level start: nothing there
        solid = _solid_column(ram, x)
        out[i, 0] = float(not solid[12])  # bottom row open: a pit
        surfaces = [r for r in range(13) if solid[r] and (r == 0 or not solid[r - 1])]
        surfaces.sort(key=lambda r: abs(r - feet_row))
        for j, r in enumerate(surfaces[:3]):
            out[i, 1 + 2 * j] = (r - feet_row) / 4.0  # > 0: below Mario's feet
            out[i, 2 + 2 * j] = 1.0
    return out


def _boxes(ram: np.ndarray) -> list[tuple[int, float, float, float, float, int]]:
    """(key, x0, y0, x1, y1, class) of every visible entity, in screen pixels."""
    left = smb.screen_left(ram)
    boxes = []
    kinds = []
    for slot in range(5):
        if not ram[smb.ENEMY_ACTIVE + slot] or ram[smb.ENEMY_Y_VIEWPORT + slot] != 1:
            continue
        kind = int(ram[smb.ENEMY_TYPE + slot])
        kinds.append(kind)
        if kind in _IGNORED_TYPES or kind in _FIREBAR_TYPES:
            continue
        x0, y0, x1, y1 = (int(v) for v in ram[smb.ENEMY_BOX + 4 * slot : smb.ENEMY_BOX + 4 * slot + 4])
        if x1 <= x0 or y1 <= y0:  # no collision box (e.g. jumpspring): use the sprite position
            x0 = int(ram[smb.ENEMY_PAGE + slot]) * 256 + int(ram[smb.ENEMY_X + slot]) - left
            y0 = int(ram[smb.ENEMY_Y + slot])
            x1, y1 = x0 + 16, y0 + 16
        cls = PLATFORM if kind in _PLATFORM_TYPES else HAZARD if kind in _HAZARD_TYPES else STOMPABLE
        boxes.append((slot * 256 + kind, x0, y0, x1, y1, cls))  # slot + type identify it across steps
    oam = ram[smb.OAM : smb.OAM + 256].reshape(64, 4)
    sprite_tiles = []
    if any(k in _FIREBAR_TYPES for k in kinds):
        sprite_tiles.append(oam[oam[:, 1] == _FIREBAR_TILE])
    if 0x05 in kinds or 0x2D in kinds:  # hammer bro / bowser hammers
        sprite_tiles.append(oam[np.isin(oam[:, 1], _HAMMER_TILES)])
    for sprites in sprite_tiles:
        for y, _t, _a, x in sprites:
            if y < 0xEF:
                boxes.append((-1, int(x), int(y) + 1, int(x) + 8, int(y) + 9, HAZARD))
    return boxes


class EntityTracker:
    """Builds the entity list and keeps the previous step's positions for velocities."""

    def __init__(self, slots: int):
        self.slots = slots
        self.prev: dict[int, tuple[float, float]] = {}  # enemy key -> centre
        self.prev_sprites: list[tuple[float, float]] = []  # sprite hazards (no identity)

    def reset(self) -> None:
        self.prev, self.prev_sprites = {}, []

    def __call__(self, ram: np.ndarray) -> np.ndarray:
        out = np.zeros((self.slots, ENTITY_FEATURES), dtype=np.float32)
        px0, py0, px1, py1 = (int(v) for v in ram[smb.PLAYER_BOX : smb.PLAYER_BOX + 4])
        mcx, mcy = (px0 + px1) / 2, (py0 + py1) / 2
        left = smb.screen_left(ram)  # velocities in level coordinates: an object property, scroll-free
        rows, prev, sprites = [], {}, []
        for key, x0, y0, x1, y1, cls in _boxes(ram):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            lx = cx + left
            if key >= 0:
                vx, vy = (lx - self.prev[key][0], cy - self.prev[key][1]) if key in self.prev else (0.0, 0.0)
                prev[key] = (lx, cy)
            else:  # sprite hazard: match the nearest one from the previous step
                near = min(self.prev_sprites, key=lambda p: (p[0] - lx) ** 2 + (p[1] - cy) ** 2, default=None)
                vx, vy = (
                    (lx - near[0], cy - near[1])
                    if near and abs(lx - near[0]) + abs(cy - near[1]) < 16
                    else (0, 0)
                )
                sprites.append((lx, cy))
            rows.append(
                ((cx - mcx) ** 2 + (cy - mcy) ** 2, cx - mcx, cy - mcy, x1 - x0, y1 - y0, vx, vy, cls)
            )
        self.prev, self.prev_sprites = prev, sprites
        rows.sort(key=lambda r: r[0])
        for i, (_d, dx, dy, w, h, vx, vy, cls) in enumerate(rows[: self.slots]):
            out[i, :7] = (1.0, dx / 64.0, dy / 64.0, w / 16.0, h / 16.0, vx / 8.0, vy / 8.0)
            out[i, 7 + cls] = 1.0
        return np.clip(out, -10.0, 10.0, out=out)  # e.g. a teleporting enemy cannot produce huge velocities
