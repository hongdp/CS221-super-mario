"""Symbolic 13x16 tile view of the screen, decoded from RAM.

This is the same idea as the 2017 project's ``Tiles`` observation, but rebuilt
from the game's block buffer and object tables. Because it discards colours and
background scenery, it looks the same in overworld, underground, water and
castle levels, which is exactly what we want for cross-level generalization.
"""

from __future__ import annotations

import numpy as np

from . import smb

ROWS, COLS = 13, 16
EMPTY, SOLID, ENEMY, PLATFORM, MARIO = range(5)
NUM_TILE_TYPES = 5
TILE_NAMES = ("empty", "solid", "enemy", "platform", "mario")

# Metatile -> class. The block buffer only holds foreground metatiles; scenery
# such as clouds and bushes never appears in it.
_METATILE_CLASS = np.full(256, SOLID, dtype=np.uint8)
_METATILE_CLASS[0x00] = EMPTY
_METATILE_CLASS[[0x24, 0x25, 0x26]] = EMPTY  # flagpole ball, flagpole, vine
_METATILE_CLASS[[0x5F, 0x60]] = EMPTY  # hidden coin / 1-up blocks
_METATILE_CLASS[[0xC2, 0xC3]] = EMPTY  # coins

# Enemy-slot object types (SMB disassembly names in comments).
_PLATFORM_TYPES = frozenset(range(0x24, 0x2D)) | {0x32}  # lifts, jumpspring
_IGNORED_TYPES = frozenset(
    {
        0x16,  # fireworks
        0x17,  # bullet bill / cheep-cheep frenzy controller (invisible)
        0x18,  # stop-frenzy controller (invisible)
        0x2E,  # power-up
        0x2F,  # vine
        0x30,  # flagpole flag
        0x31,  # star flag on castle
        0x35,  # toad / princess
    }
)
_BOWSER = 0x2D

_COL_OFFSETS = 8 + 16 * np.arange(COLS)
_ROW_INDEX = np.arange(ROWS)[:, None]


def _mark(grid: np.ndarray, row: int, col: int, value: int) -> None:
    if 0 <= row < ROWS and 0 <= col < COLS:
        grid[row, col] = value


def tile_grid(ram: np.ndarray) -> np.ndarray:
    """Return a (13, 16) uint8 grid of tile classes for the visible screen."""
    left = smb.screen_left(ram)
    buffer = ram[smb.BLOCK_BUFFER : smb.BLOCK_BUFFER + 2 * ROWS * COLS].reshape(2, ROWS, COLS)
    xs = left + _COL_OFFSETS  # sample each screen cell at its centre
    pages = (xs >> 8) & 1
    cols = (xs & 0xFF) >> 4
    grid = _METATILE_CLASS[buffer[pages[None, :], _ROW_INDEX, cols[None, :]]]

    for slot in range(5):
        if not ram[smb.ENEMY_ACTIVE + slot] or ram[smb.ENEMY_Y_VIEWPORT + slot] != 1:
            continue
        kind = int(ram[smb.ENEMY_TYPE + slot])
        if kind in _IGNORED_TYPES:
            continue
        ex = int(ram[smb.ENEMY_PAGE + slot]) * 256 + int(ram[smb.ENEMY_X + slot]) - left
        ey = int(ram[smb.ENEMY_Y + slot])
        col, row = (ex + 8) // 16, (ey - 16) // 16
        if kind in _PLATFORM_TYPES:
            for dc in range(3):  # lifts are (at least) three tiles wide
                _mark(grid, (ey - 24) // 16, (ex + 8) // 16 + dc, PLATFORM)
        elif kind == _BOWSER:
            for dr in (-1, 0):
                for dc in (0, 1):
                    _mark(grid, row + dr, col + dc, ENEMY)
        else:
            _mark(grid, row, col, ENEMY)

    if ram[smb.PLAYER_Y_VIEWPORT] == 1:
        mx = smb.x_pos(ram) - left
        y = int(ram[smb.PLAYER_Y])
        col = (mx + 8) // 16
        _mark(grid, (y - 8) // 16, col, MARIO)
        if ram[smb.PLAYER_SIZE] == 0:  # big Mario is two tiles tall
            _mark(grid, (y - 24) // 16, col, MARIO)
    return grid


def render_ascii(grid: np.ndarray) -> str:
    chars = {EMPTY: ".", SOLID: "#", ENEMY: "E", PLATFORM: "=", MARIO: "M"}
    return "\n".join("".join(chars[int(v)] for v in row) for row in grid)


# --- v2: 8px grid, richer object classes, collision boxes and a state vector --------

ROWS2, COLS2 = 26, 32
EMPTY2, SOLID2, STOMPABLE2, HAZARD2, PLATFORM2, MARIO2 = range(6)
NUM_TILE_TYPES2 = 6
TILE_NAMES2 = ("empty", "solid", "stompable", "hazard", "platform", "mario")
NUM_FEATURES = 10
FEATURE_NAMES = (
    "x_speed", "y_speed", "on_ground", "jumping", "falling", "climbing",
    "big", "fire", "swimming", "star",
)  # fmt: skip

# Enemies that hurt on any contact (cannot be stomped).
_HAZARD_TYPES = frozenset(
    {
        0x07,  # blooper
        0x0A, 0x0B,  # swimming cheep-cheeps
        0x0C,  # podoboo
        0x0D,  # piranha plant
        0x12,  # spiny
        0x15,  # bowser flame
        0x1B, 0x1C, 0x1D, 0x1E, 0x1F,  # firebars
        0x2D,  # bowser
    }
)  # fmt: skip
_FIREBAR_TYPES = frozenset({0x1B, 0x1C, 0x1D, 0x1E, 0x1F})
_HAMMER_BRO = 0x05
_FIREBAR_TILE = 0x64  # OAM tile of a firebar segment
_GRID2_ROWS = np.arange(ROWS2)[:, None] // 2
_GRID2_XS = 4 + 8 * np.arange(COLS2)


def _mark_box(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    """Mark the 8px cells overlapped by a screen-space box [x0, x1) x [y0, y1)."""
    if x1 <= x0 or y1 <= y0:
        return
    c0, c1 = max(x0 // 8, 0), min((x1 - 1) // 8, COLS2 - 1)
    r0, r1 = max((y0 - 32) // 8, 0), min((y1 - 1 - 32) // 8, ROWS2 - 1)
    if c0 <= c1 and r0 <= r1:
        grid[r0 : r1 + 1, c0 : c1 + 1] = value


def _box(ram: np.ndarray, address: int) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in ram[address : address + 4])  # type: ignore[return-value]


def tile_grid_v2(ram: np.ndarray) -> np.ndarray:
    """(26, 32) grid of 8px cells with classes empty/solid/stompable/hazard/platform/mario."""
    left = smb.screen_left(ram)
    buffer = ram[smb.BLOCK_BUFFER : smb.BLOCK_BUFFER + 2 * ROWS * COLS].reshape(2, ROWS, COLS)
    xs = left + _GRID2_XS
    pages, cols = (xs >> 8) & 1, (xs & 0xFF) >> 4
    grid = _METATILE_CLASS[buffer[pages[None, :], _GRID2_ROWS, cols[None, :]]]

    kinds = []
    for slot in range(5):
        if not ram[smb.ENEMY_ACTIVE + slot] or ram[smb.ENEMY_Y_VIEWPORT + slot] != 1:
            continue
        kind = int(ram[smb.ENEMY_TYPE + slot])
        kinds.append(kind)
        if kind in _IGNORED_TYPES or kind in _FIREBAR_TYPES:
            continue
        x0, y0, x1, y1 = _box(ram, smb.ENEMY_BOX + 4 * slot)
        if x1 <= x0 or y1 <= y0:  # no collision box (e.g. jumpspring): use the sprite position
            x0 = int(ram[smb.ENEMY_PAGE + slot]) * 256 + int(ram[smb.ENEMY_X + slot]) - left
            y0 = int(ram[smb.ENEMY_Y + slot])
            x1, y1 = x0 + 16, y0 + 16
        if kind in _PLATFORM_TYPES:
            value = PLATFORM2
        elif kind in _HAZARD_TYPES:
            value = HAZARD2
        else:
            value = STOMPABLE2
        _mark_box(grid, x0, y0, x1, y1, value)

    if any(k in _FIREBAR_TYPES for k in kinds):  # firebar segments only exist as sprites
        oam = ram[smb.OAM : smb.OAM + 256].reshape(64, 4)
        for y, _tile, _attr, x in oam[oam[:, 1] == _FIREBAR_TILE]:
            if y < 0xEF:
                _mark_box(grid, int(x), int(y) + 1, int(x) + 8, int(y) + 9, HAZARD2)
    if _HAMMER_BRO in kinds:  # misc slots then hold hammers (otherwise jumping coins)
        for slot in range(9):
            if ram[smb.MISC_STATE + slot]:
                _mark_box(grid, *_box(ram, smb.MISC_BOX + 4 * slot), HAZARD2)

    if ram[smb.PLAYER_Y_VIEWPORT] == 1:
        _mark_box(grid, *_box(ram, smb.PLAYER_BOX), MARIO2)
    return grid


def _signed(value) -> int:
    value = int(value)
    return value - 256 if value > 127 else value


def mario_features(ram: np.ndarray) -> np.ndarray:
    """Mario's dynamics and power-up state; see FEATURE_NAMES."""
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    features[0] = _signed(ram[smb.PLAYER_X_SPEED]) / 40.0
    features[1] = _signed(ram[smb.PLAYER_Y_SPEED]) / 5.0
    float_state = int(ram[smb.PLAYER_FLOAT_STATE])
    if float_state < 4:
        features[2 + float_state] = 1.0
    status = int(ram[smb.PLAYER_STATUS])
    features[6] = float(status >= 1)
    features[7] = float(status == 2)
    features[8] = float(ram[smb.SWIMMING] != 0)
    features[9] = float(ram[smb.STAR_TIMER] > 0)
    return features


def render_ascii_v2(grid: np.ndarray) -> str:
    chars = {EMPTY2: ".", SOLID2: "#", STOMPABLE2: "e", HAZARD2: "X", PLATFORM2: "=", MARIO2: "M"}
    return "\n".join("".join(chars[int(v)] for v in row) for row in grid)
