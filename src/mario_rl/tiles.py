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
