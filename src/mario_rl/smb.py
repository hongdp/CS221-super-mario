"""Super Mario Bros. game logic: RAM decoding, level select and cut-scene skips.

RAM addresses follow the community SMB disassembly. The level-select and
cut-scene skipping tricks are adapted from ``gym-super-mario-bros`` (MIT).
All readers take a RAM snapshot (``np.uint8`` array) and use explicit Python
integer arithmetic so they are NumPy 2 safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .emulator import START, NESEmulator
from .levels import Level

# --- RAM addresses ---------------------------------------------------------
PLAYER_STATE = 0x000E  # GameEngineSubroutine: 0x08 normal, 0x0B dying, 0x06 dead
PLAYER_FLOAT_STATE = 0x001D  # 0 ground, 1 jump, 2 fall, 3 climbing (flagpole)
PLAYER_PAGE = 0x006D
PLAYER_X = 0x0086
PLAYER_Y = 0x00CE
PLAYER_Y_VIEWPORT = 0x00B5  # 1 on screen, 0 above, >1 fell into a pit
PLAYER_SIZE = 0x0754  # 0 big, 1 small
PLAYER_STATUS = 0x0756  # 0 small, 1 big, 2 fire
WORLD = 0x075F
STAGE = 0x075C
AREA = 0x0760
OPER_MODE = 0x0770  # 1 playing, 2 victory (castle axe)
PRELEVEL_TIMER = 0x07A0
CHANGE_AREA_TIMER = 0x06DE
SCORE = 0x07DE  # 6 BCD digits
COINS = 0x07ED  # 2 BCD digits
TIME = 0x07F8  # 3 BCD digits
SCREEN_LEFT_PAGE = 0x071A
SCREEN_LEFT_X = 0x071C
ENEMY_ACTIVE = 0x000F  # 5 enemy slots
ENEMY_TYPE = 0x0016
ENEMY_PAGE = 0x006E
ENEMY_X = 0x0087
ENEMY_Y = 0x00CF
ENEMY_Y_VIEWPORT = 0x00B6
BLOCK_BUFFER = 0x0500  # 2 pages x 13 rows x 16 columns of metatiles

STATE_FLAGPOLE = 0x04
STATE_DEAD = 0x06
STATE_DYING = 0x0B
# Cut-scene states with no player control (entrance, vine, pipes, flagpole,
# end-of-level walk); we fast-forward through them.
BUSY_STATES = frozenset({0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07})

ENEMY_BOWSER = 0x2D
ENEMY_FLAGPOLE_FLAG = 0x31


def bcd(ram: np.ndarray, address: int, length: int) -> int:
    value = 0
    for digit in ram[address : address + length]:
        value = value * 10 + int(digit)
    return value


def x_pos(ram: np.ndarray) -> int:
    return int(ram[PLAYER_PAGE]) * 256 + int(ram[PLAYER_X])


def screen_left(ram: np.ndarray) -> int:
    return int(ram[SCREEN_LEFT_PAGE]) * 256 + int(ram[SCREEN_LEFT_X])


def game_time(ram: np.ndarray) -> int:
    return bcd(ram, TIME, 3)


def is_dead(ram: np.ndarray) -> bool:
    return int(ram[PLAYER_STATE]) in (STATE_DYING, STATE_DEAD) or int(ram[PLAYER_Y_VIEWPORT]) > 1


def flag_get(ram: np.ndarray) -> bool:
    # Touching the castle axe switches the game into victory mode.
    if ram[OPER_MODE] == 2 or ram[PLAYER_STATE] == STATE_FLAGPOLE:
        return True
    if int(ram[PLAYER_FLOAT_STATE]) != 3:
        return False
    enemies = ram[ENEMY_TYPE : ENEMY_TYPE + 5].tolist()
    return ENEMY_FLAGPOLE_FLAG in enemies or ENEMY_BOWSER in enemies


def is_busy(ram: np.ndarray) -> bool:
    return int(ram[PLAYER_STATE]) in BUSY_STATES


@dataclass(frozen=True)
class GameInfo:
    world: int
    stage: int
    x_pos: int
    time: int
    status: int
    coins: int
    score: int
    flag_get: bool
    dead: bool

    @classmethod
    def from_ram(cls, ram: np.ndarray) -> GameInfo:
        return cls(
            world=int(ram[WORLD]) + 1,
            stage=int(ram[STAGE]) + 1,
            x_pos=x_pos(ram),
            time=game_time(ram),
            status=int(ram[PLAYER_STATUS]),
            coins=bcd(ram, COINS, 2),
            score=bcd(ram, SCORE, 6),
            flag_get=flag_get(ram),
            dead=is_dead(ram),
        )


class SMBGame:
    """Owns the emulator and an in-memory save state at the start of each level."""

    MAX_SKIP_FRAMES = 3000

    def __init__(self, rom_path: str | Path | None = None):
        self.emu = NESEmulator(rom_path)
        self._states: dict[Level, bytes] = {}

    def level_state(self, level: Level) -> bytes:
        if level not in self._states:
            self._states[level] = self._boot_into(level)
        return self._states[level]

    def load(self, level: Level) -> None:
        self.emu.set_state(self.level_state(level))

    def ram(self) -> np.ndarray:
        return self.emu.ram

    def frame(self, buttons: int) -> None:
        self.emu.step(buttons)

    def skip_cutscenes(self) -> int:
        """Fast-forward pipe/vine/area-change animations. Returns frames skipped."""
        emu = self.emu
        timer = emu.read(CHANGE_AREA_TIMER)
        if 1 < timer < 255:
            emu.write(CHANGE_AREA_TIMER, 1)
        skipped = 0
        ram = emu.ram
        while is_busy(ram) and not flag_get(ram) and not is_dead(ram):
            emu.write(PRELEVEL_TIMER, 0)
            emu.step(0)
            skipped += 1
            if skipped >= self.MAX_SKIP_FRAMES:
                break
            ram = emu.ram
        return skipped

    def _boot_into(self, level: Level) -> bytes:
        """Power on, select ``level`` through RAM writes, return a save state."""
        emu = self.emu
        emu.power_cycle()

        def write_level() -> None:
            emu.write(WORLD, level.world - 1)
            emu.write(STAGE, level.stage - 1)
            emu.write(AREA, level.area - 1)

        def started() -> bool:
            # RAM powers up as 0xFF in FCEUmm, so also require "playing" mode
            # and a sane BCD clock before trusting the timer.
            ram = emu.ram
            return int(ram[OPER_MODE]) == 1 and 0 < game_time(ram) <= 999

        emu.step(START)
        emu.step(0)
        for _ in range(10_000):
            if started():
                break
            emu.step(START)
            write_level()
            emu.step(0)
            emu.write(PRELEVEL_TIMER, 0)
        else:
            raise RuntimeError(f"could not start level {level}")
        # idle until the in-game clock starts ticking
        time_last = game_time(emu.ram)
        for _ in range(10_000):
            now = game_time(emu.ram)
            if now < time_last:
                break
            time_last = now
            emu.step(START)
            emu.step(0)
        ram = emu.ram
        if (int(ram[WORLD]) + 1, int(ram[STAGE]) + 1) != (level.world, level.stage):
            raise RuntimeError(f"level select failed for {level}")
        return emu.get_state()

    def close(self) -> None:
        self.emu.close()
