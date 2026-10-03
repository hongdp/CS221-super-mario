"""Low-level NES emulator access through stable-retro's libretro FCEUmm core.

We use ``stable_retro.RetroEmulator`` directly (not ``retro.make``) because we
need raw RAM reads/writes and in-memory save states for every level, and we
do not depend on stable-retro's ROM hash check or bundled integration data.

Note: libretro cores are process-global, so only one emulator may exist per
process. Vectorised training therefore runs one emulator per worker process.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

RAM_SIZE = 0x800

# Button bitmask used throughout the package.
A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = (1 << i for i in range(8))
BUTTONS = {
    "A": A,
    "B": B,
    "select": SELECT,
    "start": START,
    "up": UP,
    "down": DOWN,
    "left": LEFT,
    "right": RIGHT,
}
# stable-retro's NES button order (see stable_retro/cores/fceumm.json).
_RETRO_ORDER = ("B", None, "select", "start", "up", "down", "left", "right", "A")

_REPO_ROM = Path(__file__).resolve().parents[2] / "roms" / "super-mario-bros.nes"


def find_rom(path: str | Path | None = None) -> Path:
    """Locate the Super Mario Bros. ROM.

    Order: explicit ``path`` -> ``$MARIO_RL_ROM`` -> ``roms/super-mario-bros.nes``
    in the repository checkout.
    """
    candidates = [path, os.environ.get("MARIO_RL_ROM"), _REPO_ROM]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"Super Mario Bros. ROM not found. Pass rom_path=..., set $MARIO_RL_ROM, or place it at {_REPO_ROM}."
    )


def _button_array(mask: int) -> np.ndarray:
    return np.array([bool(name and mask & BUTTONS[name]) for name in _RETRO_ORDER], dtype=np.uint8)


_MASKS = [_button_array(m) for m in range(256)]


class NESEmulator:
    """A single NES console. Only one instance may be alive per process."""

    def __init__(self, rom_path: str | Path | None = None):
        import stable_retro  # imported lazily: it loads native code

        rom = find_rom(rom_path)
        header = rom.read_bytes()[:4]
        if header != b"NES\x1a":
            raise ValueError(f"{rom} is not an iNES ROM")
        self._em = stable_retro.RetroEmulator(str(rom))
        self._data = stable_retro.data.GameData()
        self._em.configure_data(self._data)
        self._mem = self._data.memory
        self._buttons = -1
        self._boot_state = self._em.get_state()

    @property
    def ram(self) -> np.ndarray:
        """Snapshot (copy) of the 2 KiB console RAM."""
        blocks = self._mem.blocks
        return np.frombuffer(blocks[0] + blocks[0x400], dtype=np.uint8)

    def read(self, address: int) -> int:
        return int(self._mem.extract(address, "|u1"))

    def write(self, address: int, value: int) -> None:
        self._mem.assign(address, "|u1", int(value) & 0xFF)

    @property
    def screen(self) -> np.ndarray:
        """Current frame, (224, 240, 3) uint8 RGB (FCEUmm crops the overscan)."""
        return self._em.get_screen()

    def step(self, buttons: int = 0) -> None:
        """Advance one frame while holding ``buttons`` (bitmask of BUTTONS)."""
        if buttons != self._buttons:
            self._em.set_button_mask(_MASKS[buttons], 0)
            self._buttons = buttons
        self._em.step()

    def get_state(self) -> bytes:
        return self._em.get_state()

    def set_state(self, state: bytes) -> None:
        self._em.set_state(state)

    def power_cycle(self) -> None:
        self._em.set_state(self._boot_state)

    def close(self) -> None:
        # Dropping every reference releases the process-global core.
        self._mem = None
        self._data = None
        self._em = None
