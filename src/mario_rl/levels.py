"""Level definitions and train / held-out splits for generalization experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Level:
    world: int
    stage: int

    def __post_init__(self) -> None:
        if not (1 <= self.world <= 8 and 1 <= self.stage <= 4):
            raise ValueError(f"invalid level {self.world}-{self.stage}")

    @classmethod
    def parse(cls, text: str) -> Level:
        world, _, stage = text.strip().partition("-")
        return cls(int(world), int(stage))

    @property
    def area(self) -> int:
        """Area index in RAM; x-2 stages of worlds 1/2/4/7 have an intro area first."""
        if self.world in (1, 2, 4, 7) and self.stage >= 2:
            return self.stage + 1
        return self.stage

    @property
    def length(self) -> int:
        """Approximate x position of the level goal (used to report progress)."""
        return LEVEL_LENGTH[(self.world, self.stage)]

    def __str__(self) -> str:
        return f"{self.world}-{self.stage}"


# Goal distances measured by the original 2017 project's emulator script.
LEVEL_LENGTH = {
    (1, 1): 3266, (1, 2): 3266, (1, 3): 2514, (1, 4): 2430,
    (2, 1): 3298, (2, 2): 3266, (2, 3): 3682, (2, 4): 2430,
    (3, 1): 3298, (3, 2): 3442, (3, 3): 2498, (3, 4): 2430,
    (4, 1): 3698, (4, 2): 3266, (4, 3): 2434, (4, 4): 2942,
    (5, 1): 3282, (5, 2): 3298, (5, 3): 2514, (5, 4): 2429,
    (6, 1): 3106, (6, 2): 3554, (6, 3): 2754, (6, 4): 2429,
    (7, 1): 2962, (7, 2): 3266, (7, 3): 3682, (7, 4): 3453,
    (8, 1): 6114, (8, 2): 3554, (8, 3): 3554, (8, 4): 4989,
}  # fmt: skip

ALL_LEVELS = tuple(Level(w, s) for w in range(1, 9) for s in range(1, 5))

# Castles that loop until the right path is taken; excluded from training by default.
MAZE_LEVELS = (Level(4, 4), Level(7, 4), Level(8, 4))

# Held-out levels: never trained on, used to measure generalization. They cover
# overworld (2-1, 5-1, 6-2, 7-1), athletic (3-3), underground (4-2) and castle (6-4).
TEST_LEVELS = tuple(Level.parse(s) for s in ("2-1", "3-3", "4-2", "5-1", "6-2", "6-4", "7-1"))

TRAIN_LEVELS = tuple(lv for lv in ALL_LEVELS if lv not in TEST_LEVELS and lv not in MAZE_LEVELS)

PRESETS = {
    "all": ALL_LEVELS,
    "train": TRAIN_LEVELS,
    "test": TEST_LEVELS,
    "world1": tuple(Level(1, s) for s in range(1, 5)),
    "no-maze": tuple(lv for lv in ALL_LEVELS if lv not in MAZE_LEVELS),
}


def parse_levels(spec: str | Iterable[str | Level]) -> tuple[Level, ...]:
    """Parse ``"train"``, ``"1-1,1-2"`` or an iterable of names/levels."""
    if isinstance(spec, str):
        if spec in PRESETS:
            return PRESETS[spec]
        items: Iterable[str | Level] = [s for s in spec.split(",") if s.strip()]
    else:
        items = spec
    levels = tuple(lv if isinstance(lv, Level) else Level.parse(lv) for lv in items)
    if not levels:
        raise ValueError("empty level list")
    return levels
