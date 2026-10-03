import pytest

from mario_rl.levels import (
    ALL_LEVELS,
    LEVEL_LENGTH,
    MAZE_LEVELS,
    TEST_LEVELS,
    TRAIN_LEVELS,
    Level,
    parse_levels,
)


def test_level_parse_and_str_roundtrip():
    lv = Level.parse("4-2")
    assert (lv.world, lv.stage) == (4, 2)
    assert str(lv) == "4-2"


@pytest.mark.parametrize("bad", ["0-1", "9-1", "1-5", "1-0"])
def test_invalid_levels_rejected(bad):
    with pytest.raises(ValueError):
        Level.parse(bad)


def test_area_skips_intro_cutscene_areas():
    assert Level(1, 1).area == 1
    assert Level(1, 2).area == 3  # 1-2 starts after the "walk into the pipe" intro area
    assert Level(4, 4).area == 5
    assert Level(3, 2).area == 2  # world 3 has no intro area


def test_splits_are_disjoint_and_cover_everything():
    assert len(ALL_LEVELS) == 32
    assert set(LEVEL_LENGTH) == {(lv.world, lv.stage) for lv in ALL_LEVELS}
    train, test, maze = set(TRAIN_LEVELS), set(TEST_LEVELS), set(MAZE_LEVELS)
    assert not train & test
    assert not train & maze
    assert not test & maze
    assert train | test | maze == set(ALL_LEVELS)


def test_parse_levels_presets_and_lists():
    assert parse_levels("train") == TRAIN_LEVELS
    assert parse_levels("1-1, 2-3") == (Level(1, 1), Level(2, 3))
    assert parse_levels(["5-1", Level(6, 2)]) == (Level(5, 1), Level(6, 2))
    with pytest.raises(ValueError):
        parse_levels("")
