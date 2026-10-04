import numpy as np

from mario_rl import smb
from mario_rl.tiles import (
    EMPTY,
    ENEMY,
    HAZARD2,
    MARIO,
    MARIO2,
    PLATFORM,
    PLATFORM2,
    SOLID,
    SOLID2,
    STOMPABLE2,
    mario_features,
    render_ascii,
    render_ascii_v2,
    tile_grid,
    tile_grid_v2,
)


def blank_ram() -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[smb.PLAYER_Y_VIEWPORT] = 1
    ram[smb.PLAYER_SIZE] = 1  # small
    return ram


def put_metatile(ram, page, row, col, value):
    ram[smb.BLOCK_BUFFER + page * 13 * 16 + row * 16 + col] = value


def put_enemy(ram, slot, kind, x, y):
    ram[smb.ENEMY_ACTIVE + slot] = 1
    ram[smb.ENEMY_TYPE + slot] = kind
    ram[smb.ENEMY_PAGE + slot] = x // 256
    ram[smb.ENEMY_X + slot] = x % 256
    ram[smb.ENEMY_Y + slot] = y
    ram[smb.ENEMY_Y_VIEWPORT + slot] = 1


def test_static_scene():
    ram = blank_ram()
    for col in range(16):
        put_metatile(ram, 0, 11, col, 0x54)  # ground
        put_metatile(ram, 0, 12, col, 0x54)
    put_metatile(ram, 0, 7, 6, 0xC0)  # question block
    put_metatile(ram, 0, 5, 3, 0xC2)  # coin -> not solid
    ram[smb.PLAYER_X] = 80
    ram[smb.PLAYER_Y] = 176  # standing on the ground
    put_enemy(ram, 0, 0x06, x=160, y=184)  # goomba on the ground
    put_enemy(ram, 1, 0x30, x=200, y=100)  # flagpole flag: ignored

    grid = tile_grid(ram)
    assert grid.shape == (13, 16)
    assert (grid[11:] == SOLID).all()
    assert grid[7, 6] == SOLID
    assert grid[5, 3] == EMPTY
    assert grid[10, 5] == MARIO
    assert grid[10, 10] == ENEMY
    assert (grid == ENEMY).sum() == 1


def test_big_mario_is_two_tiles_tall():
    ram = blank_ram()
    ram[smb.PLAYER_SIZE] = 0
    ram[smb.PLAYER_X] = 80
    ram[smb.PLAYER_Y] = 176
    grid = tile_grid(ram)
    assert grid[9, 5] == MARIO and grid[10, 5] == MARIO


def test_scrolling_uses_both_buffer_pages():
    ram = blank_ram()
    ram[smb.PLAYER_Y_VIEWPORT] = 0  # hide Mario
    ram[smb.SCREEN_LEFT_PAGE] = 1
    ram[smb.SCREEN_LEFT_X] = 128  # screen spans world x 384..639
    put_metatile(ram, 1, 3, 8, 0x51)  # world x 384 -> first screen column
    put_metatile(ram, 0, 3, 0, 0x51)  # world x 512 -> wraps to page 0, screen column 8
    grid = tile_grid(ram)
    assert grid[3, 0] == SOLID
    assert grid[3, 8] == SOLID
    assert (grid == SOLID).sum() == 2


def test_platforms_are_marked_wide():
    ram = blank_ram()
    ram[smb.PLAYER_Y_VIEWPORT] = 0
    put_enemy(ram, 2, 0x25, x=64, y=120)
    grid = tile_grid(ram)
    assert (grid == PLATFORM).sum() == 3


def test_render_ascii():
    text = render_ascii(tile_grid(blank_ram()))
    assert len(text.splitlines()) == 13


def test_flag_and_death_detection():
    ram = blank_ram()
    ram[smb.PLAYER_STATE] = 0x08
    assert not smb.flag_get(ram) and not smb.is_dead(ram)
    ram[smb.OPER_MODE] = 2  # castle axe
    assert smb.flag_get(ram)
    ram[smb.OPER_MODE] = 1
    ram[smb.PLAYER_FLOAT_STATE] = 3
    assert not smb.flag_get(ram)  # climbing a vine
    ram[smb.ENEMY_TYPE + 2] = smb.ENEMY_FLAGPOLE_FLAG
    assert smb.flag_get(ram)
    ram[smb.PLAYER_STATE] = smb.STATE_DYING
    assert smb.is_dead(ram)
    ram[smb.PLAYER_STATE] = 0x08
    ram[smb.PLAYER_Y_VIEWPORT] = 2  # fell into a pit
    assert smb.is_dead(ram)


def test_bcd_and_positions():
    ram = blank_ram()
    ram[smb.TIME : smb.TIME + 3] = [3, 9, 8]
    assert smb.game_time(ram) == 398
    ram[smb.PLAYER_PAGE] = 3
    ram[smb.PLAYER_X] = 17
    assert smb.x_pos(ram) == 3 * 256 + 17


# --- v2 grid -------------------------------------------------------------------------


def put_box(ram, address, box):
    ram[address : address + 4] = box


def test_v2_uses_collision_boxes_and_classes():
    ram = blank_ram()
    for col in range(16):
        put_metatile(ram, 0, 11, col, 0x54)  # ground at screen y 208..224 -> rows 22-23
    put_box(ram, smb.PLAYER_BOX, (115, 196, 125, 208))  # small Mario standing on the ground
    put_enemy(ram, 0, 0x06, x=189, y=184)  # goomba
    put_box(ram, smb.ENEMY_BOX, (192, 196, 202, 204))
    put_enemy(ram, 1, 0x0D, x=40, y=150)  # piranha plant
    put_box(ram, smb.ENEMY_BOX + 4, (42, 150, 54, 166))
    put_enemy(ram, 2, 0x27, x=168, y=157)  # 48px lift
    put_box(ram, smb.ENEMY_BOX + 8, (168, 157, 216, 170))
    grid = tile_grid_v2(ram)
    assert grid.shape == (26, 32)
    assert (grid[22:24] == SOLID2).all()
    assert (grid[20:22, 14:16] == MARIO2).all() and (grid == MARIO2).sum() == 4
    assert (grid[20:22, 24:26] == STOMPABLE2).all()
    assert (grid == HAZARD2).sum() > 0 and grid[15, 5] == HAZARD2
    assert (grid[15, 21:27] == PLATFORM2).all()  # six 8px cells wide
    assert len(render_ascii_v2(grid).splitlines()) == 26


def test_v2_firebar_segments_and_hammers():
    ram = blank_ram()
    ram[smb.PLAYER_Y_VIEWPORT] = 0
    ram[smb.OAM : smb.OAM + 256] = 0xF8  # hide all sprites
    put_enemy(ram, 0, 0x1D, x=194, y=164)  # firebar pivot: drawn from OAM segments, not a box
    for k, (x, y) in enumerate([(194, 164), (183, 153), (172, 142)]):
        ram[smb.OAM + 4 * k : smb.OAM + 4 * k + 4] = (y - 1, 0x64, 0xC2, x)
    grid = tile_grid_v2(ram)
    assert (grid == HAZARD2).sum() >= 3
    assert grid[(164 - 32) // 8, 194 // 8] == HAZARD2

    ram = blank_ram()
    ram[smb.PLAYER_Y_VIEWPORT] = 0
    ram[smb.MISC_STATE + 3] = 0x81
    put_box(ram, smb.MISC_BOX + 12, (56, 148, 64, 156))
    assert (tile_grid_v2(ram) == HAZARD2).sum() == 0  # no hammer bro: misc objects are coins
    put_enemy(ram, 0, 0x05, x=100, y=120)
    put_box(ram, smb.ENEMY_BOX, (102, 124, 114, 144))
    grid = tile_grid_v2(ram)
    assert grid[(148 - 32) // 8, 56 // 8] == HAZARD2


def test_mario_features():
    ram = blank_ram()
    ram[smb.PLAYER_X_SPEED] = 40
    ram[smb.PLAYER_Y_SPEED] = 256 - 5  # rising
    ram[smb.PLAYER_FLOAT_STATE] = 1
    ram[smb.PLAYER_STATUS] = 2
    ram[smb.SWIMMING] = 1
    f = mario_features(ram)
    np.testing.assert_allclose(f, [1.0, -1.0, 0, 1, 0, 0, 1, 1, 1, 0])
