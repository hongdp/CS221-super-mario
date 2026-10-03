import numpy as np

from mario_rl import smb
from mario_rl.tiles import EMPTY, ENEMY, MARIO, PLATFORM, SOLID, render_ascii, tile_grid


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
