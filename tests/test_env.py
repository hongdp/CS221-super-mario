"""Tests that drive the real emulator. stable-retro allows one emulator per
process, so every test closes its environment before the next one starts."""

from contextlib import closing

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mario_rl import smb
from mario_rl.env import ACTION_SETS, MarioEnv
from mario_rl.levels import ALL_LEVELS, Level
from mario_rl.tiles import MARIO, SOLID

RIGHT = 1  # index of "right" in the simple action set


def test_gymnasium_api_compliance():
    with closing(MarioEnv(levels=("1-1", "1-2"), obs="tiles")) as env:
        check_env(env, skip_render_check=True)


def test_every_level_can_be_selected():
    with closing(MarioEnv(levels=("1-1",), noop_max=0)) as env:
        for level in ALL_LEVELS:
            _, info = env.reset(options={"level": level})
            ram = env._game.ram()
            assert info["level"] == str(level)
            assert (int(ram[smb.WORLD]) + 1, int(ram[smb.STAGE]) + 1) == (level.world, level.stage)
            assert info["time"] > 0


def test_default_observation_is_tiles2():
    with closing(MarioEnv(levels=("1-1",))) as env:
        assert set(env.observation_space.spaces) == {"grid", "vec"}


def test_initial_tile_observation_shows_mario_on_the_ground():
    with closing(MarioEnv(levels=("1-1",), noop_max=0, obs="tiles")) as env:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 13, 16)
        frame = obs[-1]
        rows, cols = np.nonzero(frame == MARIO)
        assert len(rows) == 1
        assert frame[rows[0] + 1, cols[0]] == SOLID


@pytest.mark.parametrize("obs_type", ["tiles", "tiles2"])
def test_same_seed_same_trajectory(obs_type):
    actions = np.random.default_rng(0).integers(0, len(ACTION_SETS["simple"]), size=60)

    def flat(obs):
        return np.concatenate([np.ravel(v) for v in obs.values()]) if isinstance(obs, dict) else np.ravel(obs)

    def rollout():
        with closing(MarioEnv(levels=("1-1", "4-1"), obs=obs_type)) as env:
            obs, _ = env.reset(seed=42)
            out = [flat(obs)]
            for a in actions:
                obs, *_ = env.step(int(a))
                out.append(flat(obs))
            return np.stack(out)

    np.testing.assert_array_equal(rollout(), rollout())


def test_running_into_the_first_goomba_terminates_with_penalty():
    with closing(MarioEnv(levels=("1-1",), noop_max=0)) as env:
        env.reset(seed=0)
        for _ in range(500):
            _, reward, terminated, truncated, info = env.step(RIGHT)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert reward < 0
        assert not info["flag_get"]
        ep = info["episode"]
        assert 0 < ep["progress"] < 0.2
        assert ep["l"] == env._steps


def test_no_progress_truncates():
    with closing(MarioEnv(levels=("1-1",), noop_max=0, stuck_steps=20)) as env:
        env.reset(seed=0)
        for _ in range(25):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated and not terminated


def test_level_weights_control_sampling():
    with closing(MarioEnv(levels=("1-1", "2-1", "3-1"), noop_max=0)) as env:
        env.set_level_weights([0.0, 0.0, 1.0])
        for seed in range(5):
            _, info = env.reset(seed=seed)
            assert info["level"] == "3-1"


def test_pixel_observations():
    with closing(MarioEnv(levels=("1-1",), obs="pixels", noop_max=0)) as env:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
        obs, *_ = env.step(RIGHT)
        assert obs.std() > 0
        assert env.render().shape == (224, 240, 3)


def test_forced_level_outside_training_set():
    with closing(MarioEnv(levels=("1-1",), noop_max=0)) as env:
        _, info = env.reset(options={"level": "8-3"})
        assert info["level"] == "8-3"
        for _ in range(500):
            *_, terminated, truncated, info = env.step(RIGHT)
            if terminated or truncated:
                break
        assert info["episode"]["level_id"] == -1
        assert env.level == Level(8, 3)


def test_sticky_actions():
    with closing(MarioEnv(levels=("1-1",), noop_max=0, sticky_prob=1.0)) as env:
        _, info = env.reset(seed=0)
        start = info["x_pos"]
        for _ in range(10):
            _, _, _, _, info = env.step(RIGHT)  # every frame repeats the initial NOOP
        assert info["x_pos"] == start
    with closing(MarioEnv(levels=("1-1",), noop_max=0, sticky_prob=0.25)) as env:
        _, info = env.reset(seed=0)
        start = info["x_pos"]
        for _ in range(10):
            _, _, _, _, info = env.step(RIGHT)
        assert info["x_pos"] > start


def test_tiles2_dict_observation():
    with closing(MarioEnv(levels=("1-1", "2-2"), obs="tiles2", noop_max=0)) as env:
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=0, options={"level": "1-1"})
        grid, vec = obs["grid"][-1], obs["vec"]
        rows, cols = np.nonzero(grid == 5)  # MARIO2
        assert len(rows) == 4  # small Mario: 2x2 cells of 8px
        assert (grid[rows.max() + 1, cols] == 1).all()  # standing on solid ground
        assert vec[2] == 1.0  # on the ground
        obs, _ = env.reset(options={"level": "2-2"})
        assert obs["vec"][8] == 1.0  # swimming level


def test_ego_window_follows_mario():
    with closing(MarioEnv(levels=("1-1",), obs="tiles2", noop_max=0, ego_back=4, ego_ahead=12)) as env:
        check_env(env, skip_render_check=True)
        assert env.observation_space["grid"].shape == (4, 26, 16)
        obs, _ = env.reset(seed=0, options={"level": "1-1"})
        for _ in range(60):  # run right until the screen scrolls
            obs, *_ = env.step(1)
        full = env._frames[-1]
        grid = obs["grid"][-1]
        _, cols = np.nonzero(grid == 5)  # MARIO2
        assert cols.min() == 4  # Mario's left cell sits ego_back cells into the window
        _, full_cols = np.nonzero(full == 5)
        left = full_cols.min() - 4
        assert np.array_equal(grid, full[:, left : left + 16])
        # Mario at the left screen edge: the cells behind him are padded as solid wall,
        # at the right edge the cells ahead are padded as empty
        ram = env._game.ram()
        frames = env._frames.copy()
        env._frames[:] = 3  # HAZARD2 everywhere, so padding is easy to spot
        left = smb.screen_left(ram)
        try:
            for x, (pad_slice, value) in ((left, (slice(0, 4), 1)), (left + 255, (slice(4 + 1, 16), 0))):
                ram_x = ram.copy()
                ram_x[smb.PLAYER_PAGE], ram_x[smb.PLAYER_X] = divmod(x, 256)
                window = env._ego_window(ram_x)
                assert window.shape == (4, 26, 16)
                assert (window[..., pad_slice] == value).all()
        finally:
            env._frames[:] = frames


def test_mid_level_starts_only_for_training_resets():
    with closing(MarioEnv(levels=("1-1",), obs="tiles2", noop_max=0, mid_start_prob=1.0)) as env:
        env.reset(seed=0, options={"level": "1-1"})
        start_x = None
        for t in range(400):  # hold right + B and hop over the first goombas
            _, _, term, trunc, info = env.step(4 if t % 12 < 6 else 3)
            if term or trunc:
                break
        assert info["x_pos"] > 512
        archive = env._archive[env.level]
        assert (0, 1) in archive  # the 256-511 px bucket was confirmed
        assert not info["episode"]["mid_start"]
        _, info = env.reset()  # training reset: starts from an archived state
        assert env._mid_start and info["x_pos"] >= 256
        start_x = info["x_pos"]
        _, info = env.reset(options={"level": "1-1"})  # evaluation reset: always from the start
        assert not env._mid_start and info["x_pos"] < start_x


def test_mid_level_starts_disabled_by_default():
    with closing(MarioEnv(levels=("1-1",), obs="tiles2", noop_max=0)) as env:
        env.reset(seed=0)
        for _ in range(60):
            env.step(1)
        assert env._archive == {}


def test_mid_level_start_skips_idle_frames():
    """Restarting from an archived state is deterministic: no random idle frames after the load."""
    with closing(MarioEnv(levels=("1-1",), obs="tiles2", noop_max=30, mid_start_prob=1.0)) as env:
        env.reset(seed=0, options={"level": "1-1"})
        for t in range(400):
            _, _, term, trunc, _ = env.step(4 if t % 12 < 6 else 3)
            if term or trunc:
                break
        key, state = next(iter(env._archive[env.level].items()))
        env._archive = {env.level: {key: state}}
        rams = []
        for seed in (1, 2, 3):
            env.reset(seed=seed)
            assert env._mid_start
            rams.append(env._game.ram().copy())
        assert all(np.array_equal(rams[0], r) for r in rams[1:])
