"""Tests that drive the real emulator. stable-retro allows one emulator per
process, so every test closes its environment before the next one starts."""

from contextlib import closing

import numpy as np
from gymnasium.utils.env_checker import check_env

from mario_rl import smb
from mario_rl.env import ACTION_SETS, MarioEnv
from mario_rl.levels import ALL_LEVELS, Level
from mario_rl.tiles import MARIO, SOLID

RIGHT = 1  # index of "right" in the simple action set


def test_gymnasium_api_compliance():
    with closing(MarioEnv(levels=("1-1", "1-2"))) as env:
        check_env(env, skip_render_check=True)


def test_every_level_can_be_selected():
    with closing(MarioEnv(levels=("1-1",), noop_max=0)) as env:
        for level in ALL_LEVELS:
            _, info = env.reset(options={"level": level})
            ram = env._game.ram()
            assert info["level"] == str(level)
            assert (int(ram[smb.WORLD]) + 1, int(ram[smb.STAGE]) + 1) == (level.world, level.stage)
            assert info["time"] > 0


def test_initial_tile_observation_shows_mario_on_the_ground():
    with closing(MarioEnv(levels=("1-1",), noop_max=0)) as env:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 13, 16)
        frame = obs[-1]
        rows, cols = np.nonzero(frame == MARIO)
        assert len(rows) == 1
        assert frame[rows[0] + 1, cols[0]] == SOLID


def test_same_seed_same_trajectory():
    actions = np.random.default_rng(0).integers(0, len(ACTION_SETS["simple"]), size=60)

    def rollout():
        with closing(MarioEnv(levels=("1-1", "4-1"))) as env:
            obs, _ = env.reset(seed=42)
            out = [obs]
            for a in actions:
                obs, *_ = env.step(int(a))
                out.append(obs)
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
