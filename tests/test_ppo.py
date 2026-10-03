import json

import numpy as np
import pytest
import torch

from mario_rl.cli import main as cli_main
from mario_rl.env import EnvConfig
from mario_rl.models import ActorCritic
from mario_rl.plr import LevelSampler
from mario_rl.ppo import PPOConfig, compute_gae, load_policy, plr_scores, train


def reference_gae(rewards, values, dones, next_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T)
    for t in range(T):
        total, discount = 0.0, 1.0
        for k in range(t, T):
            next_v = next_value if k == T - 1 else values[k + 1]
            delta = rewards[k] + gamma * next_v * (1 - dones[k]) - values[k]
            total += discount * delta
            if dones[k]:
                break
            discount *= gamma * lam
        adv[t] = total
    return adv


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=50).astype(np.float32)
    values = rng.normal(size=50).astype(np.float32)
    dones = rng.random(50) < 0.1
    got = compute_gae(rewards, values, dones, 0.7, 0.99, 0.95)
    np.testing.assert_allclose(
        got, reference_gae(rewards, values, dones, 0.7, 0.99, 0.95), rtol=1e-4, atol=1e-5
    )


def test_plr_scores_split_at_episode_boundaries():
    adv = np.array([1.0, -1.0, 2.0, 2.0, -4.0])
    dones = np.array([False, True, False, False, False])
    level_ids = np.array([3, 3, 5, 5, 5])
    assert list(plr_scores(adv, dones, level_ids)) == [(3, 1.0), (5, pytest.approx(8 / 3))]


def test_level_sampler():
    uniform = LevelSampler(4, "uniform")
    np.testing.assert_allclose(uniform.weights(), 0.25)

    plr = LevelSampler(4, "plr", temperature=0.1, staleness_coef=0.1)
    plr.update(0, 1.0)
    w = plr.weights()
    assert w[0] == 0 and np.isclose(w.sum(), 1)  # unseen levels first
    for level, score in [(1, 5.0), (2, 0.1), (3, 0.2)]:
        plr.update(level, score)
    w = plr.weights()
    assert np.isclose(w.sum(), 1) and w.argmax() == 1 and (w > 0).all()

    restored = LevelSampler(4, "plr")
    restored.load_state_dict(plr.state_dict())
    np.testing.assert_allclose(restored.weights(), w)


@pytest.mark.parametrize(("obs_type", "shape"), [("tiles", (4, 13, 16)), ("pixels", (4, 84, 84))])
def test_actor_critic_shapes_and_logprobs(obs_type, shape):
    model = ActorCritic(obs_type, shape, 7)
    high = 5 if obs_type == "tiles" else 256
    obs = torch.randint(0, high, (3, *shape), dtype=torch.uint8)
    logits, value = model(obs)
    assert logits.shape == (3, 7) and value.shape == (3,)
    action, logprob, _ = model.act(obs, generator=torch.Generator().manual_seed(0))
    lp2, entropy, _ = model.evaluate_actions(obs, action)
    torch.testing.assert_close(logprob, lp2)
    assert (entropy > 0).all()
    greedy, _, _ = model.act(obs, greedy=True)
    torch.testing.assert_close(greedy, logits.argmax(-1))


@pytest.mark.slow
def test_end_to_end_train_eval_play(tmp_path):
    cfg = PPOConfig(
        run_name="tiny",
        out_dir=str(tmp_path),
        total_steps=128,
        num_envs=2,
        num_steps=32,
        num_minibatches=2,
        update_epochs=1,
        eval_interval=128,
        eval_levels="2-1",
        eval_on_train=False,
        eval_episodes=1,
        save_interval=64,
        level_sampler="plr",
        torch_threads=1,
    )
    env_cfg = EnvConfig(levels=("1-1", "1-2"), max_episode_steps=40)
    run_dir = train(cfg, env_cfg)

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("loss/policy" in m for m in metrics)
    assert any("eval_test/progress" in m for m in metrics)
    assert (run_dir / "latest.pt").exists() and (run_dir / "ckpt_64.pt").exists()

    model, ckpt = load_policy(run_dir / "latest.pt")
    assert ckpt["global_step"] == 128 and model.obs_type == "tiles"

    gif = tmp_path / "play.gif"
    cli_main(["play", str(run_dir / "latest.pt"), "--level", "1-1", "--output", str(gif)])
    assert gif.stat().st_size > 0

    report = tmp_path / "eval.json"
    cli_main(["eval", str(run_dir / "latest.pt"), "--levels", "1-1", "--episodes", "1",
              "--num-envs", "1", "--output", str(report)])  # fmt: skip
    assert json.loads(report.read_text())["1-1"]["levels"]["1-1"]["episodes"] == 1

    # resuming keeps the saved configuration and continues the step counter
    cli_main(["train", "--resume", str(run_dir / "latest.pt"), "--total-steps", "192"])
    _, ckpt = load_policy(run_dir / "latest.pt")
    assert ckpt["global_step"] == 192
    assert ckpt["env_config"]["max_episode_steps"] == 40
