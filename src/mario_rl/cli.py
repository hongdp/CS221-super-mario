"""Command line interface: ``mario-rl train | eval | play``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import types
import typing
from pathlib import Path

import numpy as np

from .env import EnvConfig, MarioEnv
from .evaluate import format_table, summarize
from .levels import Level, parse_levels
from .ppo import PPOConfig, load_policy, train


def _add_dataclass_args(parser: argparse.ArgumentParser, cls, skip=()) -> None:
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        hint = hints[f.name]
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        origin = typing.get_origin(hint)
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if hint is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif origin is tuple:
            parser.add_argument(flag, default=",".join(map(str, default)), help="comma separated or preset")
        elif origin is typing.Literal:
            parser.add_argument(flag, choices=typing.get_args(hint), default=default)
        elif origin in (typing.Union, types.UnionType):
            parser.add_argument(flag, type=args[0], default=default)
        else:
            parser.add_argument(flag, type=hint, default=default)


def _build(cls, ns: argparse.Namespace, **overrides):
    values = {}
    for f in dataclasses.fields(cls):
        if hasattr(ns, f.name):
            values[f.name] = getattr(ns, f.name)
    values.update(overrides)
    return cls(**values)


def _levels_arg(spec: str) -> tuple[str, ...]:
    return tuple(str(lv) for lv in parse_levels(spec))


def cmd_train(ns: argparse.Namespace) -> None:
    if ns.resume:
        # Continue with the saved configuration; only --total-steps may be raised.
        import torch

        ckpt = torch.load(ns.resume, map_location="cpu", weights_only=False)
        env_cfg = EnvConfig(**ckpt["env_config"])
        saved = dict(ckpt["ppo_config"])
        saved["total_steps"] = max(saved["total_steps"], ns.total_steps)
        saved["resume"] = ns.resume
        ppo_cfg = PPOConfig(**saved)
    else:
        env_cfg = _build(EnvConfig, ns, levels=_levels_arg(ns.levels))
        ppo_cfg = _build(PPOConfig, ns)
    train(ppo_cfg, env_cfg)


def cmd_eval(ns: argparse.Namespace) -> None:
    from .workers import WorkerPool

    model, ckpt = load_policy(ns.checkpoint)
    env_cfg = EnvConfig(**ckpt["env_config"])
    model.share_memory()
    pool = WorkerPool(ns.num_envs, env_cfg, model, ns.seed)
    try:
        report = {"checkpoint": str(ns.checkpoint), "global_step": ckpt["global_step"]}
        for spec in ns.levels:
            levels = parse_levels(spec)
            summary = summarize(pool.evaluate(levels, ns.episodes, ns.greedy))
            report[spec] = summary
            print(format_table(summary, f"== {spec} ({ckpt['global_step']:,d} steps) =="))
    finally:
        pool.close()
    if ns.output:
        Path(ns.output).write_text(json.dumps(report, indent=2))


def cmd_play(ns: argparse.Namespace) -> None:
    import torch

    model, ckpt = load_policy(ns.checkpoint)
    env = MarioEnv(EnvConfig(**ckpt["env_config"]))
    frames = []
    gen = torch.Generator().manual_seed(ns.seed)
    obs, info = env.reset(seed=ns.seed, options={"level": Level.parse(ns.level)})
    total = 0.0
    with torch.inference_mode():
        while True:
            action, _, _ = model.act(torch.from_numpy(obs)[None], greedy=ns.greedy, generator=gen)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            frames.append(env.render())
            if terminated or truncated:
                break
    env.close()
    ep = info["episode"]
    print(
        f"level {info['level']}: progress {ep['progress']:.1%}, flag={ep['flag_get']}, "
        f"return {total:.1f}, steps {ep['l']}"
    )
    if ns.output:
        _save_video(frames, Path(ns.output), fps=ns.fps)
        print(f"saved {ns.output}")


def _save_video(frames, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    else:
        import imageio.v3 as iio  # optional dependency: pip install "mario-rl[video]"

        iio.imwrite(path, np.stack(frames), fps=fps)


def cmd_plot(ns: argparse.Namespace) -> None:
    from .plotting import plot_run  # needs matplotlib: pip install "mario-rl[plot]"

    print(f"saved {plot_run(ns.run_dir, ns.output, ns.smooth)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mario-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a PPO agent on many levels")
    _add_dataclass_args(p_train, EnvConfig, skip=("level_weights",))
    _add_dataclass_args(p_train, PPOConfig)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on level sets")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("--levels", nargs="+", default=["test", "train"])
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--num-envs", type=int, default=4)
    p_eval.add_argument("--greedy", action="store_true")
    p_eval.add_argument("--seed", type=int, default=123)
    p_eval.add_argument("--output", help="write the report as JSON")
    p_eval.set_defaults(func=cmd_eval)

    p_play = sub.add_parser("play", help="run one episode and optionally save a video")
    p_play.add_argument("checkpoint")
    p_play.add_argument("--level", default="1-1")
    p_play.add_argument("--greedy", action="store_true")
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--fps", type=int, default=15)
    p_play.add_argument("--output", help=".gif (Pillow) or .mp4 (needs imageio[ffmpeg])")
    p_play.set_defaults(func=cmd_play)

    p_plot = sub.add_parser("plot", help="plot training / evaluation curves of a run")
    p_plot.add_argument("run_dir")
    p_plot.add_argument("--output")
    p_plot.add_argument("--smooth", type=int, default=20)
    p_plot.set_defaults(func=cmd_plot)

    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
