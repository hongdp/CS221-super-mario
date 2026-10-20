"""Record one clear of every level: MP4 (full size, real time) + small GIF preview + a manifest of which agent cleared it.

For each level the candidate checkpoints are tried in order, greedy episodes first, then stochastic ones, until one reaches the
flag / axe. Levels no agent clears can fall back to replaying a route found by scripts/route_search.py (labelled as such).

usage: python scripts/record_clears.py OUT_DIR --ckpt A.pt --ckpt B.pt ... [--route experiments/routes/route_4-4.json ...]
       [--levels all] [--greedy-tries 3] [--stochastic-tries 15] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

FPS = 15  # one frame per agent step = 4 NES frames: real time


def save(frames: list[np.ndarray], out: Path, level: str, preview_scale: float, preview_every: int) -> dict:
    import imageio.v3 as iio

    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{level}.mp4"
    iio.imwrite(mp4, np.stack(frames), fps=FPS, codec="libx264", macro_block_size=8)
    w, h = frames[0].shape[1], frames[0].shape[0]
    size = (int(w * preview_scale), int(h * preview_scale))
    small = [Image.fromarray(f).resize(size, Image.NEAREST) for f in frames[::preview_every]]
    small += [small[-1]] * int(FPS / preview_every)  # hold the last frame ~1 s
    pal = [im.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for im in small]
    gif = out / "preview" / f"{level}.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(
        gif,
        save_all=True,
        append_images=pal[1:],
        duration=int(1000 * preview_every / FPS),
        loop=0,
        optimize=True,
    )
    return {
        "mp4": mp4.name,
        "gif": f"preview/{gif.name}",
        "mp4_kb": mp4.stat().st_size // 1024,
        "gif_kb": gif.stat().st_size // 1024,
    }


def play(env, model, greedy: bool, seed: int):
    import torch

    from mario_rl.obs import to_torch

    gen = torch.Generator().manual_seed(seed)
    obs, info = env.reset(seed=seed, options={"level": env.levels[0]})
    frames = [env.render()]
    with torch.inference_mode():
        while True:
            a, _, _ = model.act(to_torch(obs, batch=True), greedy=greedy, generator=gen)
            obs, _, term, trunc, info = env.step(int(a))
            frames.append(env.render())
            if term or trunc:
                return info["flag_get"], frames


def record_level(job) -> dict:
    level, ckpts, routes, greedy_tries, stoch_tries, out, scale, every = job
    from mario_rl.env import EnvConfig, MarioEnv
    from mario_rl.ppo import load_policy

    for ck in ckpts:
        model, c = load_policy(ck)
        cfg = {
            **c["env_config"],
            "levels": (level,),
            "level_weights": None,
            "mid_start_prob": 0.0,
            "mid_start_seed": None,
        }
        cfg = {k: v for k, v in cfg.items() if k in EnvConfig.__dataclass_fields__}
        env = MarioEnv(**cfg)
        try:
            for greedy, n in ((True, greedy_tries), (False, stoch_tries)):
                for seed in range(n):
                    won, frames = play(env, model, greedy, seed)
                    if won:
                        info = save(frames, Path(out), level, scale, every)
                        return {"level": level, "by": "agent", "checkpoint": ck, "greedy": greedy, "seed": seed,
                                "steps": len(frames) - 1, **info}  # fmt: skip
        finally:
            env.close()
    if level in routes:  # fall back to the search route (not the agent)
        route = json.loads(Path(routes[level]).read_text())
        env = MarioEnv(levels=(level,), obs="tiles2", actions="simple_down", noop_max=0, max_episode_steps=10**9,
                       stuck_steps=10**9)  # fmt: skip
        env.reset(seed=0, options={"level": level})
        frames, won = [env.render()], False
        for a in route["actions"]:
            _, _, term, _, info = env.step(a)
            frames.append(env.render())
            if info["flag_get"] or term:
                won = info["flag_get"]
                break
        env.close()
        if won:
            info = save(frames, Path(out), level, scale, every)
            return {
                "level": level,
                "by": "route search",
                "route": routes[level],
                "steps": len(frames) - 1,
                **info,
            }
    return {"level": level, "by": None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--route", action="append", default=[])
    ap.add_argument("--levels", default="all")
    ap.add_argument("--greedy-tries", type=int, default=3)
    ap.add_argument("--stochastic-tries", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preview-scale", type=float, default=0.5)
    ap.add_argument("--preview-every", type=int, default=2)
    a = ap.parse_args()
    from mario_rl.levels import parse_levels

    levels = [str(lv) for lv in parse_levels(a.levels)]
    routes = {json.loads(Path(r).read_text())["level"]: r for r in a.route}
    jobs = [
        (lv, a.ckpt, routes, a.greedy_tries, a.stochastic_tries, a.out, a.preview_scale, a.preview_every)
        for lv in levels
    ]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(record_level, jobs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps(results, indent=1))
    for r in results:
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in ("level", "by", "checkpoint", "greedy", "seed", "steps", "mp4_kb", "gif_kb")
                }
            )
        )
