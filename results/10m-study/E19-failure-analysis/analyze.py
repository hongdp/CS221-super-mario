"""Run a policy on held-out levels, classify failures, and save screen|grid snapshots before each failure."""

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from mario_rl import smb
from mario_rl.env import EnvConfig, MarioEnv
from mario_rl.levels import Level
from mario_rl.obs import to_torch
from mario_rl.ppo import load_policy

COLORS = {
    0: (0, 0, 0),
    1: (150, 90, 40),
    2: (60, 200, 60),
    3: (230, 30, 30),
    4: (60, 120, 255),
    5: (255, 220, 0),
}


def grid_img(grid):  # (26,32) -> 256x208 RGB, placed at y=32 like the screen
    rgb = np.zeros((240, 256, 3), np.uint8)
    rgb[:32] = 40
    for c, col in COLORS.items():
        m = np.kron(grid == c, np.ones((8, 8), bool))
        rgb[32:][m] = col
    return Image.fromarray(rgb)


def snapshot(frame, grid, path, title):
    full = np.full((240, 256, 3), 90, np.uint8)  # retro crops 8px overscan on every side
    full[8:232, 8:248] = frame
    scr = Image.fromarray(full)
    g = grid_img(grid)
    over = Image.blend(scr, g, 0.45)
    canvas = Image.new("RGB", (256 * 3 + 20, 260), (255, 255, 255))
    for i, im in enumerate([scr, g, over]):
        canvas.paste(im, (i * 266, 20))
    ImageDraw.Draw(canvas).text((4, 4), title, fill=(0, 0, 0))
    canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST).save(path)


def main(ckpt, out, episodes=20, levels=("2-1", "3-3", "4-2", "5-1", "6-2", "6-4", "7-1")):
    model, c = load_policy(ckpt)
    cfg = EnvConfig(**{**c["env_config"], "levels": tuple(levels)})
    env = MarioEnv(cfg)
    gen = torch.Generator().manual_seed(0)
    records = []
    for lv in levels:
        saved = 0
        for ep in range(episodes):
            obs, info = env.reset(seed=1000 + ep, options={"level": Level.parse(lv)})
            hist = deque(maxlen=3)
            with torch.inference_mode():
                while True:
                    hist.append((env.render(), obs["grid"][-1].copy(), info.get("x_pos")))
                    a, _, _ = model.act(to_torch(obs, batch=True), generator=gen)
                    obs, _r, term, trunc, info = env.step(int(a))
                    if term or trunc:
                        break
            ram = env._game.ram()
            e = info["episode"]
            if e["flag_get"]:
                cause = "flag"
            elif term:
                cause = "pit" if ram[smb.PLAYER_Y_VIEWPORT] > 1 else "enemy/hazard"
            else:
                cause = "timeout" if e["l"] >= cfg.max_episode_steps else "stuck"
            rec = {
                "level": lv,
                "ep": ep,
                "cause": cause,
                "progress": round(e["progress"], 3),
                "x": int(info["x_pos"]),
                "steps": e["l"],
            }
            records.append(rec)
            if cause != "flag" and saved < 3:
                frame, grid, _x = hist[0]
                name = f"{lv}_ep{ep:02d}_{cause.replace('/', '-')}_x{rec['x']}.png"
                snapshot(
                    frame,
                    grid,
                    out / name,
                    f"{lv} ep{ep} {cause} progress {rec['progress']:.0%} x={rec['x']} (2 steps before end) | screen | grid | overlay",
                )
                saved += 1
            print(json.dumps(rec), flush=True)
    env.close()
    (out / "records.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    d = Path(sys.argv[2])
    main(sys.argv[1], d / "snapshots")
