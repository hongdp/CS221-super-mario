"""Find a route through a level by Go-Explore-style search (Ecoffet et al.) from saved emulator states.

An archive maps cells (area, x // 32, y // 32, maze counters, Mario's size) to the emulator state and action sequence that
reached them. Each iteration restores a rarely chosen cell and explores with random held actions; new cells are added,
deaths discarded. Independent searches run in parallel processes; the first one to reach the flag / axe wins.

usage: python scripts/route_search.py LEVEL OUT_PREFIX [--workers 20] [--minutes 30] [--actions simple_down]
writes OUT_PREFIX.json (actions from the level start, per-step x/y) and OUT_PREFIX.states.pkl (states along the route)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time

import numpy as np

from mario_rl import smb
from mario_rl.env import MAZE_CORRECT_COUNTER, MAZE_PASS_COUNTER, MarioEnv


def cell_of(ram) -> tuple:
    return (
        int(ram[smb.AREA]),
        smb.x_pos(ram) // 32,
        int(ram[smb.PLAYER_Y]) // 32 if ram[smb.PLAYER_Y_VIEWPORT] == 1 else -1,
        int(ram[MAZE_CORRECT_COUNTER]),
        int(ram[MAZE_PASS_COUNTER]),
        int(ram[smb.PLAYER_STATUS]),
    )


def search(args) -> dict:
    level, seed, minutes, actions = args
    rng = np.random.default_rng(seed)
    env = MarioEnv(
        levels=(level,), obs="tiles2", actions=actions, noop_max=0, max_episode_steps=10**9, stuck_steps=10**9
    )
    env.reset(seed=seed, options={"level": level})
    emu = env._game.emu
    n_actions = env.action_space.n
    start = emu.get_state()
    ram = env._game.ram()
    archive = {cell_of(ram): {"state": start, "actions": [], "chosen": 0, "x": smb.x_pos(ram)}}
    best_x, steps, t_end = smb.x_pos(ram), 0, time.time() + 60 * minutes
    while time.time() < t_end:
        cells = list(archive)
        # Go-Explore selection: rarely chosen cells first, with a mild preference for progress
        w = np.array(
            [(1.0 / np.sqrt(1 + archive[c]["chosen"])) * (1.0 + archive[c]["x"] / 2000.0) for c in cells]
        )
        c = cells[rng.choice(len(cells), p=w / w.sum())]
        entry = archive[c]
        entry["chosen"] += 1
        emu.set_state(entry["state"])
        env._reset_episode_state(env._game.ram())
        acts = list(entry["actions"])
        for _ in range(int(rng.integers(1, 12))):  # a few random "held" actions
            a, hold = int(rng.integers(n_actions)), int(rng.integers(1, 10))
            dead = False
            for _ in range(hold):
                _, _, terminated, _, info = env.step(a)
                acts.append(a)
                steps += 1
                ram = env._game.ram()
                if info["flag_get"]:
                    env.close()
                    return {
                        "level": level,
                        "seed": seed,
                        "found": True,
                        "actions": acts,
                        "steps": steps,
                        "cells": len(archive),
                    }
                if terminated:
                    dead = True
                    break
                cell = cell_of(ram)
                x = smb.x_pos(ram)
                best_x = max(best_x, x)
                old = archive.get(cell)
                if old is None or len(acts) < len(old["actions"]):
                    archive[cell] = {
                        "state": emu.get_state(),
                        "actions": list(acts),
                        "chosen": 0 if old is None else old["chosen"],
                        "x": x,
                    }
            if dead:
                break
    env.close()
    return {
        "level": level,
        "seed": seed,
        "found": False,
        "best_x": best_x,
        "steps": steps,
        "cells": len(archive),
    }


def replay(level: str, actions: list[int], action_set: str) -> tuple[bool, list, list]:
    """Replay from the level start (no idle frames); returns (reached the goal, per-step (x, y, area), states)."""
    env = MarioEnv(
        levels=(level,),
        obs="tiles2",
        actions=action_set,
        noop_max=0,
        max_episode_steps=10**9,
        stuck_steps=10**9,
    )
    env.reset(seed=0, options={"level": level})
    trace, states, ok = [], [], False
    for a in actions:
        _, _, terminated, _, info = env.step(a)
        ram = env._game.ram()
        trace.append((smb.x_pos(ram), int(ram[smb.PLAYER_Y]), int(ram[smb.AREA])))
        states.append(env._game.emu.get_state())
        if info["flag_get"]:
            ok = True
            break
        if terminated:
            break
    env.close()
    return ok, trace, states


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("level")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--actions", default="simple_down")
    a = ap.parse_args()
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(search, [(a.level, s, a.minutes, a.actions) for s in range(a.workers)])
    found = [r for r in results if r["found"]]
    summary = {
        "level": a.level,
        "workers": a.workers,
        "minutes": a.minutes,
        "found": len(found),
        "best_x": max([r.get("best_x", 0) for r in results] + [0]),
        "steps_total": sum(r["steps"] for r in results),
        "wall_s": round(time.time() - t0),
    }
    if found:
        best = min(found, key=lambda r: len(r["actions"]))
        ok, trace, states = replay(a.level, best["actions"], a.actions)
        summary.update(
            {
                "replay_reaches_goal": ok,
                "route_steps": len(best["actions"]),
                "actions": best["actions"],
                "trace": trace,
            }
        )
        with open(a.out + ".states.pkl", "wb") as f:
            pickle.dump({"level": a.level, "states": states, "trace": trace}, f)
    with open(a.out + ".json", "w") as f:
        json.dump(summary, f)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("actions", "trace")}))
