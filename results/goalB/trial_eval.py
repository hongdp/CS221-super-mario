"""Trial evaluation of a recurrent (RL^2) checkpoint: K attempts per trial, memory kept across attempts, no learning.

usage: python experiments/trial_eval.py CHECKPOINT OUT.json [--levels test] [--trials 10] [--envs 10]
"""

import argparse
import json

import numpy as np
import torch

from mario_rl.env import EnvConfig
from mario_rl.levels import parse_levels
from mario_rl.models import ActorCritic
from mario_rl.recurrent import RecurrentActor, evaluate_trials, load_recurrent
from mario_rl.workers import WorkerPool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--levels", default="test")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--memory", choices=["keep", "reset"], default="keep",
                    help="reset = clear the memory at every attempt (ablation: is the improvement in-context?)")
    a = ap.parse_args()
    model, ckpt = load_recurrent(a.checkpoint)
    model = model.cuda() if torch.cuda.is_available() else model
    env_cfg = EnvConfig(**ckpt["env_config"])
    attempts = env_cfg.trial_episodes
    dummy = ActorCritic(model.obs_type, model.obs_shape, model.n_actions)
    dummy.share_memory()
    pool = WorkerPool(a.envs, env_cfg, dummy, seed=12345)
    try:
        if a.memory == "reset":  # one-attempt "trials", repeated: same levels, memory never carried over
            results = []
            for k in range(attempts):
                eps = evaluate_trials(pool, RecurrentActor(model), parse_levels(a.levels), a.trials, 1)
                for e in eps:
                    e["attempt"] = k
                results += eps
        else:
            results = evaluate_trials(pool, RecurrentActor(model), parse_levels(a.levels), a.trials, attempts)
    finally:
        pool.close()
    report = {"checkpoint": a.checkpoint, "global_step": ckpt["global_step"], "memory": a.memory, "attempts": attempts}
    for k in range(attempts):
        eps = [e for e in results if e["attempt"] == k]
        per = {}
        for e in eps:
            per.setdefault(e["level"], []).append(e)
        report[f"attempt{k + 1}"] = {
            "progress": float(np.mean([np.mean([e["progress"] for e in v]) for v in per.values()])),
            "completion": float(np.mean([np.mean([e["flag_get"] for e in v]) for v in per.values()])),
            "levels": {L: {"progress": float(np.mean([e["progress"] for e in v])), "completion": float(np.mean([e["flag_get"] for e in v])), "n": len(v)} for L, v in sorted(per.items())},
        }
    with open(a.output, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: {m: round(v[m], 3) for m in ("progress", "completion")} for k, v in report.items() if k.startswith("attempt") and isinstance(v, dict)}))
