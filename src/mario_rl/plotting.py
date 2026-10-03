"""Plot training curves and train-vs-held-out evaluation from a run directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_run(run_dir: str | Path, output: str | Path | None = None, smooth: int = 20) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if "train/progress" in r]
    evals = (
        [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
        if (run_dir / "eval.jsonl").exists()
        else []
    )
    evals = list({e["step"]: e for e in evals}.values())  # keep the last record per step

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    steps = np.array([r["step"] for r in train]) / 1e6
    for key, label in [("train/progress", "progress"), ("train/completion", "completion rate")]:
        values = _smooth(np.array([r[key] for r in train]), smooth)
        axes[0].plot(steps[len(steps) - len(values) :], values, label=label)
    axes[0].set(title="Training episodes (all training levels)", xlabel="env steps (M)", ylim=(0, 1))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if evals:
        eval_steps = np.array([e["step"] for e in evals]) / 1e6
        for split, color in [("train", "tab:blue"), ("test", "tab:orange")]:
            if split not in evals[0]:
                continue
            axes[1].plot(eval_steps, [e[split]["progress"] for e in evals], "o-", color=color,
                         label=f"{split} levels: progress")  # fmt: skip
            axes[1].plot(eval_steps, [e[split]["completion"] for e in evals], "s--", color=color,
                         label=f"{split} levels: completion")  # fmt: skip
        axes[1].set(title="Evaluation: training vs held-out levels", xlabel="env steps (M)", ylim=(0, 1))
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        last = evals[-1]
        names, values, colors = [], [], []
        for split, color in [("train", "tab:blue"), ("test", "tab:orange")]:
            for name, stats in last.get(split, {}).get("levels", {}).items():
                names.append(name)
                values.append(stats["progress"])
                colors.append(color)
        axes[2].bar(range(len(names)), values, color=colors)
        axes[2].set_xticks(range(len(names)), names, rotation=90, fontsize=8)
        axes[2].set(title=f"Per-level progress @ {last['step'] / 1e6:.1f}M (orange = held-out)", ylim=(0, 1))
        axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    output = Path(output) if output else run_dir / "curves.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
