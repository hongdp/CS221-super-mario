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


def plot_checkpoint_evals(eval_dir: str | Path, output: str | Path | None = None) -> Path:
    """Train vs held-out curves from ``mario-rl eval --output`` reports of several checkpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = Path(eval_dir)
    reports = sorted(
        (json.loads(p.read_text()) for p in eval_dir.glob("*.json")), key=lambda r: r["global_step"]
    )
    trained = [r for r in reports if r["global_step"] > 0]
    baseline = next((r for r in reports if r["global_step"] == 0), None)
    steps = np.array([r["global_step"] for r in trained]) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "progress", "Mean level progress"),
        (axes[1], "completion", "Level completion rate"),
    ]:
        for split, color, label in [("train", "tab:blue", "training levels (22)"),
                                    ("test", "tab:orange", "held-out levels (7)")]:  # fmt: skip
            if split not in trained[0]:
                continue
            ax.plot(steps, [r[split][metric] for r in trained], "o-", color=color, label=label)
            if baseline is not None and split in baseline:
                ax.axhline(baseline[split][metric], color=color, ls=":", lw=1)
        ax.set(title=title, xlabel="env steps (M)", ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].legend(title="dotted: untrained policy", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    output = Path(output) if output else eval_dir / "generalization.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def plot_eval_comparison(
    runs: dict[str, str | Path], output: str | Path, max_step: float | None = None
) -> Path:
    """Overlay train / held-out progress of several runs (dirs of `mario-rl eval --output` reports)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, eval_dir), color in zip(runs.items(), colors, strict=False):
        reports = sorted(
            (json.loads(p.read_text()) for p in Path(eval_dir).glob("*.json")), key=lambda r: r["global_step"]
        )
        reports = [r for r in reports if max_step is None or r["global_step"] <= max_step * 1.01]
        steps = np.array([r["global_step"] for r in reports]) / 1e6
        for ax, split in zip(axes, ("train", "test"), strict=True):
            ax.plot(steps, [r[split]["progress"] for r in reports], "o-", color=color, label=label)
    axes[0].set(title="Training levels (22): mean progress", xlabel="env steps (M)", ylim=(0, 1))
    axes[1].set(title="Held-out levels (7): mean progress", xlabel="env steps (M)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
