"""Held-out progress over training, mean of 3 seeds per family: python results/10m-study/plot_families.py"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FAMILIES = [  # (label, run dirs, colour, marker): fixed categorical order
    ("full screen, uniform", ["E11-tiles2-uniform-10m", "E15-tiles2-uniform-10m-seed*"], "#2a78d6", "o"),
    ("full screen, PLR (β=0.1)", ["E12-tiles2-plr-10m", "E16-tiles2-plr-10m-seed*"], "#eb6834", "s"),
    ("ego window 4/12, uniform", ["E20-tiles2-ego4x12-10m-seed*"], "#1baf7a", "^"),
]


def curve(run: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    rs = sorted((json.loads(f.read_text()) for f in run.glob("*.json") if f.name != "config.json"),
                key=lambda r: r["global_step"])  # fmt: skip
    return np.array([round(r["global_step"] / 1e6) for r in rs]), np.array(
        [100 * r[split]["progress"] for r in rs]
    )


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for ax, split, title in zip(
    axes, ["test", "train"], ["Held-out levels (7)", "Training levels (22)"], strict=True
):
    for label, pats, color, marker in FAMILIES:
        runs = [Path(p) for pat in pats for p in sorted(glob.glob(str(HERE / pat)))]
        steps, ys = zip(*(curve(r, split) for r in runs), strict=True)
        y = np.mean(ys, axis=0)
        ax.fill_between(steps[0], np.min(ys, axis=0), np.max(ys, axis=0), color=color, alpha=0.12, lw=0)
        ax.plot(
            steps[0],
            y,
            color=color,
            lw=2,
            marker=marker,
            ms=8,
            mec="#fcfcfb",
            mew=2,
            label=f"{label} (n={len(runs)})",
        )
        if split == "test":  # training curves overlap; their end labels would collide
            ax.annotate(f"{y[-1]:.1f}", (steps[0][-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#0b0b0b")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_xlabel("env steps (millions)", color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.set_xlim(0.6, 10.9)
axes[0].set_ylabel("progress (% of level), mean of seeds", color="#52514e")
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Egocentric window raises held-out progress without costing training levels (band = min–max over seeds)",
             x=0.01, ha="left", fontsize=11, color="#0b0b0b")  # fmt: skip
fig.patch.set_facecolor("#fcfcfb")
for ax in axes:
    ax.set_facecolor("#fcfcfb")
fig.tight_layout()
fig.savefig(HERE / "families_10m.png", dpi=130)
print("saved", HERE / "families_10m.png")
