"""Evaluation helpers: aggregate per-level results into summary metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def summarize(results: Iterable[dict]) -> dict:
    """Per-level and overall progress / completion statistics.

    Overall numbers average over levels (not episodes) so every level counts equally.
    """
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(r)
    levels = {}
    for name in sorted(by_level, key=lambda s: tuple(int(p) for p in s.split("-"))):
        eps = by_level[name]
        levels[name] = {
            "progress": float(np.mean([e["progress"] for e in eps])),
            "completion": float(np.mean([e["flag_get"] for e in eps])),
            "return": float(np.mean([e["r"] for e in eps])),
            "length": float(np.mean([e["l"] for e in eps])),
            "episodes": len(eps),
        }
    summary = {
        "progress": float(np.mean([v["progress"] for v in levels.values()])) if levels else 0.0,
        "completion": float(np.mean([v["completion"] for v in levels.values()])) if levels else 0.0,
        "levels": levels,
    }
    return summary


def format_table(summary: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'level':>6} {'progress':>9} {'complete':>9} {'return':>8} {'eps':>4}")
    for name, v in summary["levels"].items():
        lines.append(
            f"{name:>6} {v['progress']:>9.1%} {v['completion']:>9.1%} {v['return']:>8.1f} {v['episodes']:>4d}"
        )
    lines.append(f"{'mean':>6} {summary['progress']:>9.1%} {summary['completion']:>9.1%}")
    return "\n".join(lines)
