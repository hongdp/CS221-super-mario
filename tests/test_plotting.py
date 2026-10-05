import json

import pytest

from mario_rl.cli import main as cli_main

pytest.importorskip("matplotlib")


def write_reports(directory, steps):
    directory.mkdir()
    for step in steps:
        summary = {"progress": 0.1 + step / 1e7, "completion": step / 2e7, "levels": {}}
        report = {"global_step": step, "train": summary, "test": summary}
        (directory / f"ckpt_{step}.json").write_text(json.dumps(report))


def test_plot_evals_and_compare(tmp_path):
    write_reports(tmp_path / "a", [0, 1_000_000, 2_000_000])
    write_reports(tmp_path / "b", [1_000_000, 2_000_000, 3_000_000])
    cli_main(["plot", str(tmp_path / "a"), "--evals", "--output", str(tmp_path / "gen.png")])
    cli_main(["compare", str(tmp_path / "cmp.png"), f"A={tmp_path / 'a'}", f"B={tmp_path / 'b'}",
              "--max-step", "2000000"])  # fmt: skip
    assert (tmp_path / "gen.png").stat().st_size > 0
    assert (tmp_path / "cmp.png").stat().st_size > 0
