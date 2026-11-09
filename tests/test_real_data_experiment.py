"""Checks for the real-data analysis and the disabled legacy execution."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from saber.cohort.release_cohort import assemble_release_cohort

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_execution_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Historical cohort unavailable"):
        assemble_release_cohort(None, None)  # type: ignore[arg-type]


def test_real_table_held_out_split_and_metrics(tmp_path: Path) -> None:
    table = ROOT / "data/public/brain_organoid_author_table/df_pred.csv"
    if not table.exists():
        pytest.skip("Real dataset table has not been downloaded")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_brain_organoid_growth.py"),
                    "--data", str(table.parent), "--out", str(tmp_path)], check=True, capture_output=True)
    with (tmp_path / "selected_observations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36
    assert sum(r["role"] == "test" for r in rows) == 4
    assert all(int(r["day"]) == 30 for r in rows if r["role"] == "test")
    with (tmp_path / "day30_predictions.csv").open(newline="") as f:
        predictions = list(csv.DictReader(f))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(predictions) == 4
    assert summary["linear_mae_um2"] == pytest.approx(
        sum(abs(float(p["linear_prediction_um2"]) - float(p["actual_um2"])) for p in predictions) / 4
    )
