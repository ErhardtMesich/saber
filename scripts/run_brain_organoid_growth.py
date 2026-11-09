#!/usr/bin/env python3
"""Predict held-out day-30 measured organoid area from earlier real observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ORGANOIDS = ("3", "23", "41", "50")  # one per clone; fixed before analysis
TRAIN_DAYS = (2, 5, 8, 10, 16, 19, 22, 25)  # day 12 embedding changes are excluded
TEST_DAY = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/public/brain_organoid_author_table"))
    parser.add_argument("--out", type=Path, default=Path("results/real_brain_organoid_growth"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    table = (args.data / "df_pred.csv").read_bytes()
    if hashlib.sha256(table).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Downloaded table SHA-256 mismatch")
    with (args.data / "df_pred.csv").open(newline="") as f:
        source_rows = list(csv.DictReader(f))
    rows = []
    predictions = []
    for org in ORGANOIDS:
        selected = [r for r in source_rows if r["org_id"] == org and r["Imaging"] == "LabA"
                    and int(r["Day"]) in (*TRAIN_DAYS, TEST_DAY)]
        by_day = {int(r["Day"]): r for r in selected}
        if len(selected) != len(TRAIN_DAYS) + 1 or set(by_day) != {*TRAIN_DAYS, TEST_DAY}:
            raise ValueError(f"Missing or duplicated selected records for organoid {org}")
        for day in (*TRAIN_DAYS, TEST_DAY):
            r = by_day[day]
            rows.append({"org_id": org, "clone": r["Clone"], "lab": r["Imaging"], "day": day,
                         "image_id": r["img_id"], "annotated_area_um2": float(r["org_size_mikrometer2"]),
                         "role": "test" if day == TEST_DAY else "train"})
        x = np.asarray(TRAIN_DAYS, dtype=float)
        y = np.asarray([float(by_day[d]["org_size_mikrometer2"]) for d in TRAIN_DAYS])
        slope, intercept = np.polyfit(x, y, 1)
        actual = float(by_day[TEST_DAY]["org_size_mikrometer2"])
        predicted = float(slope * TEST_DAY + intercept)
        last = float(y[-1])
        predictions.append({"org_id": org, "clone": by_day[TEST_DAY]["Clone"], "test_day": TEST_DAY,
                            "actual_um2": actual, "linear_prediction_um2": predicted,
                            "last_observation_um2": last, "linear_absolute_error_um2": abs(predicted - actual),
                            "last_observation_absolute_error_um2": abs(last - actual)})
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, records in (("selected_observations.csv", rows), ("day30_predictions.csv", predictions)):
        with (args.out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    summary = {"source": manifest, "cohort": "Four brain organoids, one from each clone, Lab A",
               "train_days": TRAIN_DAYS, "test_day": TEST_DAY, "n_organoids": len(predictions),
               "n_training_observations": len(ORGANOIDS) * len(TRAIN_DAYS),
               "method": "Per-organoid ordinary least-squares linear fit to annotated area vs day; extrapolate to day 30",
               "comparator": "Last observed (day 25) area",
               "linear_mae_um2": float(np.mean([r["linear_absolute_error_um2"] for r in predictions])),
               "last_observation_mae_um2": float(np.mean([r["last_observation_absolute_error_um2"] for r in predictions])),
               "limitations": "Four hand-selected brain organoids; annotation-derived area, not raw images. This is a small descriptive held-out-timepoint experiment, not liver cancer, drug response, closed-loop policy evaluation, or replication of paper results."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != 'source'}, indent=2))


if __name__ == "__main__":
    main()
