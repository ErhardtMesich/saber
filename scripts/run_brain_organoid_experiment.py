#!/usr/bin/env python3
"""Evaluate a fixed, unsupervised segmentation baseline on real organoid images.

This feasibility experiment is independent of the historical paper results.
It does not create microenvironment labels, dose responses or logged policies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def otsu_threshold(image: np.ndarray) -> int:
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    cumulative = np.cumsum(hist)
    weighted = np.cumsum(hist * np.arange(256))
    total, total_weight = cumulative[-1], weighted[-1]
    valid = (cumulative > 0) & (cumulative < total)
    score = np.full(256, -1.0)
    score[valid] = (total_weight * cumulative[valid] - weighted[valid] * total) ** 2 / (
        cumulative[valid] * (total - cumulative[valid])
    )
    return int(np.argmax(score))


def segment(image: np.ndarray) -> tuple[np.ndarray, int]:
    # Fixed baseline: blur, global Otsu dark-foreground threshold, largest component.
    blurred = np.clip(ndimage.gaussian_filter(image.astype(float), sigma=2), 0, 255).astype(np.uint8)
    threshold = otsu_threshold(blurred)
    binary = blurred < threshold
    components, count = ndimage.label(binary)
    if count == 0:
        return np.zeros(image.shape, dtype=bool), threshold
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = ndimage.binary_fill_holes(components == np.argmax(sizes))
    return np.asarray(mask, dtype=bool), threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/public/brain_organoid_zenodo_10301912"))
    parser.add_argument("--out", type=Path, default=Path("results/real_brain_organoids"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    for entry in manifest["files"]:
        content = (args.data / entry["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {entry['path']}")
    rows = []
    for image_id in manifest["image_ids"]:
        image_path = args.data / "imgs" / f"{image_id}.jpg"
        label_path = args.data / "labels" / f"{image_id}.npy"
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)
        truth = np.load(label_path, allow_pickle=False).astype(bool)
        if image.shape != truth.shape:
            raise ValueError(f"Image and label shapes differ for {image_id}: {image.shape} vs {truth.shape}")
        prediction, threshold = segment(image)
        intersection = int(np.count_nonzero(prediction & truth))
        pred_area = int(prediction.sum())
        truth_area = int(truth.sum())
        union = pred_area + truth_area - intersection
        rows.append({"image_id": image_id, "height": image.shape[0], "width": image.shape[1],
                     "threshold": threshold, "truth_area_pixels": truth_area, "predicted_area_pixels": pred_area,
                     "dice": 2 * intersection / (pred_area + truth_area) if pred_area + truth_area else 1.0,
                     "iou": intersection / union if union else 1.0})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "segmentation_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset_doi": manifest["dataset_doi"], "image_ids": manifest["image_ids"],
               "n_images": len(rows), "n_organoids": len({x.split('_')[0] for x in manifest['image_ids']}),
               "method": "Gaussian sigma 2, Otsu dark foreground, largest connected component, fill holes",
               "mean_dice": float(np.mean([r["dice"] for r in rows])),
               "mean_iou": float(np.mean([r["iou"] for r in rows])),
               "limitations": "Brain organoid images; author masks are reference labels. Descriptive feasibility test, no model training or held-out evaluation. Not liver cancer, drug response, a closed-loop policy, or a reproduction of the manuscript's historical numbers.",
               "files": manifest["files"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("files", "image_ids")}, indent=2))


if __name__ == "__main__":
    main()
