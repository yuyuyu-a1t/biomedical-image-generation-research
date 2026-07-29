from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_model
from utils import PROJECT_ROOT, count_parameters, load_config, resolve_project_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config = load_config("configs/default.yaml")
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    outputs_dir = resolve_project_path(config["paths"]["outputs_dir"])
    checkpoints_dir = resolve_project_path(config["paths"]["checkpoints_dir"])
    report_path = PROJECT_ROOT / "reports" / "experiment_report.md"
    bib_path = PROJECT_ROOT / "reports" / "references.bib"

    required_outputs = [
        "real_samples.png",
        "generated_samples.png",
        "training_loss.png",
        "real_vs_generated_histogram.png",
        "nearest_neighbors.png",
        "failure_cases.png",
        "training_metadata.json",
        "sampling_metadata.json",
        "evaluation_metrics.json",
        "results_table.md",
    ]
    for name in required_outputs:
        path = outputs_dir / name
        require(path.exists() and path.stat().st_size > 0, f"Missing output: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()

    preprocess_summary = json.loads(
        (processed_dir / "preprocess_summary.json").read_text(encoding="utf-8")
    )
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    with (processed_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(preprocess_summary["scan_count"] == 89, "Expected 89 subset0 scans")
    require(len(rows) == preprocess_summary["retained_slice_count"], "Manifest count mismatch")
    split_sets = {name: set(values) for name, values in splits.items()}
    require(split_sets["train"].isdisjoint(split_sets["validation"]), "Train/val leakage")
    require(split_sets["train"].isdisjoint(split_sets["test"]), "Train/test leakage")
    require(split_sets["validation"].isdisjoint(split_sets["test"]), "Val/test leakage")
    split_lookup = {
        seriesuid: split for split, seriesuids in splits.items() for seriesuid in seriesuids
    }
    observed_min, observed_max = 1.0, -1.0
    for row in rows:
        require(split_lookup[row["seriesuid"]] == row["split"], "Manifest split mismatch")
        array_path = processed_dir / row["relative_path"]
        require(array_path.exists(), f"Missing processed slice: {array_path}")
        array = np.load(array_path, mmap_mode="r")
        require(array.shape == (64, 64), f"Unexpected slice shape: {array.shape}")
        require(array.dtype == np.float32, f"Unexpected dtype: {array.dtype}")
        observed_min = min(observed_min, float(array.min()))
        observed_max = max(observed_max, float(array.max()))
    require(observed_min >= -1.00001 and observed_max <= 1.00001, "Normalization out of range")

    training = json.loads(
        (outputs_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    sampling = json.loads(
        (outputs_dir / "sampling_metadata.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (outputs_dir / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    require(training["completed_steps"] == 3000, "Formal training did not reach 3000 steps")
    require(training["training_slice_count"] == 4524, "Training slice count mismatch")
    require(training["training_scan_count"] == 71, "Training scan count mismatch")
    require(evaluation["held_out_split"] == "test", "Evaluation did not use test split")
    require(evaluation["held_out_real_count_for_histogram"] == 706, "Test histogram count mismatch")
    require(sampling["count"] >= 16, "Need at least 16 generated samples")

    checkpoint_path = checkpoints_dir / "latest.pt"
    require(checkpoint_path.exists(), "Missing latest checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model"])
    require(count_parameters(model) == training["model_parameters"], "Parameter count mismatch")
    require(checkpoint["step"] == training["completed_steps"], "Checkpoint step mismatch")

    report = report_path.read_text(encoding="utf-8")
    for section in range(1, 13):
        require(re.search(rf"^## {section}\.", report, re.MULTILINE) is not None, f"Missing report section {section}")
    for literal in ("89", "5733", "4524", "7,522,305", "3000", "3分21秒", "0.022714", "0.732"):
        require(literal in report, f"Report is missing verified value: {literal}")
    for warning in ("教学", "不能用于任何临床诊断", "没有验证病灶真实性"):
        require(warning in report, f"Missing safety boundary: {warning}")
    citations = set(re.findall(r"@([A-Za-z0-9_-]+)", report))
    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib_path.read_text(encoding="utf-8")))
    require(citations <= bib_keys, f"Unknown citations: {sorted(citations - bib_keys)}")

    python_files = list(PROJECT_ROOT.glob("*.py"))
    for path in python_files:
        if path.name == "verify_project.py":
            continue
        text = path.read_text(encoding="utf-8")
        require("autodl-tmp" not in text, f"Hard-coded server path in {path.name}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "scans": preprocess_summary["scan_count"],
                "slices": len(rows),
                "split_scans": preprocess_summary["scan_count_by_split"],
                "split_slices": preprocess_summary["retained_slice_count_by_split"],
                "normalized_range": [observed_min, observed_max],
                "checkpoint_step": checkpoint["step"],
                "model_parameters": count_parameters(model),
                "verified_outputs": len(required_outputs),
                "report_citations": sorted(citations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
