from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config_path = ROOT / "configs" / "improved_128.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config["preprocess"]["resolution"] == 128, "resolution is not 128")
    require(config["train"]["max_steps"] == 10000, "formal step count is not 10000")
    require(config["train"]["batch_size"] == 32, "unexpected batch size")
    require(
        config["paths"]["outputs_dir"] == "outputs/improved_128",
        "improved outputs are not isolated",
    )
    require(
        config["paths"]["checkpoints_dir"] == "checkpoints/improved_128",
        "improved checkpoints are not isolated",
    )

    processed = ROOT / config["paths"]["processed_dir"]
    with (processed / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 6780, f"expected 6780 slices, found {len(rows)}")
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    require(
        {key: len(value) for key, value in split_rows.items()}
        == {"train": 5361, "validation": 652, "test": 767},
        "slice split counts do not match the completed run",
    )
    split_series = {
        key: {row["seriesuid"] for row in value} for key, value in split_rows.items()
    }
    require(
        not (split_series["train"] & split_series["validation"])
        and not (split_series["train"] & split_series["test"])
        and not (split_series["validation"] & split_series["test"]),
        "seriesuid leakage detected",
    )
    require(
        {key: len(value) for key, value in split_series.items()}
        == {"train": 71, "validation": 9, "test": 9},
        "scan split counts do not match the completed run",
    )

    available_slices = [
        processed / row["relative_path"]
        for row in (split_rows["train"][:2] + split_rows["test"][:2])
    ]
    if all(path.exists() for path in available_slices):
        for path in available_slices:
            array = np.load(path)
            require(array.shape == (128, 128), f"wrong slice shape: {path}")
            require(
                array.dtype == np.float32
                and float(array.min()) >= -1.0
                and float(array.max()) <= 1.0,
                f"invalid normalization: {path}",
            )

    outputs = ROOT / config["paths"]["outputs_dir"]
    required_outputs = [
        "generated_samples.png",
        "generated_samples.npy",
        "real_samples.png",
        "training_loss.png",
        "real_vs_generated_histogram.png",
        "nearest_neighbors.png",
        "failure_cases.png",
        "training_metadata.json",
        "sampling_metadata.json",
        "evaluation_metrics.json",
        "sampling_50.png",
        "sampling_100.png",
        "sampling_200.png",
        "sampling_steps_comparison.png",
        "sampling_50.npy",
        "sampling_100.npy",
        "sampling_200.npy",
        "sampling_steps_metrics.json",
    ]
    for name in required_outputs:
        require((outputs / name).exists(), f"missing output: {name}")
    for path in outputs.glob("*.png"):
        with Image.open(path) as image:
            image.verify()

    generated = np.load(outputs / "generated_samples.npy")
    require(generated.shape == (16, 1, 128, 128), "wrong generated array shape")
    require(
        float(generated.min()) >= -1.0 and float(generated.max()) <= 1.0,
        "generated samples are outside [-1, 1]",
    )
    training = json.loads((outputs / "training_metadata.json").read_text("utf-8"))
    require(training["completed_steps"] == 10000, "formal training is incomplete")
    require(training["training_slice_count"] == 5361, "wrong training slice count")
    metrics = json.loads((outputs / "evaluation_metrics.json").read_text("utf-8"))
    require(metrics["generated_count"] == 16, "wrong generated sample count")
    require(
        metrics["nearest_neighbor_training_candidates"] == 5361,
        "nearest-neighbor evaluation did not use the full training split",
    )
    sampling_steps = json.loads(
        (outputs / "sampling_steps_metrics.json").read_text("utf-8")
    )
    require(
        sampling_steps["checkpoint_step"] == 10000
        and sampling_steps["same_initial_noise"] is True
        and sampling_steps["eta"] == 0.0,
        "sampling-step experiment did not use the intended checkpoint/noise setup",
    )
    require(
        set(sampling_steps["timings_seconds"]) == {"50", "100", "200"},
        "sampling-step timings are incomplete",
    )
    for steps in (50, 100, 200):
        array = np.load(outputs / f"sampling_{steps}.npy")
        require(
            array.shape == (16, 1, 128, 128),
            f"wrong sampling_{steps}.npy shape",
        )

    checkpoint_path = ROOT / config["paths"]["checkpoints_dir"] / "latest.pt"
    require(
        checkpoint_path.exists() and checkpoint_path.stat().st_size > 100_000_000,
        "final checkpoint is missing or unexpectedly small",
    )
    checkpoint_validation = "file-size-and-training-metadata"
    checkpoint_step = training["completed_steps"]
    if torch is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint["step"] == 10000, "checkpoint is not step 10000")
        require(len(checkpoint["loss_history"]) == 10000, "loss history is incomplete")
        checkpoint_validation = "torch-load"
        checkpoint_step = checkpoint["step"]

    require((ROOT / "configs" / "default.yaml").exists(), "baseline config is missing")
    require((ROOT / "checkpoints" / "latest.pt").exists(), "baseline checkpoint is missing")
    require(
        (ROOT / "reports" / "experiment_report.md").exists(),
        "baseline report is missing",
    )
    require(
        (ROOT / "outputs" / "generated_samples.png").exists(),
        "baseline output is missing",
    )
    report = (ROOT / "reports" / "improved_128_experiment_report.md").read_text(
        encoding="utf-8"
    )
    require("不能用于诊断" in report, "clinical-use disclaimer is missing")
    for phrase in (
        "增加采样步数有时可以缓解",
        "提高模型容量",
        "结构损失",
        "肺部 mask",
        "2.5D",
        "三维扩散模型",
        "不能只根据采样步数排序效果",
    ):
        require(phrase in report, f"sampling-step discussion is missing: {phrase}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "resolution": 128,
                "scans": {key: len(value) for key, value in split_series.items()},
                "slices": {key: len(value) for key, value in split_rows.items()},
                "checkpoint_step": checkpoint_step,
                "checkpoint_validation": checkpoint_validation,
                "generated_shape": list(generated.shape),
                "histogram_total_variation": metrics["histogram_total_variation"],
                "sampling_steps_seconds": sampling_steps["timings_seconds"],
                "same_initial_noise": sampling_steps["same_initial_noise"],
                "baseline_preserved": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
