from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_diffusion, build_model
from utils import (
    ensure_project_dirs,
    load_config,
    save_image_grid,
    save_json,
    seed_everything,
)


def image_statistics(images: np.ndarray) -> dict[str, float]:
    pixels = images[:, 0].astype(np.float32)
    gradient_x = np.abs(np.diff(pixels, axis=2))
    gradient_y = np.abs(np.diff(pixels, axis=1))
    center = pixels[:, 1:-1, 1:-1]
    laplacian = (
        pixels[:, 1:-1, :-2]
        + pixels[:, 1:-1, 2:]
        + pixels[:, :-2, 1:-1]
        + pixels[:, 2:, 1:-1]
        - 4.0 * center
    )
    return {
        "pixel_mean": float(pixels.mean()),
        "pixel_std": float(pixels.std()),
        "mean_absolute_gradient": float(
            (gradient_x.mean() + gradient_y.mean()) / 2.0
        ),
        "mean_absolute_laplacian": float(np.abs(laplacian).mean()),
    }


def save_comparison(
    results: dict[int, np.ndarray],
    timings: dict[int, float],
    output_path: Path,
) -> None:
    step_counts = sorted(results)
    selected_indices = [0, 5, 10, 15]
    fig, axes = plt.subplots(
        len(selected_indices),
        len(step_counts),
        figsize=(3.4 * len(step_counts), 3.4 * len(selected_indices)),
    )
    for row, sample_index in enumerate(selected_indices):
        for column, steps in enumerate(step_counts):
            axis = axes[row, column]
            image = np.clip((results[steps][sample_index, 0] + 1.0) / 2.0, 0, 1)
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axis.set_title(f"{steps}-step DDIM\n{timings[steps]:.3f} s / 16 images")
            if column == 0:
                axis.set_ylabel(f"Fixed noise sample {sample_index + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Same initial noise: DDIM sampling-step comparison", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 50/100/200-step DDIM sampling from one checkpoint."
    )
    parser.add_argument("--config", default="configs/improved_128.yaml")
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200])
    args = parser.parse_args()

    config = load_config(args.config)
    paths = ensure_project_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = paths["checkpoints_dir"] / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint:
        for name, parameter in model.named_parameters():
            if name in checkpoint["ema"]["shadow"]:
                parameter.data.copy_(checkpoint["ema"]["shadow"][name].to(device))
    model.eval()
    diffusion = build_diffusion(config["diffusion"]).to(device)

    count = int(args.count)
    resolution = int(config["preprocess"]["resolution"])
    shape = (count, int(config["model"]["in_channels"]), resolution, resolution)
    seed = int(config["train"]["seed"])

    # Warm up kernels before timing. The experiment seeds again afterwards.
    with torch.inference_mode():
        diffusion.ddim_sample(model, (1, 1, resolution, resolution), device, 5, 0.0)
    if device.type == "cuda":
        torch.cuda.synchronize()

    seed_everything(seed)
    initial_noise_probe = torch.randn(shape, device=device)
    noise_sha256 = hashlib.sha256(
        initial_noise_probe.detach().float().cpu().numpy().tobytes()
    ).hexdigest()
    del initial_noise_probe

    results: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    statistics: dict[int, dict[str, float]] = {}
    for steps in args.steps:
        # ddim_sample's first operation is torch.randn. Resetting the RNG here
        # makes that tensor identical for every deterministic eta=0 run.
        seed_everything(seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            samples = diffusion.ddim_sample(
                model,
                shape,
                device,
                sampling_steps=int(steps),
                eta=0.0,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        array = samples.detach().float().cpu().numpy()
        results[int(steps)] = array
        timings[int(steps)] = elapsed
        statistics[int(steps)] = image_statistics(array)
        output_path = paths["outputs_dir"] / f"sampling_{steps}.png"
        save_image_grid(
            array,
            output_path,
            nrow=4,
            title=f"DDIM {steps} steps — fixed seed {seed}",
        )
        np.save(output_path.with_suffix(".npy"), array)

    pairwise_differences: dict[str, dict[str, float]] = {}
    ordered_steps = sorted(results)
    for first, second in zip(ordered_steps[:-1], ordered_steps[1:]):
        difference = results[first] - results[second]
        pairwise_differences[f"{first}_vs_{second}"] = {
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "mean_squared_difference": float(np.mean(difference**2)),
        }

    comparison_path = paths["outputs_dir"] / "sampling_steps_comparison.png"
    save_comparison(results, timings, comparison_path)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "seed": seed,
        "same_initial_noise": True,
        "initial_noise_sha256": noise_sha256,
        "eta": 0.0,
        "count": count,
        "resolution": resolution,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timings_seconds": {str(key): value for key, value in timings.items()},
        "seconds_per_image": {
            str(key): value / count for key, value in timings.items()
        },
        "image_statistics": {
            str(key): value for key, value in statistics.items()
        },
        "pairwise_output_differences": pairwise_differences,
        "comparison_indices_zero_based": [0, 5, 10, 15],
    }
    save_json(metadata, paths["outputs_dir"] / "sampling_steps_metrics.json")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
