from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import LunaSliceDataset
from utils import ensure_project_dirs, load_config, save_image_grid, save_json


def load_dataset_images(dataset: LunaSliceDataset, limit: int | None = None) -> np.ndarray:
    count = len(dataset) if limit is None else min(len(dataset), limit)
    return np.stack([dataset[index]["image"].numpy() for index in range(count)])


def nearest_neighbors(
    generated: np.ndarray,
    training: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    generated_flat = generated.reshape(len(generated), -1).astype(np.float32)
    best_distances = np.full(len(generated), np.inf, dtype=np.float64)
    best_indices = np.full(len(generated), -1, dtype=np.int64)
    for start in range(0, len(training), batch_size):
        batch = training[start : start + batch_size]
        batch_flat = batch.reshape(len(batch), -1).astype(np.float32)
        distances = np.mean(
            (generated_flat[:, None, :] - batch_flat[None, :, :]) ** 2,
            axis=-1,
        )
        local_indices = distances.argmin(axis=1)
        local_distances = distances[np.arange(len(generated)), local_indices]
        improved = local_distances < best_distances
        best_distances[improved] = local_distances[improved]
        best_indices[improved] = start + local_indices[improved]
    return best_indices, best_distances


def save_neighbor_grid(
    generated: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    path: Path,
) -> None:
    shown = min(16, len(generated))
    rows, pairs_per_row = 4, 4
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(14, 7))
    for index in range(rows * pairs_per_row):
        row, pair = divmod(index, pairs_per_row)
        gen_axis = axes[row, pair * 2]
        real_axis = axes[row, pair * 2 + 1]
        if index < shown:
            gen_axis.imshow((generated[index, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
            real_axis.imshow((neighbors[index, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
            gen_axis.set_title(f"Generated {index + 1}")
            real_axis.set_title(f"Nearest, MSE={distances[index]:.4f}")
        gen_axis.axis("off")
        real_axis.axis("off")
    fig.suptitle("Generated samples and nearest training slices")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pairwise_diversity(images: np.ndarray) -> float:
    flat = images.reshape(len(images), -1)
    distances = []
    for first in range(len(flat)):
        for second in range(first + 1, len(flat)):
            distances.append(float(np.mean(np.abs(flat[first] - flat[second]))))
    return float(np.mean(distances)) if distances else 0.0


def normalized_to_hu(values: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return (values + 1.0) / 2.0 * (hu_max - hu_min) + hu_min


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated 2D CT slices.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generated", default="generated_samples.npy")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ensure_project_dirs(config)
    generated_path = Path(args.generated)
    if not generated_path.is_absolute():
        generated_path = paths["outputs_dir"] / generated_path
    generated = np.load(generated_path).astype(np.float32)
    if generated.ndim == 3:
        generated = generated[:, None]

    processed_dir = config["paths"]["processed_dir"]
    evaluation_config = config["evaluate"]
    training_dataset = LunaSliceDataset(
        processed_dir,
        split="train",
        max_slices=int(evaluation_config["nearest_neighbor_candidates"]),
        seed=int(config["train"]["seed"]),
    )
    training_images = load_dataset_images(training_dataset)
    try:
        real_dataset = LunaSliceDataset(
            processed_dir,
            split="test",
            seed=int(config["train"]["seed"]),
        )
    except RuntimeError:
        try:
            real_dataset = LunaSliceDataset(
                processed_dir,
                split="validation",
                seed=int(config["train"]["seed"]),
            )
        except RuntimeError:
            real_dataset = training_dataset
    real_count = min(16, len(real_dataset))
    rng = np.random.default_rng(int(config["train"]["seed"]))
    real_indices = rng.choice(len(real_dataset), size=real_count, replace=False)
    real_grid_images = np.stack(
        [real_dataset[int(index)]["image"].numpy() for index in real_indices]
    )
    save_image_grid(
        real_grid_images,
        paths["outputs_dir"] / "real_samples.png",
        nrow=4,
        title=f"Random held-out real CT slices ({real_dataset.split})",
    )
    real_histogram_images = load_dataset_images(real_dataset)

    neighbor_indices, neighbor_distances = nearest_neighbors(generated, training_images)
    neighbors = training_images[neighbor_indices]
    save_neighbor_grid(
        generated,
        neighbors,
        neighbor_distances,
        paths["outputs_dir"] / "nearest_neighbors.png",
    )

    hu_min = float(config["preprocess"]["hu_min"])
    hu_max = float(config["preprocess"]["hu_max"])
    real_hu = normalized_to_hu(real_histogram_images, hu_min, hu_max).ravel()
    generated_hu = normalized_to_hu(generated, hu_min, hu_max).ravel()
    bins = int(evaluation_config["histogram_bins"])
    histogram_range = (hu_min, hu_max)
    real_hist, edges = np.histogram(
        real_hu, bins=bins, range=histogram_range, density=True
    )
    generated_hist, _ = np.histogram(
        generated_hu, bins=bins, range=histogram_range, density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]
    histogram_total_variation = float(
        0.5 * np.sum(np.abs(real_hist - generated_hist)) * bin_width
    )
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(centers, real_hist, label="Held-out real", linewidth=2)
    axis.plot(centers, generated_hist, label="Generated", linewidth=2)
    axis.set_xlabel("HU after inverse normalization")
    axis.set_ylabel("Density")
    axis.set_title("Real vs. generated intensity distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        paths["outputs_dir"] / "real_vs_generated_histogram.png",
        dpi=160,
    )
    plt.close(fig)

    failure_indices = np.argsort(neighbor_distances)[-4:][::-1]
    failure_grid = []
    for index in failure_indices:
        failure_grid.extend((generated[index], neighbors[index]))
    save_image_grid(
        np.stack(failure_grid),
        paths["outputs_dir"] / "failure_cases.png",
        nrow=4,
        title="Highest nearest-neighbor errors: generated / nearest pairs",
    )

    metadata_path = paths["outputs_dir"] / "training_metadata.json"
    sampling_path = paths["outputs_dir"] / "sampling_metadata.json"
    training_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    sampling_metadata = (
        json.loads(sampling_path.read_text(encoding="utf-8"))
        if sampling_path.exists()
        else {}
    )
    metrics = {
        "held_out_split": real_dataset.split,
        "held_out_real_count_for_histogram": len(real_histogram_images),
        "nearest_neighbor_training_candidates": len(training_images),
        "generated_count": len(generated),
        "generated_pixel_mean_normalized": float(generated.mean()),
        "generated_pixel_std_normalized": float(generated.std()),
        "real_pixel_mean_normalized": float(real_histogram_images.mean()),
        "real_pixel_std_normalized": float(real_histogram_images.std()),
        "histogram_total_variation": histogram_total_variation,
        "generated_pairwise_mean_absolute_difference": pairwise_diversity(generated),
        "nearest_neighbor_mse_mean": float(neighbor_distances.mean()),
        "nearest_neighbor_mse_min": float(neighbor_distances.min()),
        "nearest_neighbor_mse_max": float(neighbor_distances.max()),
    }
    save_json(metrics, paths["outputs_dir"] / "evaluation_metrics.json")

    table = [
        "# 实验结果表",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 训练 CT 扫描数 | {training_metadata.get('training_scan_count', 'N/A')} |",
        f"| 训练切片数 | {training_metadata.get('training_slice_count', 'N/A')} |",
        f"| 图像尺寸 | {training_metadata.get('resolution', config['preprocess']['resolution'])}×{training_metadata.get('resolution', config['preprocess']['resolution'])} |",
        f"| 模型参数量 | {training_metadata.get('model_parameters', 'N/A')} |",
        f"| Batch size | {training_metadata.get('batch_size', config['train']['batch_size'])} |",
        f"| 训练步数 | {training_metadata.get('completed_steps', 'N/A')} |",
        f"| 训练时长 | {training_metadata.get('elapsed_hms_this_run', 'N/A')} |",
        f"| 最终 loss | {training_metadata.get('final_loss', 'N/A')} |",
        f"| 末 100 步平均 loss | {training_metadata.get('mean_loss_last_100', 'N/A')} |",
        f"| 生成 16 张耗时（秒） | {sampling_metadata.get('generation_seconds', 'N/A')} |",
        f"| 最近邻平均 MSE | {metrics['nearest_neighbor_mse_mean']:.6f} |",
        f"| 灰度直方图 TV 距离 | {metrics['histogram_total_variation']:.6f} |",
        f"| 生成样本两两平均绝对差 | {metrics['generated_pairwise_mean_absolute_difference']:.6f} |",
        "",
        "> 这些指标只用于教学演示中的分布和记忆检查，不代表临床有效性。",
    ]
    (paths["outputs_dir"] / "results_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
