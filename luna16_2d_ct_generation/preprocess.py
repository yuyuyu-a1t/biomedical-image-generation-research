from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import (
    ensure_project_dirs,
    load_config,
    resolve_project_path,
    save_csv,
    save_image_grid,
    save_json,
    seed_everything,
)


def require_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "SimpleITK is required for .mhd/.raw CT files. "
            "Install it with: pip install SimpleITK"
        ) from error
    return sitk


def stable_uid(path: Path) -> str:
    stem = path.stem
    if stem:
        return stem
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def split_series(
    seriesuids: list[str],
    ratios: list[float],
    seed: int,
) -> dict[str, list[str]]:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("preprocess.split_ratios must contain three values summing to 1")
    shuffled = sorted(seriesuids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count == 1:
        return {"train": shuffled, "validation": [], "test": []}
    test_count = max(1, int(round(count * ratios[2])))
    validation_count = max(1, int(round(count * ratios[1]))) if count >= 3 else 0
    while test_count + validation_count >= count:
        if test_count > 1:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    train_count = count - validation_count - test_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def resize_center_crop(slice_hu: np.ndarray, resolution: int) -> np.ndarray:
    height, width = slice_hu.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = slice_hu[top : top + side, left : left + side]
    image = Image.fromarray(cropped.astype(np.float32))
    image = image.resize((resolution, resolution), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def candidate_indices(
    volume_hu: np.ndarray,
    config: dict,
) -> list[tuple[int, float, float]]:
    stride = int(config["slice_stride"])
    selected: list[tuple[int, float, float]] = []
    for index in range(0, volume_hu.shape[0], stride):
        slice_hu = volume_hu[index]
        non_air_ratio = float(np.mean(slice_hu > float(config["non_air_threshold_hu"])))
        lung_ratio = float(
            np.mean(
                (slice_hu > float(config["lung_hu_low"]))
                & (slice_hu < float(config["lung_hu_high"]))
            )
        )
        if (
            non_air_ratio >= float(config["min_non_air_ratio"])
            and lung_ratio >= float(config["min_lung_ratio"])
        ):
            selected.append((index, non_air_ratio, lung_ratio))
    maximum = config.get("max_slices_per_scan")
    if maximum is not None and len(selected) > int(maximum):
        positions = np.linspace(0, len(selected) - 1, int(maximum)).round().astype(int)
        selected = [selected[position] for position in positions]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LUNA16 CT volumes into 2D slices.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value, e.g. preprocess.max_scans=2",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ensure_project_dirs(config)
    preprocess_config = config["preprocess"]
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    sitk = require_simpleitk()

    data_root = resolve_project_path(config["paths"]["data_root"])
    scan_paths = sorted(data_root.rglob("*.mhd"))
    scan_paths = [
        path
        for path in scan_paths
        if "seg-lungs" not in str(path).lower() and "mask" not in str(path).lower()
    ]
    max_scans = preprocess_config.get("max_scans")
    if max_scans is not None:
        scan_paths = scan_paths[: int(max_scans)]
    if not scan_paths:
        raise FileNotFoundError(
            f"No .mhd scans found under {data_root}. Expected paths such as "
            f"{data_root / 'subset0' / '<seriesuid>.mhd'}"
        )

    series_to_path = {stable_uid(path): path for path in scan_paths}
    splits = split_series(
        list(series_to_path),
        list(preprocess_config["split_ratios"]),
        seed,
    )
    split_lookup = {
        seriesuid: split for split, seriesuids in splits.items() for seriesuid in seriesuids
    }
    processed_dir = paths["processed_dir"]
    slice_root = processed_dir / "slices"
    for split in ("train", "validation", "test"):
        (slice_root / split).mkdir(parents=True, exist_ok=True)

    hu_min = float(preprocess_config["hu_min"])
    hu_max = float(preprocess_config["hu_max"])
    resolution = int(preprocess_config["resolution"])
    rows: list[dict] = []
    scan_summaries: list[dict] = []
    preview_slices: list[np.ndarray] = []
    start_time = time.perf_counter()

    for seriesuid, scan_path in tqdm(series_to_path.items(), desc="CT scans"):
        image = sitk.ReadImage(str(scan_path))
        volume_hu = sitk.GetArrayFromImage(image).astype(np.float32)
        spacing_xyz = tuple(float(value) for value in image.GetSpacing())
        selected = candidate_indices(volume_hu, preprocess_config)
        split = split_lookup[seriesuid]

        for slice_index, non_air_ratio, lung_ratio in selected:
            slice_hu = np.clip(volume_hu[slice_index], hu_min, hu_max)
            slice_hu = resize_center_crop(slice_hu, resolution)
            normalized = (slice_hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
            normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
            relative_path = (
                Path("slices") / split / f"{seriesuid}_{slice_index:04d}.npy"
            )
            np.save(processed_dir / relative_path, normalized)
            rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "seriesuid": seriesuid,
                    "slice_index": slice_index,
                    "split": split,
                    "non_air_ratio": f"{non_air_ratio:.6f}",
                    "lung_ratio": f"{lung_ratio:.6f}",
                    "source_mhd": str(scan_path.relative_to(data_root)).replace("\\", "/"),
                }
            )
            if len(preview_slices) < 32:
                preview_slices.append(normalized)

        scan_summary = {
            "seriesuid": seriesuid,
            "source_mhd": str(scan_path.relative_to(data_root)).replace("\\", "/"),
            "split": split,
            "shape_zyx": list(map(int, volume_hu.shape)),
            "spacing_xyz_mm": list(spacing_xyz),
            "hu_min_observed": float(volume_hu.min()),
            "hu_max_observed": float(volume_hu.max()),
            "candidate_slices_before_stride": int(volume_hu.shape[0]),
            "retained_slices": len(selected),
        }
        scan_summaries.append(scan_summary)
        print(json.dumps(scan_summary, ensure_ascii=False))

    rows.sort(key=lambda row: (row["split"], row["seriesuid"], int(row["slice_index"])))
    save_csv(
        rows,
        processed_dir / "manifest.csv",
        [
            "relative_path",
            "seriesuid",
            "slice_index",
            "split",
            "non_air_ratio",
            "lung_ratio",
            "source_mhd",
        ],
    )
    save_json(splits, processed_dir / "splits.json")
    retained_by_split = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    summary = {
        "data_root": str(data_root),
        "processed_dir": str(processed_dir),
        "scan_count": len(scan_paths),
        "scan_count_by_split": {key: len(value) for key, value in splits.items()},
        "retained_slice_count": len(rows),
        "retained_slice_count_by_split": retained_by_split,
        "resolution": resolution,
        "hu_window": [hu_min, hu_max],
        "slice_stride": int(preprocess_config["slice_stride"]),
        "filter": {
            "non_air_threshold_hu": float(preprocess_config["non_air_threshold_hu"]),
            "min_non_air_ratio": float(preprocess_config["min_non_air_ratio"]),
            "lung_hu_range": [
                float(preprocess_config["lung_hu_low"]),
                float(preprocess_config["lung_hu_high"]),
            ],
            "min_lung_ratio": float(preprocess_config["min_lung_ratio"]),
        },
        "elapsed_seconds": time.perf_counter() - start_time,
        "scans": scan_summaries,
    }
    save_json(summary, processed_dir / "preprocess_summary.json")
    if preview_slices:
        save_image_grid(
            np.stack(preview_slices[:16]),
            paths["outputs_dir"] / "preprocess_real_samples.png",
            nrow=4,
            title="Preprocessed LUNA16 axial CT slices",
        )
    print(json.dumps({key: value for key, value in summary.items() if key != "scans"}, indent=2))


if __name__ == "__main__":
    main()
