from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import resolve_project_path, seed_worker


class LunaSliceDataset(Dataset):
    """Dataset backed by the manifest created by preprocess.py."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        max_slices: int | None = None,
        seed: int = 42,
    ) -> None:
        self.processed_dir = resolve_project_path(processed_dir)
        manifest_path = self.processed_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run preprocess.py before training."
            )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not rows:
            raise RuntimeError(f"No slices found for split='{split}' in {manifest_path}")
        if max_slices is not None and len(rows) > int(max_slices):
            rng = random.Random(seed)
            rows = rng.sample(rows, int(max_slices))
            rows.sort(key=lambda row: (row["seriesuid"], int(row["slice_index"])))
        self.rows = rows
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        array = np.load(self.processed_dir / row["relative_path"]).astype(np.float32)
        array = np.clip(array, -1.0, 1.0)
        return {
            "image": torch.from_numpy(array).unsqueeze(0),
            "seriesuid": row["seriesuid"],
            "slice_index": int(row["slice_index"]),
        }

    @property
    def scan_count(self) -> int:
        return len({row["seriesuid"] for row in self.rows})


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle and len(dataset) >= batch_size,
    )
