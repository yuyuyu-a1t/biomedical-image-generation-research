from __future__ import annotations

import csv
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = config
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return config


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_project_dirs(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: resolve_project_path(value) for key, value in config["paths"].items()}
    for key in ("processed_dir", "outputs_dir", "checkpoints_dir", "reports_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def to_display_range(images: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = images.detach().float().cpu().numpy()
    images = np.asarray(images)
    if images.ndim == 4:
        images = images[:, 0]
    return np.clip((images + 1.0) / 2.0, 0.0, 1.0)


def save_image_grid(
    images: torch.Tensor | np.ndarray,
    path: str | Path,
    nrow: int | None = None,
    title: str | None = None,
) -> None:
    images = to_display_range(images)
    count = len(images)
    nrow = nrow or int(math.ceil(math.sqrt(count)))
    ncol = int(math.ceil(count / nrow))
    fig, axes = plt.subplots(ncol, nrow, figsize=(2.1 * nrow, 2.1 * ncol))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for index, image in enumerate(images):
        axes[index].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(loss_history: list[float], path: str | Path) -> None:
    if not loss_history:
        return
    values = np.asarray(loss_history, dtype=np.float64)
    window = min(100, max(1, len(values) // 20))
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(np.arange(1, len(values) + 1), values, alpha=0.25, label="step loss")
    axis.plot(
        np.arange(window, window + len(smoothed)),
        smoothed,
        linewidth=2,
        label=f"moving average ({window})",
    )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Noise-prediction MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }

    def copy_to(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.data.copy_(self.shadow[name].data)

    def restore(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: tensor.clone() for name, tensor in state["shadow"].items()}
