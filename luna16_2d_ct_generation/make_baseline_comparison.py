from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def load_generated(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected [N, 1, H, W], got {array.shape} from {path}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact baseline-vs-improved result figure."
    )
    parser.add_argument(
        "--baseline",
        default="outputs/generated_samples.npy",
    )
    parser.add_argument(
        "--improved",
        default="outputs/improved_128/generated_samples.npy",
    )
    parser.add_argument(
        "--output",
        default="outputs/baseline_vs_improved.png",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15],
    )
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else PROJECT_ROOT / value

    baseline = load_generated(resolve(args.baseline))
    improved = load_generated(resolve(args.improved))
    indices = [index for index in args.indices if index < len(baseline) and index < len(improved)]
    if not indices:
        raise ValueError("No valid comparison indices")

    fig, axes = plt.subplots(len(indices), 2, figsize=(7.2, 3.4 * len(indices)))
    axes = np.asarray(axes).reshape(len(indices), 2)
    for row, index in enumerate(indices):
        for column, (images, label) in enumerate(
            ((baseline, "64×64 baseline"), (improved, "128×128 improved"))
        ):
            axis = axes[row, column]
            axis.imshow(
                np.clip((images[index, 0] + 1.0) / 2.0, 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"Generated sample {index + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("LUNA16 2D DDPM: baseline vs. improved experiment", fontsize=15)
    fig.tight_layout()
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
