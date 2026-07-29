from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D CT slices from random noise.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", default="generated_samples.png")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ensure_project_dirs(config)
    seed_everything(int(config["train"]["seed"]))
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

    count = args.count or int(config["evaluate"]["generated_count"])
    resolution = int(config["preprocess"]["resolution"])
    shape = (count, int(config["model"]["in_channels"]), resolution, resolution)
    start = time.perf_counter()
    with torch.no_grad():
        if config["diffusion"]["sampling_method"].lower() == "ddpm":
            samples = diffusion.ddpm_sample(model, shape, device)
        else:
            samples = diffusion.ddim_sample(
                model,
                shape,
                device,
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                eta=float(config["diffusion"]["ddim_eta"]),
            )
    generation_seconds = time.perf_counter() - start

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = paths["outputs_dir"] / output_path
    save_image_grid(
        samples,
        output_path,
        nrow=4,
        title=f"Generated LUNA16-style CT slices ({config['diffusion']['sampling_method'].upper()})",
    )
    array_path = output_path.with_suffix(".npy")
    np.save(array_path, samples.detach().float().cpu().numpy())
    sampling_metadata = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "array_output": str(array_path),
        "count": count,
        "resolution": resolution,
        "method": config["diffusion"]["sampling_method"],
        "sampling_steps": int(config["diffusion"]["sampling_steps"]),
        "generation_seconds": generation_seconds,
        "seconds_per_image": generation_seconds / count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    save_json(sampling_metadata, paths["outputs_dir"] / "sampling_metadata.json")
    print(json.dumps(sampling_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
