from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import LunaSliceDataset, create_dataloader
from model import build_diffusion, build_model
from utils import (
    EMA,
    atomic_torch_save,
    count_parameters,
    ensure_project_dirs,
    format_seconds,
    load_config,
    plot_training_loss,
    save_image_grid,
    save_json,
    seed_everything,
)


def make_checkpoint(
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    loss_history: list[float],
    config: dict,
    metadata: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "loss_history": loss_history,
        "config": config,
        "metadata": metadata,
    }


@torch.no_grad()
def sample_with_ema(
    model: torch.nn.Module,
    ema: EMA,
    diffusion: torch.nn.Module,
    count: int,
    resolution: int,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    ema.store(model)
    ema.copy_to(model)
    model.eval()
    shape = (count, int(config["model"]["in_channels"]), resolution, resolution)
    diffusion_config = config["diffusion"]
    if diffusion_config["sampling_method"].lower() == "ddpm":
        images = diffusion.ddpm_sample(model, shape, device)
    else:
        images = diffusion.ddim_sample(
            model,
            shape,
            device,
            sampling_steps=int(diffusion_config["sampling_steps"]),
            eta=float(diffusion_config["ddim_eta"]),
        )
    model.train()
    ema.restore(model)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small 2D DDPM on LUNA16 slices.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.resume:
        config["train"]["resume"] = args.resume
    paths = ensure_project_dirs(config)
    train_config = config["train"]
    seed = int(train_config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_config["mixed_precision"]) and device.type == "cuda"

    dataset = LunaSliceDataset(
        config["paths"]["processed_dir"],
        split="train",
        max_slices=train_config.get("max_train_slices"),
        seed=seed,
    )
    loader = create_dataloader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        seed=seed,
        shuffle=True,
    )
    model = build_model(config["model"]).to(device)
    diffusion = build_diffusion(config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, float(train_config["ema_decay"]))

    step = 0
    loss_history: list[float] = []
    resume_path = train_config.get("resume")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = paths["checkpoints_dir"] / resume_path
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        ema.shadow = {name: value.to(device) for name, value in ema.shadow.items()}
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        loss_history = list(checkpoint["loss_history"])
        print(f"Resumed from {resume_path} at step {step}")

    max_steps = int(train_config["max_steps"])
    resolution = int(config["preprocess"]["resolution"])
    model_parameters = count_parameters(model)
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision": use_amp,
        "training_slice_count": len(dataset),
        "training_scan_count": dataset.scan_count,
        "resolution": resolution,
        "batch_size": int(train_config["batch_size"]),
        "model_parameters": model_parameters,
        "start_step": step,
        "target_steps": max_steps,
    }
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    start_time = time.perf_counter()
    model.train()
    progress = tqdm(total=max_steps, initial=step, desc="DDPM training")
    loader_iterator = iter(loader)
    while step < max_steps:
        try:
            batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            batch = next(loader_iterator)
        clean_images = batch["image"].to(device, non_blocking=True)
        timesteps = torch.randint(
            0, diffusion.timesteps, (clean_images.shape[0],), device=device
        )
        noise = torch.randn_like(clean_images)
        noisy_images = diffusion.q_sample(clean_images, timesteps, noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_config["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        step += 1
        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        progress.update(1)
        if step % int(train_config["log_every"]) == 0:
            recent = loss_history[-int(train_config["log_every"]) :]
            progress.set_postfix(loss=f"{sum(recent) / len(recent):.5f}")

        should_sample = step % int(train_config["sample_every"]) == 0 or step == max_steps
        if should_sample:
            samples = sample_with_ema(
                model,
                ema,
                diffusion,
                int(train_config["sample_count"]),
                resolution,
                config,
                device,
            )
            save_image_grid(
                samples,
                paths["outputs_dir"] / f"samples_step_{step:06d}.png",
                nrow=4,
                title=f"EMA DDIM samples at step {step}",
            )

        should_save = step % int(train_config["save_every"]) == 0 or step == max_steps
        if should_save:
            elapsed = time.perf_counter() - start_time
            metadata.update(
                {
                    "completed_steps": step,
                    "elapsed_seconds_this_run": elapsed,
                    "elapsed_hms_this_run": format_seconds(elapsed),
                    "final_loss": loss_history[-1],
                    "mean_loss_last_100": sum(loss_history[-100:])
                    / min(100, len(loss_history)),
                    "effective_epochs": step
                    * int(train_config["batch_size"])
                    / len(dataset),
                }
            )
            checkpoint = make_checkpoint(
                model,
                ema,
                optimizer,
                scaler,
                step,
                loss_history,
                config,
                metadata,
            )
            atomic_torch_save(
                checkpoint, paths["checkpoints_dir"] / f"step_{step:06d}.pt"
            )
            atomic_torch_save(checkpoint, paths["checkpoints_dir"] / "latest.pt")
            plot_training_loss(
                loss_history, paths["outputs_dir"] / "training_loss.png"
            )
            save_json(metadata, paths["outputs_dir"] / "training_metadata.json")
    progress.close()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
