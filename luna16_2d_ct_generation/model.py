from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dimension, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x).reshape(batch, channels, height * width)
        query, key, value = self.qkv(hidden).chunk(3, dim=1)
        scale = channels**-0.5
        attention = torch.einsum("bcn,bcm->bnm", query * scale, key)
        attention = attention.softmax(dim=-1)
        hidden = torch.einsum("bnm,bcm->bcn", attention, value)
        hidden = self.projection(hidden).reshape(batch, channels, height, width)
        return x + hidden


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SmallUNet(nn.Module):
    """Compact single-channel U-Net for 64x64 or 128x128 noise prediction."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_embedding_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        base = base_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        self.down1 = ResidualBlock(base, base, time_embedding_dim, dropout)
        self.downsample1 = Downsample(base)
        self.down2 = ResidualBlock(base, base * 2, time_embedding_dim, dropout)
        self.downsample2 = Downsample(base * 2)
        self.down3 = ResidualBlock(base * 2, base * 4, time_embedding_dim, dropout)
        self.downsample3 = Downsample(base * 4)

        self.middle1 = ResidualBlock(base * 4, base * 4, time_embedding_dim, dropout)
        self.middle_attention = AttentionBlock(base * 4)
        self.middle2 = ResidualBlock(base * 4, base * 4, time_embedding_dim, dropout)

        self.upsample3 = Upsample(base * 4)
        self.up3 = ResidualBlock(base * 8, base * 2, time_embedding_dim, dropout)
        self.upsample2 = Upsample(base * 2)
        self.up2 = ResidualBlock(base * 4, base, time_embedding_dim, dropout)
        self.upsample1 = Upsample(base)
        self.up1 = ResidualBlock(base * 2, base, time_embedding_dim, dropout)

        self.output_norm = nn.GroupNorm(_group_count(base), base)
        self.output_conv = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        x = self.input_conv(x)

        skip1 = self.down1(x, time_embedding)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, time_embedding)
        x = self.downsample2(skip2)
        skip3 = self.down3(x, time_embedding)
        x = self.downsample3(skip3)

        x = self.middle1(x, time_embedding)
        x = self.middle_attention(x)
        x = self.middle2(x, time_embedding)

        x = self.upsample3(x)
        x = self.up3(torch.cat((x, skip3), dim=1), time_embedding)
        x = self.upsample2(x)
        x = self.up2(torch.cat((x, skip2), dim=1), time_embedding)
        x = self.upsample1(x)
        x = self.up1(torch.cat((x, skip1), dim=1), time_embedding)
        return self.output_conv(F.silu(self.output_norm(x)))


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (len(tuple(shape)) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_previous = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = (
            betas * (1.0 - alpha_cumprod_previous) / (1.0 - alpha_cumprod)
        )
        posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_cumprod_previous) / (1.0 - alpha_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_cumprod_previous)
            * torch.sqrt(alphas)
            / (1.0 - alpha_cumprod)
        )

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_images) if noise is None else noise
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, clean_images.shape) * clean_images
            + extract(
                self.sqrt_one_minus_alpha_cumprod, timesteps, clean_images.shape
            )
            * noise
        )

    def predict_clean(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = extract(self.alpha_cumprod, timesteps, noisy_images.shape)
        return (
            noisy_images - torch.sqrt(1.0 - alpha) * predicted_noise
        ) / torch.sqrt(alpha)

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        images = torch.randn(shape, device=device)
        for time_index in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (shape[0],), time_index, device=device, dtype=torch.long
            )
            predicted_noise = model(images, timesteps)
            clean = self.predict_clean(images, timesteps, predicted_noise).clamp(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, timesteps, images.shape) * clean
                + extract(self.posterior_mean_coef2, timesteps, images.shape) * images
            )
            if time_index > 0:
                variance = extract(self.posterior_variance, timesteps, images.shape)
                images = mean + torch.sqrt(variance) * torch.randn_like(images)
            else:
                images = mean
        return images.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        sampling_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps must be between 1 and train timesteps")
        images = torch.randn(shape, device=device)
        schedule = (
            torch.linspace(self.timesteps - 1, 0, sampling_steps, device=device)
            .round()
            .long()
        )
        for index, time_index in enumerate(schedule):
            timesteps = torch.full(
                (shape[0],), int(time_index.item()), device=device, dtype=torch.long
            )
            predicted_noise = model(images, timesteps)
            alpha = self.alpha_cumprod[time_index]
            if index + 1 < len(schedule):
                alpha_previous = self.alpha_cumprod[schedule[index + 1]]
            else:
                alpha_previous = torch.tensor(1.0, device=device)
            clean = (
                images - torch.sqrt(1.0 - alpha) * predicted_noise
            ) / torch.sqrt(alpha)
            clean = clean.clamp(-1.0, 1.0)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha))
                * (1.0 - alpha / alpha_previous)
            ).clamp(min=0.0)
            direction = torch.sqrt(
                (1.0 - alpha_previous - sigma**2).clamp(min=0.0)
            ) * predicted_noise
            noise = torch.randn_like(images) if float(sigma) > 0 else 0.0
            images = torch.sqrt(alpha_previous) * clean + direction + sigma * noise
        return images.clamp(-1.0, 1.0)


def build_model(config: dict) -> SmallUNet:
    return SmallUNet(
        in_channels=int(config["in_channels"]),
        base_channels=int(config["base_channels"]),
        time_embedding_dim=int(config["time_embedding_dim"]),
        dropout=float(config["dropout"]),
    )


def build_diffusion(config: dict) -> GaussianDiffusion:
    return GaussianDiffusion(
        timesteps=int(config["train_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )
