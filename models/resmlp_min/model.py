"""Rung B: wide residual-MLP trunk + voxel/U-Net residual sidecar.

Single-member ResMLP (#5) design with no TTA and no ensemble. Sits between
pre- and post-blocks: `hidden = hidden + unet(voxelize(hidden), pos01)`.
Tests the U-Net-as-sidecar hypothesis against rung A (no U-Net).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


T_IN = 5
T_OUT = 5
DOMAIN_MIN = (0.0, -0.41, 0.0)
DOMAIN_MAX = (2.10, 0.41, 1.22)


def _fourier(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1).reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


def _sdf(pos: torch.Tensor, idcs_airfoil: list[torch.Tensor]) -> torch.Tensor:
    batch_size, num_points, _ = pos.shape
    out = torch.empty(batch_size, num_points, device=pos.device, dtype=pos.dtype)
    for b in range(batch_size):
        idx = idcs_airfoil[b].to(pos.device).long()
        airfoil_pts = pos[b].index_select(0, idx)
        chunks = [torch.cdist(c, airfoil_pts).min(dim=1).values for c in pos[b].split(8192)]
        out[b] = torch.cat(chunks, dim=0)
    return out


class _ResMLPBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VoxelUNetSidecar(nn.Module):
    """Voxelize (scatter-mean) → N-level 3D U-Net → devoxelize (grid_sample).

    Width-preserving: input and output channels both equal `hidden`. Level `i`
    (0 = finest) has `ch_base * 2**i` channels; the bottleneck is level
    `num_levels - 1`.
    """

    def __init__(self, hidden: int, grid: tuple[int, int, int], ch_base: int, num_levels: int = 3):
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self.grid = tuple(int(v) for v in grid)
        self.num_levels = int(num_levels)

        encoders = []
        for i in range(self.num_levels):
            c_in = hidden if i == 0 else ch_base * (2 ** (i - 1))
            c_out = ch_base * (2 ** i)
            encoders.append(_DoubleConv(c_in, c_out))
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        for i in range(self.num_levels - 1):
            c_skip = ch_base * (2 ** i)
            c_up = ch_base * (2 ** (i + 1))
            decoders.append(_DoubleConv(c_up + c_skip, c_skip))
        self.decoders = nn.ModuleList(decoders)

        self.out = nn.Conv3d(ch_base, hidden, 1)
        self.pool = nn.MaxPool3d(2)

    def _voxelize(self, x: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, channels = x.shape
        grid_x, grid_y, grid_z = self.grid
        idx = pos01.clamp(0, 1 - 1e-6)
        idx = torch.stack(
            [
                (idx[..., 0] * grid_x).floor().long(),
                (idx[..., 1] * grid_y).floor().long(),
                (idx[..., 2] * grid_z).floor().long(),
            ],
            dim=-1,
        )
        flat = idx[..., 0] * (grid_y * grid_z) + idx[..., 1] * grid_z + idx[..., 2]
        voxel = torch.zeros(batch_size, channels, grid_x * grid_y * grid_z, device=x.device, dtype=x.dtype)
        count = torch.zeros(batch_size, 1, grid_x * grid_y * grid_z, device=x.device, dtype=x.dtype)
        voxel.scatter_add_(2, flat.unsqueeze(1).expand(-1, channels, -1), x.transpose(1, 2))
        ones = torch.ones(batch_size, 1, num_points, device=x.device, dtype=x.dtype)
        count.scatter_add_(2, flat.unsqueeze(1), ones)
        voxel = voxel / count.clamp(min=1.0)
        return voxel.view(batch_size, channels, grid_x, grid_y, grid_z)

    def _devoxelize(self, voxel: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        batch_size, channels = voxel.shape[:2]
        num_points = pos01.shape[1]
        pos_grid = pos01 * 2.0 - 1.0
        sample_grid = pos_grid[..., [2, 1, 0]].view(batch_size, 1, 1, num_points, 3)
        out = F.grid_sample(voxel, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        v0 = self._voxelize(x, pos01)
        skips = [self.encoders[0](v0)]
        for i in range(1, self.num_levels):
            skips.append(self.encoders[i](self.pool(skips[-1])))

        d = skips[-1]
        for i in range(self.num_levels - 2, -1, -1):
            u = F.interpolate(d, size=skips[i].shape[-3:], mode="trilinear", align_corners=False)
            d = self.decoders[i](torch.cat([u, skips[i]], dim=1))
        return self._devoxelize(self.out(d), pos01)


class ResMLPMin(nn.Module):

    def __init__(
        self,
        hidden: int = 256,
        num_pre: int = 2,
        num_post: int = 4,
        grid: tuple[int, int, int] = (64, 32, 32),
        ch_base: int = 64,
        num_levels: int = 3,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 3,
        num_dist_freqs: int = 6,
    ):
        super().__init__()
        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vel_hist_dim = T_IN * 3
        vel_last_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        airfoil_mask_dim = 1
        in_dim = pos_dim + vel_hist_dim + vel_last_fourier_dim + dist_dim + airfoil_mask_dim

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([_ResMLPBlock(hidden) for _ in range(num_pre)])
        self.unet = _VoxelUNetSidecar(hidden=hidden, grid=grid, ch_base=ch_base, num_levels=num_levels)
        self.blocks_post = nn.ModuleList([_ResMLPBlock(hidden) for _ in range(num_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, T_OUT * 3)

        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))
        self.register_buffer("domain_min", torch.tensor(DOMAIN_MIN).view(1, 1, 3))
        self.register_buffer("domain_max", torch.tensor(DOMAIN_MAX).view(1, 1, 3))

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            known = {"vel_mean", "vel_std"}
            for k, v in stats.items():
                if k in known:
                    getattr(self, k).copy_(v)
            print(
                f"[ResMLPMin] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print("[ResMLPMin] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        batch_size, num_steps, num_points, _ = velocity_in.shape

        dist_airfoil = _sdf(pos, idcs_airfoil)

        last_velocity = velocity_in[:, -1]
        vel_norm = (velocity_in - self.vel_mean) / self.vel_std
        last_norm = vel_norm[:, -1]

        pos_feat = _fourier(pos, self.num_pos_freqs)
        vel_flat = vel_norm.permute(0, 2, 1, 3).reshape(batch_size, num_points, num_steps * 3)

        vel_freqs = 2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        vel_fourier = last_norm.unsqueeze(-1) * vel_freqs
        vel_fourier = torch.cat([vel_fourier.sin(), vel_fourier.cos()], dim=-1).reshape(batch_size, num_points, -1)

        airfoil_mask = torch.zeros(batch_size, num_points, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_mask[b, idx.to(pos.device).long(), 0] = 1.0

        dist = dist_airfoil.unsqueeze(-1)
        dist_log = torch.log1p(dist)
        dist_freqs = 2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        dist_fourier = dist_log * dist_freqs
        dist_feat = torch.cat([dist, dist_log, dist_fourier.sin(), dist_fourier.cos()], dim=-1)

        feat = torch.cat([pos_feat, vel_flat, vel_fourier, dist_feat, airfoil_mask], dim=-1)

        hidden = self.proj_in(feat)
        for block in self.blocks_pre:
            hidden = block(hidden)

        pos01 = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        hidden = hidden + self.unet(hidden, pos01)

        for block in self.blocks_post:
            hidden = block(hidden)
        hidden = self.norm_out(hidden)

        delta_norm = self.proj_out(hidden).reshape(batch_size, num_points, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_velocity.unsqueeze(1) + delta

        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0
        return pred
