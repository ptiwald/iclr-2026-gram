"""Rung A: wide residual-MLP trunk, no voxelization, no U-Net.

Mirrors the ResMLP (#5) single-member design with the voxelize+U-Net sidecar
removed. Tests whether the wide point-wise residual-MLP trunk alone — with
SDF features and velocity normalization — gets near the top-5 tier, or
whether the 3D U-Net sidecar is doing the real work.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn


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


class ResMLPNoUNet(nn.Module):

    def __init__(
        self,
        hidden: int = 256,
        num_pre: int = 2,
        num_post: int = 4,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 3,
        num_dist_freqs: int = 6,
    ):
        super().__init__()
        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        pos_dim = 3 * (1 + 2 * num_pos_freqs)               # 63
        vel_hist_dim = T_IN * 3                              # 15
        vel_last_fourier_dim = 3 * 2 * num_vel_freqs         # 18
        dist_dim = 2 + 2 * num_dist_freqs                    # 14
        airfoil_mask_dim = 1
        in_dim = pos_dim + vel_hist_dim + vel_last_fourier_dim + dist_dim + airfoil_mask_dim

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([_ResMLPBlock(hidden) for _ in range(num_pre)])
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
                f"[ResMLPNoUNet] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print("[ResMLPNoUNet] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _per_point_features(
        self,
        pos: torch.Tensor,
        velocity_in: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        dist_airfoil: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_steps, num_points, _ = velocity_in.shape

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
        return feat, velocity_in[:, -1], vel_norm

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        batch_size = pos.shape[0]
        num_points = pos.shape[1]

        dist_airfoil = _sdf(pos, idcs_airfoil)
        feat, last_velocity, _ = self._per_point_features(pos, velocity_in, idcs_airfoil, dist_airfoil)

        hidden = self.proj_in(feat)
        for block in self.blocks_pre:
            hidden = block(hidden)
        for block in self.blocks_post:
            hidden = block(hidden)
        hidden = self.norm_out(hidden)

        delta_norm = self.proj_out(hidden).reshape(batch_size, num_points, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_velocity.unsqueeze(1) + delta

        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0
        return pred
