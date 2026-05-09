"""Rung 2 of the voxel ablation ladder: voxelize + 3-level 3D U-Net.

Identical to rung 1 (VoxelBaseline) except the flat conv stack is replaced by
a 3-level U-Net (down/up sampling with skip connections). All other components
— per-point input features, mean-pool voxelization, grid_sample devoxelization,
residual-delta head, training recipe — are unchanged.

The delta from rung 1 isolates what hierarchical multi-scale processing buys.
"""
import math
import os

import torch
from torch.nn import (
    BatchNorm3d, Conv3d, LayerNorm, Linear, MaxPool3d,
    Module, ModuleList, ReLU, Sequential,
)


def _double_conv(c_in: int, c_out: int) -> Sequential:
    return Sequential(
        Conv3d(c_in, c_out, kernel_size=3, padding=1),
        BatchNorm3d(c_out),
        ReLU(),
        Conv3d(c_out, c_out, kernel_size=3, padding=1),
        BatchNorm3d(c_out),
        ReLU(),
    )


class _UNet3D(Module):
    """3-level 3D U-Net.

    Encoder: full -> /2 -> /4 (channel multipliers 1, 2, 4 of `hidden`).
    Decoder: trilinear upsample + skip-cat + double conv at each level.
    Output: 1x1x1 conv back to `hidden` channels.
    """

    def __init__(self, c_in: int, hidden: int):
        super().__init__()
        c1, c2, c3 = hidden, hidden * 2, hidden * 4

        self.enc1 = _double_conv(c_in, c1)
        self.pool1 = MaxPool3d(2)
        self.enc2 = _double_conv(c1, c2)
        self.pool2 = MaxPool3d(2)
        self.enc3 = _double_conv(c2, c3)  # bottleneck

        self.dec2 = _double_conv(c3 + c2, c2)
        self.dec1 = _double_conv(c2 + c1, c1)

        self.out = Conv3d(c1, hidden, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x, size=ref.shape[-3:], mode="trilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)            # full res
        e2 = self.enc2(self.pool1(e1))  # /2
        e3 = self.enc3(self.pool2(e2))  # /4 — bottleneck

        d2 = self.dec2(torch.cat([self._up(e3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.out(d1)


class VoxelUNet(Module):

    POS_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    GRID_DIMS = (64, 32, 32)  # (X, Y, Z) — anisotropic, matches box aspect 2.6:1:1.5

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden

        self.register_buffer(
            "pos_freqs",
            2.0 * math.pi * torch.tensor(self.POS_FREQS, dtype=torch.float32),
        )
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))

        # Per-point input features: pos (3) + fourier(pos) (3*2*F) + velocity_in (15) + airfoil mask (1).
        in_dim = 3 + 3 * 2 * len(self.POS_FREQS) + 15 + 1

        self.point_encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        # Voxel grid input: hidden + 1 occupancy channel.
        self.unet = _UNet3D(c_in=hidden + 1, hidden=hidden)

        self.point_decoder = Sequential(
            LayerNorm(2 * hidden),
            Linear(2 * hidden, hidden),
            ReLU(),
            Linear(hidden, 5 * 3),
        )
        with torch.no_grad():
            self.point_decoder[-1].weight.zero_()
            self.point_decoder[-1].bias.zero_()

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key, val in stats.items():
                getattr(self, key).copy_(val)
            print(
                f"[VoxelUNet] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print("[VoxelUNet] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.pos_freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=2)

    def _voxelize(self, feats: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        B, N, C = feats.shape
        X, Y, Z = self.GRID_DIMS
        device = feats.device

        idx_xyz = (pos01 * torch.tensor([X, Y, Z], device=device, dtype=pos01.dtype)).long()
        idx_xyz = idx_xyz.clamp(min=torch.zeros(3, device=device, dtype=torch.long),
                                max=torch.tensor([X - 1, Y - 1, Z - 1], device=device, dtype=torch.long))
        ix, iy, iz = idx_xyz.unbind(dim=-1)
        flat_idx = (ix * Y + iy) * Z + iz

        batch_offset = torch.arange(B, device=device).view(B, 1) * (X * Y * Z)
        scatter_idx = (flat_idx + batch_offset).reshape(-1)

        flat_feats = torch.zeros(B * X * Y * Z, C, device=device, dtype=feats.dtype)
        flat_counts = torch.zeros(B * X * Y * Z, 1, device=device, dtype=feats.dtype)
        flat_feats.index_add_(0, scatter_idx, feats.reshape(B * N, C))
        ones = torch.ones(B * N, 1, device=device, dtype=feats.dtype)
        flat_counts.index_add_(0, scatter_idx, ones)

        flat_feats = flat_feats / flat_counts.clamp(min=1.0)
        occupancy = (flat_counts > 0).to(feats.dtype)

        feats_grid = flat_feats.view(B, X, Y, Z, C).permute(0, 4, 1, 2, 3).contiguous()
        occ_grid = occupancy.view(B, X, Y, Z, 1).permute(0, 4, 1, 2, 3).contiguous()
        return torch.cat([feats_grid, occ_grid], dim=1)

    def _devoxelize(self, voxel_feats: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        B, C, _, _, _ = voxel_feats.shape
        N = pos01.shape[1]
        coords = (pos01 * 2.0 - 1.0)[..., [2, 1, 0]]
        grid = coords.view(B, N, 1, 1, 3)
        sampled = torch.nn.functional.grid_sample(
            voxel_feats, grid, mode="bilinear", align_corners=False,
        )
        return sampled.view(B, C, N).transpose(1, 2).contiguous()

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        B, NUM_T_IN, N, _ = velocity_in.shape

        pos_norm = (pos - self.pos_mean) / self.pos_scale
        velocity_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        vel_flat = velocity_in_norm.transpose(1, 2).reshape(B, N, NUM_T_IN * 3)
        pos_fourier = self._fourier(pos_norm)
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i, idcs in enumerate(idcs_airfoil):
            airfoil_mask[i, idcs, 0] = 1.0
        x = torch.cat([pos_norm, pos_fourier, vel_flat, airfoil_mask], dim=2)
        point_feat = self.point_encoder(x)

        pos01 = ((pos_norm + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-6)
        voxel_feats = self._voxelize(point_feat, pos01)
        voxel_feats = self.unet(voxel_feats)
        voxel_point_feat = self._devoxelize(voxel_feats, pos01)

        combined = torch.cat([point_feat, voxel_point_feat], dim=2)
        delta = self.point_decoder(combined).view(B, N, NUM_T_IN, 3)

        last_frame = velocity_in_norm[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta
        out = out * self.vel_std + self.vel_mean

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2).contiguous()
