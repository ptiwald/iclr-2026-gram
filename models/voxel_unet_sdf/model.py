"""Rung 3 of the voxel ablation ladder: U-Net + SDF features.

Identical to rung 2 (VoxelUNet) except per-point input gains a 14-dim
signed-distance-to-airfoil feature stack: [SDF, log1p(SDF), Fourier(log1p(SDF), 6 bands)].
SDF is computed on the fly inside forward via chunked cdist over the airfoil
indices (8192-point chunks, matching the top-5 voxel-entry convention).

Delta from rung 2 isolates what an explicit near-wall coordinate buys.
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
    """3-level 3D U-Net (identical to rung 2's _UNet3D)."""

    def __init__(self, c_in: int, hidden: int):
        super().__init__()
        c1, c2, c3 = hidden, hidden * 2, hidden * 4

        self.enc1 = _double_conv(c_in, c1)
        self.pool1 = MaxPool3d(2)
        self.enc2 = _double_conv(c1, c2)
        self.pool2 = MaxPool3d(2)
        self.enc3 = _double_conv(c2, c3)

        self.dec2 = _double_conv(c3 + c2, c2)
        self.dec1 = _double_conv(c2 + c1, c1)

        self.out = Conv3d(c1, hidden, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x, size=ref.shape[-3:], mode="trilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.dec2(torch.cat([self._up(e3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.out(d1)


def _sdf_chunked(pos: torch.Tensor, surface: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Min Euclidean distance from each point in `pos` to any point in `surface`.

    pos: (N, 3). surface: (M, 3). Returns (N,). Computes cdist in chunks of
    `chunk` query points to keep peak memory bounded (top-5 voxel entries use
    8192 by convention).
    """
    N = pos.shape[0]
    if surface.shape[0] == 0:
        return torch.full((N,), float("inf"), device=pos.device, dtype=pos.dtype)
    out = torch.empty(N, device=pos.device, dtype=pos.dtype)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        d = torch.cdist(pos[start:end], surface)  # (chunk, M)
        out[start:end] = d.min(dim=1).values
    return out


class VoxelUNetSDF(Module):

    POS_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    SDF_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    GRID_DIMS = (64, 32, 32)

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden

        self.register_buffer(
            "pos_freqs",
            2.0 * math.pi * torch.tensor(self.POS_FREQS, dtype=torch.float32),
        )
        self.register_buffer(
            "sdf_freqs",
            2.0 * math.pi * torch.tensor(self.SDF_FREQS, dtype=torch.float32),
        )
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))

        # Per-point input features:
        #   pos (3) + Fourier(pos, 10) (60) + velocity_in (15) + airfoil mask (1)
        # + SDF stack (14):
        #   sdf (1) + log1p(sdf) (1) + Fourier(log1p(sdf), 6) (12)
        sdf_dim = 1 + 1 + 2 * len(self.SDF_FREQS)
        in_dim = 3 + 3 * 2 * len(self.POS_FREQS) + 15 + 1 + sdf_dim

        self.point_encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

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
                f"[VoxelUNetSDF] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print("[VoxelUNetSDF] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: (..., D), freqs: (F,) -> (..., D*2*F)
        angles = x.unsqueeze(-1) * freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=-3) if x.dim() >= 2 else feats.flatten()

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

    @torch.no_grad()
    def _compute_sdf_features(
        self, pos_norm: torch.Tensor, idcs_airfoil: list[torch.Tensor]
    ) -> torch.Tensor:
        """Per-point SDF stack: [sdf, log1p(sdf), Fourier(log1p(sdf), 6 bands)].

        Returns (B, N, sdf_dim). Computed on normalized pos so distances live
        in the same scale as the rest of the input. No gradient needed: SDF is
        a function of geometry, not learned.
        """
        B, N, _ = pos_norm.shape
        sdfs = []
        for b in range(B):
            idcs = idcs_airfoil[b]
            surface = pos_norm[b, idcs]  # (M, 3)
            sdfs.append(_sdf_chunked(pos_norm[b], surface))
        sdf = torch.stack(sdfs, dim=0)  # (B, N)

        log1p_sdf = torch.log1p(sdf)
        sdf_fourier = self._fourier(log1p_sdf.unsqueeze(-1), self.sdf_freqs)  # (B, N, 1*2*6)
        return torch.cat([sdf.unsqueeze(-1), log1p_sdf.unsqueeze(-1), sdf_fourier], dim=-1)

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
        pos_fourier = self._fourier(pos_norm, self.pos_freqs)
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i, idcs in enumerate(idcs_airfoil):
            airfoil_mask[i, idcs, 0] = 1.0
        sdf_stack = self._compute_sdf_features(pos_norm, idcs_airfoil)  # (B, N, 14)

        x = torch.cat([pos_norm, pos_fourier, vel_flat, airfoil_mask, sdf_stack], dim=2)
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
