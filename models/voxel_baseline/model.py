"""Rung 1 of the voxel ablation ladder: voxelize + flat 3D convs.

The deliberately-simple baseline: no U-Net, no SDF, no adaptive warp. Mean-pool
splat onto a fixed 64x32x32 grid, run a stack of 3D conv blocks at full grid
resolution, devoxelize via grid_sample, predict a residual delta on the last
input frame. Per-point input is bare: pos + Fourier(pos) + 5-frame velocity
history + binary airfoil mask.

Each subsequent rung adds one isolated change (U-Net hierarchy, then SDF, then
y-mirror) so the delta from this baseline attributes a single ingredient.
"""
import math
import os

import torch
from torch.nn import BatchNorm3d, Conv3d, LayerNorm, Linear, Module, ModuleList, ReLU, Sequential


class VoxelBaseline(Module):

    POS_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    GRID_DIMS = (64, 32, 32)  # (X, Y, Z) — anisotropic, matches box aspect 2.6:1:1.5

    def __init__(
        self,
        hidden: int = 64,
        num_conv_blocks: int = 6,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_conv_blocks = num_conv_blocks

        self.register_buffer(
            "pos_freqs",
            2.0 * math.pi * torch.tensor(self.POS_FREQS, dtype=torch.float32),
        )
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))

        # Per-point input features.
        # pos (3) + fourier(pos) (3*2*F) + velocity_in (5*3) + airfoil mask (1)
        in_dim = 3 + 3 * 2 * len(self.POS_FREQS) + 15 + 1

        self.point_encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        # Add 1 channel for occupancy (whether the voxel had any points splatted
        # into it) so the conv stack can distinguish empty from "averaged to zero".
        conv_in = hidden + 1
        blocks = []
        for i in range(num_conv_blocks):
            ch_in = conv_in if i == 0 else hidden
            blocks.append(Sequential(
                Conv3d(ch_in, hidden, kernel_size=3, padding=1),
                BatchNorm3d(hidden),
                ReLU(),
            ))
        self.conv_blocks = ModuleList(blocks)

        self.point_decoder = Sequential(
            LayerNorm(2 * hidden),
            Linear(2 * hidden, hidden),
            ReLU(),
            Linear(hidden, 5 * 3),
        )

        # Zero-init the output head so training starts from the residual identity
        # (pred = last input frame); top-5 voxel entries do this.
        with torch.no_grad():
            self.point_decoder[-1].weight.zero_()
            self.point_decoder[-1].bias.zero_()

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key, val in stats.items():
                getattr(self, key).copy_(val)
            print(
                f"[VoxelBaseline] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print("[VoxelBaseline] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, N, 3) -> (B, N, 3*2*F)
        angles = pos.unsqueeze(-1) * self.pos_freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=2)

    def _voxelize(self, feats: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        """Mean-pool point features into a (B, C+1, X, Y, Z) voxel grid.

        feats: (B, N, C). pos01: (B, N, 3) in [0, 1]. Returns voxel features
        with an extra occupancy channel appended at the channel dimension.
        """
        B, N, C = feats.shape
        X, Y, Z = self.GRID_DIMS
        device = feats.device

        # Per-axis integer index, clamped just inside the grid.
        idx_xyz = (pos01 * torch.tensor([X, Y, Z], device=device, dtype=pos01.dtype)).long()
        idx_xyz = idx_xyz.clamp(min=torch.zeros(3, device=device, dtype=torch.long),
                                max=torch.tensor([X - 1, Y - 1, Z - 1], device=device, dtype=torch.long))
        ix, iy, iz = idx_xyz.unbind(dim=-1)  # each (B, N)
        flat_idx = (ix * Y + iy) * Z + iz    # (B, N)

        # Add a per-batch offset so we can scatter into a single flat tensor.
        batch_offset = torch.arange(B, device=device).view(B, 1) * (X * Y * Z)
        scatter_idx = (flat_idx + batch_offset).reshape(-1)  # (B*N,)

        # Scatter features (sum) and counts.
        flat_feats = torch.zeros(B * X * Y * Z, C, device=device, dtype=feats.dtype)
        flat_counts = torch.zeros(B * X * Y * Z, 1, device=device, dtype=feats.dtype)
        flat_feats.index_add_(0, scatter_idx, feats.reshape(B * N, C))
        ones = torch.ones(B * N, 1, device=device, dtype=feats.dtype)
        flat_counts.index_add_(0, scatter_idx, ones)

        # Mean-pool: divide by counts (clamped to avoid div-by-zero in empty voxels).
        flat_feats = flat_feats / flat_counts.clamp(min=1.0)

        # Occupancy channel: 1 where the voxel got any point, else 0.
        occupancy = (flat_counts > 0).to(feats.dtype)

        feats_grid = flat_feats.view(B, X, Y, Z, C).permute(0, 4, 1, 2, 3).contiguous()
        occ_grid = occupancy.view(B, X, Y, Z, 1).permute(0, 4, 1, 2, 3).contiguous()
        return torch.cat([feats_grid, occ_grid], dim=1)  # (B, C+1, X, Y, Z)

    def _devoxelize(self, voxel_feats: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample voxel features back to per-point features.

        voxel_feats: (B, C, X, Y, Z). pos01: (B, N, 3) in [0, 1].
        grid_sample expects coordinates in [-1, 1] and the spatial dim order
        z-y-x, so we flip the per-point axes accordingly.
        """
        B, C, _, _, _ = voxel_feats.shape
        N = pos01.shape[1]
        # (B, N, 3) in [0,1] -> (B, N, 3) in [-1,1], then reorder x,y,z -> z,y,x.
        coords = (pos01 * 2.0 - 1.0)
        coords = coords[..., [2, 1, 0]]  # (B, N, 3) z,y,x order
        grid = coords.view(B, N, 1, 1, 3)  # (B, N, 1, 1, 3) for grid_sample
        sampled = torch.nn.functional.grid_sample(
            voxel_feats, grid, mode="bilinear", align_corners=False,
        )  # (B, C, N, 1, 1)
        return sampled.view(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t  # not consumed; temporal info is in the velocity history
        B, NUM_T_IN, N, _ = velocity_in.shape

        # Normalize.
        pos_norm = (pos - self.pos_mean) / self.pos_scale          # ~[-1, 1]
        velocity_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        # Per-point feature stack.
        vel_flat = velocity_in_norm.transpose(1, 2).reshape(B, N, NUM_T_IN * 3)  # (B, N, 15)
        pos_fourier = self._fourier(pos_norm)                                     # (B, N, 60)
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i, idcs in enumerate(idcs_airfoil):
            airfoil_mask[i, idcs, 0] = 1.0
        x = torch.cat([pos_norm, pos_fourier, vel_flat, airfoil_mask], dim=2)
        point_feat = self.point_encoder(x)                                        # (B, N, hidden)

        # Voxelize: map normalized pos in [-1, 1] -> [0, 1] for grid indexing.
        pos01 = (pos_norm + 1.0) * 0.5
        pos01 = pos01.clamp(0.0, 1.0 - 1e-6)
        voxel_feats = self._voxelize(point_feat, pos01)  # (B, hidden+1, X, Y, Z)

        # Flat 3D convs (no down/up sampling, no skip connections).
        h = voxel_feats
        for block in self.conv_blocks:
            h = block(h)
        # h: (B, hidden, X, Y, Z)

        # Devoxelize back to points.
        voxel_point_feat = self._devoxelize(h, pos01)  # (B, N, hidden)

        # Combine the original per-point feature with the voxel-context feature.
        combined = torch.cat([point_feat, voxel_point_feat], dim=2)  # (B, N, 2*hidden)
        delta = self.point_decoder(combined).view(B, N, NUM_T_IN, 3)  # (B, N, 5, 3)

        # Residual on the last input frame, then denormalize.
        last_frame = velocity_in_norm[:, -1, :, :]  # (B, N, 3)
        out = last_frame.unsqueeze(2) + delta       # (B, N, 5, 3)
        out = out * self.vel_std + self.vel_mean

        # Hard zero on airfoil indices for the no-slip BC.
        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2).contiguous()  # (B, 5, N, 3)
