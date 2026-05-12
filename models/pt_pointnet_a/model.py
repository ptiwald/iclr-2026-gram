"""PTPointNet + voxel U-Net sidecar (injection position A).

Position A: the residual sidecar `h = h + unet(h, pos01)` is inserted
*between* the attention blocks (after `num_blocks // 2` blocks have run).
With the default `num_blocks=2` this puts the sidecar between block 0 and
block 1, so the second attention block's local kNN attention can mix the
spatially-resolved global signal produced by the U-Net.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Module, ModuleList, ReLU, Sequential
from torch.utils.checkpoint import checkpoint
from torch_cluster import knn


DOMAIN_MIN = (0.0, -0.41, 0.0)
DOMAIN_MAX = (2.10, 0.41, 1.22)


class PTPointNetA(Module):
    FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

    def __init__(
        self,
        num_blocks: int = 2,
        k: int = 16,
        hidden: int = 128,
        num_heads: int = 4,
        ffn_mult: int = 2,
        grid: tuple[int, int, int] = (64, 32, 32),
        ch_base: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert hidden % num_heads == 0
        assert num_blocks >= 2, "Position-A sidecar needs at least one block before and after."

        self.num_blocks = num_blocks
        self.k = k
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.ffn_mult = ffn_mult
        self.use_checkpoint = use_checkpoint
        self.sidecar_after = num_blocks // 2 - 1  # insert after this block index

        self.register_buffer(
            "freqs",
            2.0 * math.pi * torch.tensor(self.FREQS, dtype=torch.float32),
        )
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))
        self.register_buffer("domain_min", torch.tensor(DOMAIN_MIN).view(1, 1, 3))
        self.register_buffer("domain_max", torch.tensor(DOMAIN_MAX).view(1, 1, 3))

        in_dim = 3 + 3 * 2 * len(self.FREQS) + 15 + 1

        self.encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        self.blocks = ModuleList(
            [PTBlock(hidden, num_heads, ffn_mult=ffn_mult) for _ in range(num_blocks)]
        )

        self.sidecar = _VoxelUNetSidecar(hidden=hidden, grid=grid, ch_base=ch_base)

        self.decoder = Sequential(
            Linear(3 * hidden, 2 * hidden),
            LayerNorm(2 * hidden),
            ReLU(),
            Linear(2 * hidden, 15),
        )

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            known = {"pos_mean", "pos_scale", "vel_mean", "vel_std"}
            for key, val in stats.items():
                if key in known:
                    getattr(self, key).copy_(val)
            print(
                f"[PTPointNetA] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print(f"[PTPointNetA] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=2)

    def _knn_indices(self, pos_flat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        edge = knn(pos_flat, pos_flat, self.k, batch_x=batch, batch_y=batch)
        n_total = pos_flat.shape[0]
        return edge[1].view(n_total, self.k)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape

        pos01 = ((pos - self.domain_min) / (self.domain_max - self.domain_min)).clamp(0.0, 1.0)

        pos_norm = (pos - self.pos_mean) / self.pos_scale
        velocity_in = (velocity_in - self.vel_mean) / self.vel_std

        vel_flat = velocity_in.transpose(1, 2).reshape(batch_size, num_pos, num_t_in * 3)
        t_start = t[:, 0:1].unsqueeze(1).expand(-1, num_pos, -1)
        pos_fourier = self._fourier(pos_norm)

        x = torch.cat([pos_norm, pos_fourier, vel_flat, t_start], dim=2)

        point_feat = self.encoder(x)  # (B, N, H)

        feats_flat = point_feat.reshape(batch_size * num_pos, -1)
        pos_flat = pos_norm.reshape(batch_size * num_pos, 3)
        batch_vec = torch.arange(batch_size, device=pos.device).repeat_interleave(num_pos)

        neigh_idx = self._knn_indices(pos_flat, batch_vec)
        rel_pos = pos_flat.unsqueeze(1) - pos_flat[neigh_idx]

        for i, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                feats_flat = checkpoint(
                    block, feats_flat, neigh_idx, rel_pos, use_reentrant=False,
                )
            else:
                feats_flat = block(feats_flat, neigh_idx, rel_pos)

            if i == self.sidecar_after:
                feats_bnh = feats_flat.view(batch_size, num_pos, self.hidden)
                if self.use_checkpoint and self.training:
                    side = checkpoint(self.sidecar, feats_bnh, pos01, use_reentrant=False)
                else:
                    side = self.sidecar(feats_bnh, pos01)
                feats_flat = (feats_bnh + side).reshape(batch_size * num_pos, self.hidden)

        neighborhood_feat = feats_flat.view(batch_size, num_pos, -1)

        global_feat = neighborhood_feat.max(dim=1).values
        global_feat = global_feat.unsqueeze(1).expand(-1, num_pos, -1)
        combined = torch.cat([point_feat, neighborhood_feat, global_feat], dim=2)

        delta = self.decoder(combined).view(batch_size, num_pos, num_t_in, 3)
        last_frame = velocity_in[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta

        out = out * self.vel_std + self.vel_mean

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2)


class PTBlock(Module):
    """Point Transformer block: subtraction attention over kNN + FFN, pre-LN, residual."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.ln_attn = LayerNorm(hidden)
        self.q_proj = Linear(hidden, hidden)
        self.k_proj = Linear(hidden, hidden)
        self.v_proj = Linear(hidden, hidden)

        self.pos_enc = Sequential(
            Linear(3, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

        self.gamma = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, num_heads),
        )

        self.out_proj = Linear(hidden, hidden)

        self.ln_ffn = LayerNorm(hidden)
        self.ffn = Sequential(
            Linear(hidden, ffn_mult * hidden),
            ReLU(),
            Linear(ffn_mult * hidden, hidden),
        )

    def forward(
        self,
        x: torch.Tensor,
        neigh_idx: torch.Tensor,
        rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        h = self.ln_attn(x)

        q = self.q_proj(h)
        k = self.k_proj(h)[neigh_idx]
        v = self.v_proj(h)[neigh_idx]
        delta = self.pos_enc(rel_pos)

        pre = q.unsqueeze(1) - k + delta
        w = self.gamma(pre)
        w = torch.softmax(w, dim=1)

        n_total, k_neigh, _ = v.shape
        v_heads = (v + delta).view(n_total, k_neigh, self.num_heads, self.head_dim)
        w_heads = w.unsqueeze(-1)
        out = (w_heads * v_heads).sum(dim=1)
        out = out.reshape(n_total, self.hidden)
        out = self.out_proj(out)

        x = residual + out
        x = x + self.ffn(self.ln_ffn(x))
        return x


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
    """Voxelize (scatter-mean) → 3-level 3D U-Net → devoxelize (grid_sample).

    Width-preserving: input and output channels both equal `hidden`.
    Identical to the sidecar used in ResMLPMin, kept here so the model
    directory stays self-contained.
    """

    def __init__(self, hidden: int, grid: tuple[int, int, int], ch_base: int):
        super().__init__()
        self.grid = tuple(int(v) for v in grid)
        self.enc1 = _DoubleConv(hidden, ch_base)
        self.enc2 = _DoubleConv(ch_base, ch_base * 2)
        self.enc3 = _DoubleConv(ch_base * 2, ch_base * 4)
        self.dec2 = _DoubleConv(ch_base * 4 + ch_base * 2, ch_base * 2)
        self.dec1 = _DoubleConv(ch_base * 2 + ch_base, ch_base)
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
        e1 = self.enc1(v0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self._devoxelize(self.out(d1), pos01)
