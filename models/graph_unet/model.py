"""Graph-U-Net: a lightweight, configurable graph-transformer trunk with an
optional graph-native multiscale hierarchy (pool / unpool + skip connections).

This is a from-scratch, narrow-and-deep design tuned for *tunability by design*:
the graph-transformer block is **pre-LN** (`x = x + sublayer(LN(x))`), so it trains
stably at depth without LR warmup or any change to the training recipe — unlike a
post-LN stack, which needs warmup to avoid an early loss stall. Depth is the only
reach mechanism in a flat k-NN GNN, so the baseline is set deep enough to cover the
reach the receptive-field analysis showed is sufficient (~8% of the x-span ≈ 0.13
chord, reached at k=16, L≈8 — see analyze/receptive_field_physics/report.md).

One class, three regimes for clean ablation:
  - `num_levels=1`                        -> flat baseline (reach via depth only)
  - `num_levels>1, pool_ratio=1.0`        -> flat + skips (multi-depth features,
                                             SAME reach as flat — no coarsening)
  - `num_levels>1, pool_ratio<1.0`        -> Graph-U-Net (pooling buys reach cheaply,
                                             skips preserve fine structure)

So if the hierarchy moves the holdout floor, the A->B->C ladder says *immediately*
whether it was the skips (B already moves) or the pooling (only C moves).

Features match ResMLPMin (pos-Fourier + velocity history + velocity-Fourier +
unsigned distance-to-airfoil + airfoil mask) so the comparison isolates
architecture, not the input encoding. Prediction is residual-delta off the last
input frame with hard no-slip zeroing on the airfoil surface.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_cluster import knn


T_IN = 5
T_OUT = 5


# ---------------------------------------------------------------------------
# Feature helpers (matched to ResMLPMin)
# ---------------------------------------------------------------------------

def _fourier(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1).reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


def _sdf_single(pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Unsigned distance from every point to its nearest airfoil-surface point."""
    if idx.numel() == 0:
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    airfoil_pts = pos.index_select(0, idx)
    chunks = [torch.cdist(c, airfoil_pts).min(dim=1).values for c in pos.split(8192)]
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Graph utilities (kNN graph, boundary-aware pooling, IDW unpool)
# ---------------------------------------------------------------------------

def _knn_graph(pos: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense kNN graph on a single point cloud.

    Returns (neighbors [N, k], rel_pos [N, k, 3], dist [N, k]). Includes self as a
    neighbor (distance 0), matching the PTPointNetA convention.
    """
    edge = knn(pos, pos, k)                       # [2, N*k]: row0 = query, row1 = neighbor
    neighbors = edge[1].view(pos.shape[0], k)
    rel_pos = pos[neighbors] - pos.unsqueeze(1)
    dist = rel_pos.norm(dim=-1)
    return neighbors, rel_pos, dist


def _voxel_grid_representatives(pos: torch.Tensor, num_samples: int) -> torch.Tensor:
    """One representative per occupied voxel; voxel size targets ~num_samples cells."""
    n = pos.size(0)
    if num_samples >= n:
        return torch.arange(n, device=pos.device)

    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    extent = (maxs - mins).clamp(min=1e-6)
    cell_size = (extent.prod() / max(num_samples, 1)) ** (1.0 / 3.0)
    cell_size = cell_size.clamp(min=1e-6)

    coords = torch.floor((pos - mins) / cell_size).to(torch.long)
    grid_shape = coords.max(dim=0).values + 1
    s1 = grid_shape[1] + 1
    s2 = grid_shape[2] + 1
    keys = coords[:, 0] * s1 * s2 + coords[:, 1] * s2 + coords[:, 2]

    order = torch.argsort(keys)
    keys_sorted = keys[order]
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    reps = order[first_mask]

    if reps.numel() > num_samples:
        pick = torch.linspace(0, reps.numel() - 1, steps=num_samples, device=pos.device)
        reps = reps[pick.round().to(torch.long)]
    return reps


def _boundary_aware_pool(pos: torch.Tensor, num_samples: int, priority_idx: torch.Tensor) -> torch.Tensor:
    """Pick `num_samples` representatives, keeping airfoil/priority points first."""
    n = pos.size(0)
    if num_samples >= n:
        return torch.arange(n, device=pos.device)

    if priority_idx.numel() == 0:
        return _voxel_grid_representatives(pos, num_samples)

    priority_idx = torch.unique(priority_idx, sorted=False)
    if priority_idx.numel() >= num_samples:
        local = _voxel_grid_representatives(pos[priority_idx], num_samples)
        return priority_idx[local]

    mask = torch.ones(n, dtype=torch.bool, device=pos.device)
    mask[priority_idx] = False
    remaining = torch.nonzero(mask, as_tuple=False).flatten()
    need = num_samples - priority_idx.numel()
    local = _voxel_grid_representatives(pos[remaining], need)
    return torch.cat([priority_idx, remaining[local]], dim=0)


def _knn_interpolate(feat_sub: torch.Tensor, pos_sub: torch.Tensor, pos_full: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Inverse-distance-weighted unpool from coarse (sub) to fine (full) points."""
    k = min(k, pos_sub.shape[0])
    edge = knn(pos_sub, pos_full, k)              # row0 = full query, row1 = sub neighbor
    nbr = edge[1].view(pos_full.shape[0], k)
    diff = pos_full.unsqueeze(1) - pos_sub[nbr]
    dist = diff.norm(dim=-1).clamp(min=1e-8)
    w = 1.0 / dist
    w = w / w.sum(dim=1, keepdim=True)
    return (w.unsqueeze(-1) * feat_sub[nbr]).sum(dim=1)


# ---------------------------------------------------------------------------
# Backbone layers (pre-LN graph attention)
# ---------------------------------------------------------------------------

class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EdgeEncoder(nn.Module):
    """Encode raw edge features [rel_pos(3) + dist(1)] into hidden_dim."""

    def __init__(self, hidden_dim: int, edge_in: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rel_pos: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rel_pos, dist.unsqueeze(-1)], dim=-1))


class _GraphTransformerLayer(nn.Module):
    """Local attention with edge-conditioned value bias, **pre-LN**.

    Pre-LN keeps the residual stream a clean identity path and feeds each sublayer a
    normalized copy (`x = x + sublayer(LN(x))`). This trains stably at depth without
    LR warmup — the architectural fix for the post-LN stall, not a recipe change.
    """

    def __init__(self, dim: int, heads: int, edge_dim: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_e = nn.Linear(edge_dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _attn(self, h: torch.Tensor, neighbors: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        N, k = neighbors.shape
        H, d = self.heads, self.head_dim

        q = self.W_q(h).view(N, 1, H, d)
        k_nbr = self.W_k(h)[neighbors].view(N, k, H, d)
        v_nbr = self.W_v(h)[neighbors].view(N, k, H, d)
        v_nbr = v_nbr + self.W_e(edge_feat).view(N, k, H, d)

        attn = (q * k_nbr).sum(-1) / self.scale          # (N, k, H)
        attn = F.softmax(attn, dim=1)
        out = (attn.unsqueeze(-1) * v_nbr).sum(1).reshape(N, -1)
        return self.W_o(out)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x), neighbors, edge_feat)
        x = x + self.ff(self.norm2(x))
        return x


class _GATLayer(nn.Module):
    """GAT-style: edge bias added to attention scores, **pre-LN** (alternative backbone)."""

    def __init__(self, dim: int, heads: int, edge_dim: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, heads)

        self.norm1 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _attn(self, h: torch.Tensor, neighbors: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        N, k = neighbors.shape
        H, d = self.heads, self.head_dim

        q = self.W_q(h).view(N, 1, H, d)
        k_nbr = self.W_k(h)[neighbors].view(N, k, H, d)
        v_nbr = self.W_v(h)[neighbors].view(N, k, H, d)

        attn = (q * k_nbr).sum(-1) / self.scale + self.edge_proj(edge_feat)
        attn = F.softmax(attn, dim=1)
        out = (attn.unsqueeze(-1) * v_nbr).sum(1).reshape(N, -1)
        return self.W_o(out)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x), neighbors, edge_feat)
        x = x + self.ff(self.norm2(x))
        return x


_LAYERS = {"graph_transformer": _GraphTransformerLayer, "gat": _GATLayer}


class _Level(nn.Module):
    """One hierarchy level: a shared edge encoder + a stack of graph layers."""

    def __init__(self, hidden: int, heads: int, blocks: int, backbone: str, use_checkpoint: bool = False):
        super().__init__()
        layer_cls = _LAYERS[backbone]
        self.edge_enc = _EdgeEncoder(hidden)
        self.layers = nn.ModuleList([layer_cls(hidden, heads, edge_dim=hidden) for _ in range(blocks)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, rel_pos: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        edge_feat = self.edge_enc(rel_pos, dist)
        for layer in self.layers:
            # Checkpoint per layer (not per level): each layer's (N, k, D) gathers
            # are recomputed alone in backward, so peak memory is one layer's worth.
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, neighbors, edge_feat, use_reentrant=False)
            else:
                x = layer(x, neighbors, edge_feat)
        return x


class _TemporalAttentionHead(nn.Module):
    """Per-node self-attention across the T_out output tokens, **pre-LN**."""

    def __init__(self, hidden: int, t_out: int, num_heads: int, num_attn_layers: int):
        super().__init__()
        self.t_out = t_out
        self.proj = nn.Linear(hidden, t_out * hidden)
        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.attn_layers.append(nn.MultiheadAttention(hidden, num_heads, batch_first=True))
            self.ff_layers.append(nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden),
            ))
            self.norm1.append(nn.LayerNorm(hidden))
            self.norm2.append(nn.LayerNorm(hidden))
        self.time_pe = nn.Parameter(torch.randn(1, t_out, hidden) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.proj(x).view(N, self.t_out, -1) + self.time_pe
        for attn, ff, n1, n2 in zip(self.attn_layers, self.ff_layers, self.norm1, self.norm2):
            hn = n1(h)
            h = h + attn(hn, hn, hn)[0]
            h = h + ff(n2(h))
        return h                                          # (N, T_out, hidden)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GraphUNet(nn.Module):

    def __init__(
        self,
        backbone: str = "graph_transformer",
        hidden: int = 128,
        heads: int = 4,
        k: int = 16,
        num_levels: int = 1,
        blocks_per_level: int = 8,
        pool_ratio: float = 0.25,
        use_skips: bool = True,
        temporal_attn_layers: int = 2,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 3,
        num_dist_freqs: int = 6,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if backbone not in _LAYERS:
            raise ValueError(f"backbone must be one of {list(_LAYERS)}, got {backbone}")

        self.hidden = hidden
        self.k = k
        self.num_levels = num_levels
        self.pool_ratio = pool_ratio
        self.use_skips = use_skips
        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs
        self.use_checkpoint = use_checkpoint

        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vel_hist_dim = T_IN * 3
        vel_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vel_hist_dim + vel_fourier_dim + dist_dim + 1

        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

        self.enc_levels = nn.ModuleList(
            [_Level(hidden, heads, blocks_per_level, backbone, use_checkpoint) for _ in range(num_levels)]
        )
        self.dec_levels = nn.ModuleList(
            [_Level(hidden, heads, blocks_per_level, backbone, use_checkpoint) for _ in range(num_levels - 1)]
        )
        fuse_in = (2 if use_skips else 1) * hidden
        self.fuse = nn.ModuleList(
            [nn.Linear(fuse_in, hidden) for _ in range(num_levels - 1)]
        )

        # Pre-LN leaves the residual stream unnormalized; a final norm before the
        # head is standard (and necessary) for stable pre-LN transformers.
        self.out_norm = nn.LayerNorm(hidden)

        self.temporal = _TemporalAttentionHead(hidden, T_OUT, heads, temporal_attn_layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )

        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key in ("vel_mean", "vel_std"):
                if key in stats:
                    getattr(self, key).copy_(stats[key])
            print(f"[GraphUNet] loaded norm_stats.pt: vel_mean={self.vel_mean.tolist()}, "
                  f"vel_std={self.vel_std.tolist()}")
        else:
            print("[GraphUNet] norm_stats.pt not found — using identity normalization")

        n_params = sum(p.numel() for p in self.parameters())
        mode = ("flat" if num_levels == 1
                else f"flat+skips" if pool_ratio >= 1.0
                else "graph-unet")
        print(f"[GraphUNet] mode={mode} backbone={backbone} hidden={hidden} heads={heads} "
              f"num_levels={num_levels} blocks_per_level={blocks_per_level} k={k} "
              f"pool_ratio={pool_ratio} use_skips={use_skips} | {n_params / 1e6:.2f}M params")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    # ------------------------------------------------------------------

    def _features(self, pos: torch.Tensor, vel_in: torch.Tensor, airfoil_idx: torch.Tensor) -> torch.Tensor:
        vel_norm = (vel_in - self.vel_mean) / self.vel_std          # (T, N, 3)
        last_norm = vel_norm[-1]                                     # (N, 3)

        pos_feat = _fourier(pos, self.num_pos_freqs)
        vel_flat = vel_norm.permute(1, 0, 2).reshape(pos.shape[0], -1)

        vel_freqs = 2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        vf = last_norm.unsqueeze(-1) * vel_freqs
        vel_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(pos.shape[0], -1)

        dist = _sdf_single(pos, airfoil_idx).unsqueeze(-1)
        dist_log = torch.log1p(dist)
        dist_freqs = 2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        df = dist_log * dist_freqs
        dist_feat = torch.cat([dist, dist_log, df.sin(), df.cos()], dim=-1)

        mask = torch.zeros(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
        mask[airfoil_idx] = 1.0

        return torch.cat([pos_feat, vel_flat, vel_fourier, dist_feat, mask], dim=-1)

    def _forward_single(self, pos, vel_in, airfoil_idx):
        num_points = pos.shape[0]
        feat = self._features(pos, vel_in, airfoil_idx)
        x = self.node_enc(feat)                                     # (N, H)

        air_mask = torch.zeros(num_points, dtype=torch.bool, device=pos.device)
        air_mask[airfoil_idx] = True

        # ---- encoder (coarsen; pool_ratio>=1.0 keeps full resolution) ----
        p, air = pos, air_mask
        skips_h: list[torch.Tensor] = []
        skips_pos: list[torch.Tensor] = []
        for i in range(self.num_levels):
            if i > 0 and self.pool_ratio < 1.0:
                target = max(int(round(p.shape[0] * self.pool_ratio)), self.k + 1)
                sel = _boundary_aware_pool(p, target, air.nonzero(as_tuple=False).flatten())
                p, x, air = p[sel], x[sel], air[sel]
            neighbors, rel_pos, dist = _knn_graph(p, self.k)
            x = self.enc_levels[i](x, neighbors, rel_pos, dist)
            if i < self.num_levels - 1:
                skips_h.append(x)
                skips_pos.append(p)

        # ---- decoder (refine, optional skip concat) ----
        for i in range(self.num_levels - 2, -1, -1):
            x_up = _knn_interpolate(x, p, skips_pos[i])
            if self.use_skips:
                x = self.fuse[i](torch.cat([x_up, skips_h[i]], dim=-1))
            else:
                x = self.fuse[i](x_up)
            p = skips_pos[i]
            neighbors, rel_pos, dist = _knn_graph(p, self.k)
            x = self.dec_levels[i](x, neighbors, rel_pos, dist)

        # ---- temporal head + residual-delta + no-slip ----
        x = self.out_norm(x)
        xt = self.temporal(x)                                       # (N, T_out, H)
        delta = self.decoder(xt) * self.vel_std                     # (N, T_out, 3)
        pred = vel_in[-1].unsqueeze(1) + delta                      # (N, T_out, 3)
        pred[airfoil_idx] = 0.0
        return pred.permute(1, 0, 2)                                # (T_out, N, 3)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        outputs = []
        for b in range(velocity_in.shape[0]):
            idx = idcs_airfoil[b].to(pos.device).long().clamp_(0, pos.shape[1] - 1)
            outputs.append(self._forward_single(pos[b], velocity_in[b], idx))
        return torch.stack(outputs)                                 # (B, T_out, N, 3)
