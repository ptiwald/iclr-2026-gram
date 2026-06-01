# Graph-U-Net for the GRaM forecasting task — Design

**Date:** 2026-06-01
**Branch:** `voxel-ladder`
**Status:** Approved design, pending implementation plan

## 1. Motivation & hypothesis

LinkedIn post #2 (`gram-writeup/post/linkedin_2.md`) closes with a promise: take the
6th-place model (EnsembleSpatioTemporalModels, a flat full-resolution kNN
graph-transformer, holdout 0.0589 as a 2-member ensemble) and **add a
Graph-U-Net-style hierarchy with skip connections so fine structure survives to the
head**. The post's claim is that the 6th place keeps all 100k points at native
resolution but stacks 10 message-passing layers, which **smooths the fine structure
away before the head reads each node**. A multiscale hierarchy with skip connections
should preserve it.

**Hypothesis under test:** a *graph-native* multiscale hierarchy (graph pooling /
unpooling + skip connections, **no voxel grid anywhere**) can lift a graph trunk to
ResMLPMin's level — holdout rel-L2 **0.0632** (single-model, 64-grid baseline). This
is distinct from `PTPointNetA`, which already ties ResMLPMin but does so with a
**voxel** sidecar. Here the multiscale structure is built entirely in the graph
domain.

## 2. What the 6th place actually is (reference)

Source: `silentkernel3/iclr-2026`, branch `submission/spatiotemporal-gnn`,
`models/ensemble_spatiotemporal_models/`. Three layers:

- **Member model** `SpatioTemporalGNN` (`gnn_base.py`): per-sample forward, full
  100k points throughout. Pipeline: Fourier(pos,8)+vel_flat(15)+mask(1) → node
  encoder MLP → one kNN graph (k=24) → **10× graph-transformer layers** → per-node
  **temporal MHSA head** over 5 output tokens → decoder MLP → residual-add last
  frame → zero airfoil rows. ~7.5M params @ hidden=256, 8 heads.
- **Swappable backbone** (`backbones.py`): `{gat, meshgraphnet, graph_transformer}`;
  the submitted checkpoint uses `graph_transformer` (local attention with
  edge-conditioned **value** bias). All three are flat single-scale stacks.
- **Wrapper tricks** (`model.py`), orthogonal to the architecture: 2-member mean
  ensemble (members differ only in input features — base = mask, "physfeat" = mask +
  unsigned distance-to-airfoil + 3D direction), plus a persistence fallback heuristic.

**Hierarchy ghost:** `graph_utils.py` ships `farthest_point_sample` (a fast
boundary-aware voxel-representative pooler that keeps airfoil points first) and
`knn_interpolate` (IDW unpool), and the member takes a `use_hierarchical` flag — but
it **raises NotImplementedError** (`gnn_base.py:798`). The pooling primitives and
on-ramp exist; the hierarchy was never switched on. This design finishes that.

## 3. Experiment structure: two runs of one configurable model

- **Run 1 — flat baseline** (`num_levels=1`): a faithful single-member
  graph-transformer trunk with our recipe/features. Establishes where a flat graph
  net lands. NOT directly comparable to 0.0589 (that is a 2-member ensemble + fallback
  with their features); the meaningful comparisons are Run 2 vs Run 1 vs ResMLPMin,
  all under our identical recipe and features.
- **Run 2 — Graph-U-Net** (`num_levels=3`): same trunk made hierarchical. The delta
  Run 2 − Run 1 isolates the hierarchy — the post's core claim.

Per the "isolate one change per training run" rule, these are two separate runs; the
model code is shipped once, then the `num_levels` knob is flipped.

## 4. Model: `GraphUNet` (`models/graph_unet/`)

One configurable class. `num_levels=1` collapses to the flat baseline (switching on
the on-ramp the original stubbed out). Self-contained directory: `model.py`,
`__init__.py`, `norm_stats.pt` (copied from `models/resmlp_min/` — identical velocity
normalization). Registered in `models/__init__.py`. `main.py` updated to import it as
`Model` for the contract check.

### 4.1 Constructor (defaults — capacity-matched to ResMLPMin ~7.5M @ hidden=256)

```
backbone="graph_transformer"   # gat / meshgraphnet also selectable
hidden=256
heads=8
k=24                            # full-res kNN; 6th-place value
num_levels=3                    # 1 => flat baseline
pool_ratio=0.25                 # each level keeps 1/4 of the points
blocks_per_level=2             # graph-transformer layers per encoder level (decoder mirrors)
                               #   tuned at impl time so total params ~= 7.5M for BOTH configs
# temporal head
t_out=5, temporal_attn_layers=2
# feature freqs (ResMLPMin set)
num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6
use_checkpoint=True
```

**Param budget:** both the flat baseline and the Graph-U-Net are tuned (via layer
counts) to land at **~7.5M params**, matching ResMLPMin (7.72M) and the 6th-place
member (~7.5M). A param-count assertion/printout is added so the budget is verified,
not assumed. Capacity held ~constant; only the structure varies across the three-way
comparison.

### 4.2 Features (ResMLPMin set)

Per node: `[Fourier(pos, 10 bands) | velocity_history (5x3=15) |
Fourier(last_velocity, 3 bands) | distance-to-airfoil (dist, log1p(dist),
Fourier(log-dist, 6 bands)) | airfoil_mask (1)]`. Velocity normalized by
`vel_mean/vel_std` from `norm_stats.pt`. Distance-to-airfoil is the unsigned nearest
-airfoil-surface-point distance (same `_sdf` chunked-cdist helper as ResMLPMin), NOT
a true signed field.

### 4.3 Forward (per sample; loops over batch B, since pooling makes batches ragged)

```
feats -> node encoder MLP -> h0 @ full-res positions p0
ENCODER
  level 0: knn(p0,k) -> blocks_per_level x graph-transformer layers -> skip s0
  pool p0 -> p1 (ratio 0.25, boundary-aware, airfoil points first); gather h0->h1
  level 1: knn(p1,k) -> blocks_per_level layers -> skip s1
  pool p1 -> p2; gather
  level 2 (bottleneck): knn(p2,k) -> blocks_per_level layers
DECODER
  interp p2->p1 (IDW kNN, k=3); concat with s1; linear fuse; blocks_per_level layers
  interp p1->p0; concat with s0; linear fuse; blocks_per_level layers
HEAD
  per-node temporal MHSA (5 tokens, temporal_attn_layers) -> decoder MLP
  -> + last input frame (residual delta) -> zero airfoil rows (hard no-slip)
```

- **Pooling:** boundary-aware voxel-representative pooler (the original's
  `farthest_point_sample` with `priority_idx=airfoil_idx`), so airfoil points and
  fine structure survive to coarse levels — important for no-slip and the wake region.
- **kNN:** `torch_cluster.knn` on GPU (as in `PTPointNetA`), rebuilt per level from
  pooled positions. Replaces the original's CPU scipy.cKDTree.
- **Unpool:** inverse-distance-weighted kNN interpolation (k=3).
- **Skips:** **concatenated** then linearly fused (mirrors the voxel U-Net's
  `cat([u, skip])`), so fine detail is never overwritten — the "don't blur"
  mechanism.
- **Backbone layer:** the 6th-place `GraphTransformerLayer` (edge-conditioned value
  bias via `EdgeEncoder(rel_pos, dist)`), vendored into our directory. `gat` /
  `meshgraphnet` kept selectable for optional sweeps.

### 4.4 Contract

Implements `(t:[B,10], pos:[B,N,3], idcs_airfoil:list[Tensor], velocity_in:[B,5,N,3])
-> velocity_out:[B,5,N,3]`, residual-delta against `velocity_in[:,-1]`, hard no-slip
zeroing of airfoil rows. Constructable with `model_kwargs`; loads `state_dict.pt` if
present (for the `main.py` inference path). Passes `python main.py`.

## 5. Configs (with `run_name`) & training recipe

Four files, all carrying a `run_name` (like `configs/resmlp_min.yaml`):

- `configs/graph_baseline.yaml` — `run_name: graph_baseline`, `model_kwargs.num_levels: 1`
- `configs/graph_unet.yaml` — `run_name: graph_unet`, `model_kwargs.num_levels: 3`
- `configs/eval/graph_baseline_holdout.yaml`
- `configs/eval/graph_unet_holdout.yaml`

Locked recipe (matched to the ablation ladder for comparability): lr 5e-4 cosine,
150 epochs, bf16, EMA 0.999, augmented split (`augmented_split.json`,
`/workspace/data/augmented_train_warped-ifw`), `batch_size: 4` (full-res memory),
`num_workers`, `use_checkpoint: true` for train / `false` for eval. Training uses the
existing `train.py` (masked per-point L2 loss, `train.py:106-110`); evaluation uses
the existing `eval.py` on the 95-sample geometry-disjoint holdout
(`holdout_split.json`, `/workspace/data/warped-ifw-test-set`), reporting All / Wake
(10%) / Rest (90%).

## 6. Success criteria

- **Primary:** Graph-U-Net holdout rel-L2 <= ~0.063 (ties ResMLPMin) => the
  trunk+hierarchy recipe ports to a pure graph net.
- **Secondary:** Graph-U-Net < flat baseline => the hierarchy itself is the lever
  (validates the post's "flat stack blurs, hierarchy preserves" claim), at held-
  constant ~7.5M params.

## 7. Risks / mitigations

- **Full-resolution memory is the real cost (not depth).** Each full-res
  graph-transformer layer materializes a `(N, k, D)` neighbor gather. Mitigation:
  few full-res layers + gradient checkpointing + the hierarchy itself, which offloads
  reach to cheap coarse levels (N/4, N/16). Depth scales compute linearly and, with
  checkpointing, is ~memory-free.
- **Not apples-to-apples with 0.0589.** Our flat baseline is single-member with our
  features, not their 2-member + fallback. Stated explicitly; comparisons are within
  our own recipe.
- **Per-sample loop speed.** kNN rebuilt per level per sample on GPU; acceptable at
  B=4 (matches the original's per-sample structure).
- **Pooling determinism / airfoil preservation.** Boundary-aware pooler keeps airfoil
  points first; deterministic representative selection.

## 8. Deliverables

1. `models/graph_unet/{model.py, __init__.py, norm_stats.pt}`
2. `models/__init__.py` registration + `main.py` import line
3. `configs/graph_baseline.yaml`, `configs/graph_unet.yaml`
4. `configs/eval/graph_baseline_holdout.yaml`, `configs/eval/graph_unet_holdout.yaml`
5. Forward-shape smoke test (`python main.py`) + a param-count check confirming
   ~7.5M for both configs.
```
