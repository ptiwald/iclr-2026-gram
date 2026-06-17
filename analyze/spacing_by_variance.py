"""Is the HARD (high-variance) region finely or coarsely sampled?

The holdout "Wake (10%)" split = top-10% of points by temporal velocity
variance (p90 cut), per voxel_ladder_results.md / receptive_field report.
Here we ask the spacing question conditioned on that split directly, instead
of using distance-to-surface as a proxy:

  for each sample, compute per-point variance of the 10-frame velocity stack,
  take the p90 mask, and compare 1-NN spacing of the hot (wake) points vs the
  cold (rest) points -- then line both up against the voxel-grid pitches.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATA = Path("/home/paul/scratch/gram-competition/warped-ifw")
X_EXTENT = 2.25
LADDER = [("32³", 32), ("64³ baseline", 64), ("128³", 128), ("256³ higher_res", 256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(DATA.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in sorted(rng.permutation(len(files))[: args.num_samples])]
    print(f"{len(files)} samples")

    hot_nn, cold_nn = [], []
    for f in files:
        d = np.load(f)
        pos = d["pos"].astype(np.float32)
        v = np.concatenate([d["velocity_in"], d["velocity_out"]], axis=0)  # (10,N,3)
        var = v.var(axis=0).sum(axis=-1)        # (N,) trace of temporal covariance
        thr = np.percentile(var, 90)
        hot = var >= thr
        nn = cKDTree(pos).query(pos, k=2)[0][:, 1]
        hot_nn.append(nn[hot])
        cold_nn.append(nn[~hot])

    hot_nn = np.concatenate(hot_nn)
    cold_nn = np.concatenate(cold_nn)

    print("\n=== voxel pitch ===")
    for label, N in LADDER:
        print(f"  {label:<16} Δx = {X_EXTENT / N:.4f}")

    print("\n=== 1-NN spacing: hot (p90 variance = 'Wake') vs cold (rest) ===")
    print(f"{'set':<14} {'n':>10} {'p05':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    for name, a in [("hot (wake)", hot_nn), ("cold (rest)", cold_nn)]:
        ps = np.percentile(a, [5, 25, 50, 75, 95])
        print(f"{name:<14} {a.size:>10} " + " ".join(f"{x:>8.4f}" for x in ps))

    hm, cm = np.median(hot_nn), np.median(cold_nn)
    print(f"\nhot/cold median spacing ratio = {hm/cm:.2f}  "
          f"(hot {hm:.4f} vs cold {cm:.4f})")
    print(f"finest grid (256³) Δx = {X_EXTENT/256:.4f}")
    print(f"  -> hot-region spacing is {(X_EXTENT/256)/hm:.1f}x FINER than the 256³ grid pitch")
    for label, N in LADDER:
        dx = X_EXTENT / N
        frac = (hot_nn < dx).mean()
        print(f"  {label:<16} Δx={dx:.4f}: {frac:5.1%} of wake points are finer-spaced than this grid")


if __name__ == "__main__":
    main()
