"""Diagnose information content of surface supernodes in AB-UPT.

Question: when we sample N_s supernodes from idcs_airfoil and then run kNN
against the FULL 100k-point cloud (as the model's encoder does), how many of
each supernode's k neighbors are themselves on-surface vs. off-surface (volume)?

If most neighbors are on-surface, surface supernodes pool near-zero velocities
and mostly act as geometry anchors — in that regime, increasing k or decreasing
N_s (sparser surface sampling) would help them see actual flow.

Configurable via CLI flags:
  --num_surface_supernodes  N_s (matches model default 128)
  --k                       fixed k to analyze in detail (matches model default 8)
  --k_sweep                 comma-separated k values for the sweep table
  --num_samples             how many .npz files to process
  --seed                    RNG seed for reproducible supernode sampling
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")


def sample_surface_supernodes(idcs_airfoil: np.ndarray, n_s: int, rng: np.random.Generator) -> np.ndarray:
    """Mirror model._sample_supernodes: random choice with replacement if too few."""
    n = idcs_airfoil.size
    if n >= n_s:
        picks = rng.permutation(n)[:n_s]
    else:
        picks = rng.integers(0, n, size=n_s)
    return idcs_airfoil[picks]


def analyze_one_sample(pos: np.ndarray, idcs_airfoil: np.ndarray,
                       n_s: int, k_values: list[int], rng: np.random.Generator):
    """Return dict: k -> (N_s,) array with count of on-surface neighbors per supernode."""
    surf_idx = sample_surface_supernodes(idcs_airfoil, n_s, rng)
    surf_pos = pos[surf_idx]

    k_max = max(k_values)
    tree = cKDTree(pos)
    # Query k_max+1 because the supernode itself is in `pos` and will be its own nearest neighbor.
    # We strip that self-match below so the reported counts describe "other points in the neighborhood".
    _, neigh = tree.query(surf_pos, k=k_max + 1)  # (N_s, k_max+1)

    is_surface = np.zeros(pos.shape[0], dtype=bool)
    is_surface[idcs_airfoil] = True

    # Drop the self-match (always the 0-distance hit at column 0).
    neigh = neigh[:, 1:]  # (N_s, k_max)

    neigh_on_surface = is_surface[neigh]  # (N_s, k_max) bool

    out = {}
    for k in k_values:
        out[k] = neigh_on_surface[:, :k].sum(axis=1)  # (N_s,) counts in [0, k]
    return out, surf_idx.size, idcs_airfoil.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_surface_supernodes", type=int, default=128)
    parser.add_argument("--k", type=int, default=8, help="k used in the detailed histogram")
    parser.add_argument("--k_sweep", type=str, default="1,2,4,8,16,32,64",
                        help="comma-separated k values for the sweep table")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    k_sweep = sorted({int(x) for x in args.k_sweep.split(",")} | {args.k})
    rng = np.random.default_rng(args.seed)

    files = sorted(DATA_DIR.glob("*.npz"))[:args.num_samples]
    print(f"Data dir: {DATA_DIR}")
    print(f"Analyzing {len(files)} files | N_s = {args.num_surface_supernodes} | "
          f"k values = {k_sweep}")
    print()

    # Collect counts across all samples and all supernodes, per k.
    all_counts = {k: [] for k in k_sweep}
    airfoil_sizes = []
    n_total_points = None

    for f in files:
        data = np.load(f)
        pos = data["pos"]              # (100000, 3)
        idcs = data["idcs_airfoil"]    # (variable,)
        if n_total_points is None:
            n_total_points = pos.shape[0]

        counts, n_s_actual, n_airfoil = analyze_one_sample(
            pos, idcs, args.num_surface_supernodes, k_sweep, rng,
        )
        airfoil_sizes.append(n_airfoil)
        for k in k_sweep:
            all_counts[k].append(counts[k])

    airfoil_sizes = np.array(airfoil_sizes)
    surface_fraction = airfoil_sizes / n_total_points

    print(f"=== Airfoil-point counts across {len(files)} samples ===")
    print(f"  airfoil points per sample: min={airfoil_sizes.min()}  "
          f"median={int(np.median(airfoil_sizes))}  max={airfoil_sizes.max()}  "
          f"mean={airfoil_sizes.mean():.0f}")
    print(f"  fraction of cloud that is on-surface: mean={surface_fraction.mean():.4f}  "
          f"(=> random baseline for 'neighbor is on-surface')")
    print()

    print("=== Per-supernode: fraction of k nearest neighbors that are on-surface ===")
    print(f"{'k':>4} | {'mean':>7} {'std':>7} {'p05':>7} {'p50':>7} {'p95':>7} | "
          f"{'P(all k on-surf)':>18} {'P(none on-surf)':>17}")
    print("-" * 92)
    for k in k_sweep:
        counts = np.concatenate(all_counts[k])  # (num_samples * N_s,)
        frac = counts / k
        all_on = (counts == k).mean()
        all_off = (counts == 0).mean()
        print(f"{k:>4} | "
              f"{frac.mean():>7.4f} {frac.std():>7.4f} "
              f"{np.percentile(frac, 5):>7.4f} {np.percentile(frac, 50):>7.4f} "
              f"{np.percentile(frac, 95):>7.4f} | "
              f"{all_on:>18.4f} {all_off:>17.4f}")
    print()

    # Detailed histogram at the chosen k.
    k = args.k
    counts = np.concatenate(all_counts[k])
    print(f"=== Histogram of on-surface neighbor count @ k={k} "
          f"({counts.size} supernodes across {len(files)} samples) ===")
    bins = np.bincount(counts, minlength=k + 1)
    for c in range(k + 1):
        bar = "#" * int(60 * bins[c] / bins.max()) if bins.max() > 0 else ""
        print(f"  {c:>2}/{k} on-surface: {bins[c]:>7d}  ({bins[c]/counts.size:>6.2%})  {bar}")
    print()

    # Interpretation nudge.
    mean_frac_at_k = np.concatenate(all_counts[k]).mean() / k
    random_baseline = surface_fraction.mean()
    enrichment = mean_frac_at_k / random_baseline if random_baseline > 0 else float("nan")
    print(f"=== Summary @ k={k} ===")
    print(f"  mean on-surface fraction: {mean_frac_at_k:.4f}")
    print(f"  random baseline:          {random_baseline:.4f}")
    print(f"  enrichment over random:   {enrichment:.1f}x")
    print("  (high enrichment + high fraction => supernode neighborhoods are")
    print("   dominated by on-surface points and carry little velocity signal)")


if __name__ == "__main__":
    main()
