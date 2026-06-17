"""Point-spacing distribution vs. the model voxel grids.

Question (for the substrate/Nyquist thesis): the 100k cloud is sampled
*non-uniformly* — dense near the airfoil surface, sparse in the far field.
What is the actual local point spacing, how does it vary by region, and how
does it line up with the voxel pitch of each rung of the resolution ladder?

Method
------
- Local spacing = distance to the 1st nearest neighbour (1-NN) per point.
  The mesh is dyadically refined, so 1-NN distances cluster at a few discrete
  levels -> the pooled histogram should show clear peaks.
- Region = distance from each point to the nearest airfoil-surface point
  (KDTree over idcs_airfoil), bucketed into surface / near / mid / far bands
  scaled by the sample's streamwise chord so bands are geometry-relative.
- Overlay the streamwise voxel pitch Dx = x_extent / N for each ladder rung.
  x_extent = 2 * POS_SCALE[0] = 2.25 (union-of-bboxes from norm_stats).

Outputs a pooled-across-samples histogram PNG plus a printed numeric summary
(peak locations, per-region percentiles, and the grid resolution implied by
the near-surface spacing).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

DATA = Path("/home/paul/scratch/gram-competition/warped-ifw")
OUT = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/point_spacing")

# union-of-training-bboxes from norm_stats (pos_mean +/- pos_scale)
POS_SCALE = np.array([1.125, 0.4391, 0.640625])
X_EXTENT = 2 * POS_SCALE[0]  # 2.25 streamwise span

# resolution-ladder rungs: (label, N_x).  Dx = X_EXTENT / N_x.
LADDER = [
    ("32x16x16", 32),
    ("64x32x32 (baseline)", 64),
    ("128x64x64", 128),
    ("256x128x128 (higher_res)", 256),
]

# region bands as fractions of streamwise chord (distance to nearest surface pt)
# surface = exactly on idcs_airfoil; the rest split by chord-relative distance.
BANDS = [
    ("surface", 0.0, 0.0),       # on idcs_airfoil
    ("near (<0.1c)", 0.0, 0.10),
    ("mid (0.1-0.5c)", 0.10, 0.50),
    ("far (>0.5c)", 0.50, np.inf),
]


def spacing_one_sample(pos: np.ndarray, idcs_airfoil: np.ndarray):
    """Return (nn_dist[N], dist_to_surface[N], on_surface[N] bool, chord_x)."""
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=2)        # col 0 is self (dist 0)
    nn = d[:, 1]

    surf_pos = pos[idcs_airfoil]
    surf_tree = cKDTree(surf_pos)
    d2s, _ = surf_tree.query(pos, k=1)

    on_surf = np.zeros(pos.shape[0], dtype=bool)
    on_surf[idcs_airfoil] = True

    chord_x = surf_pos[:, 0].max() - surf_pos[:, 0].min()
    return nn, d2s, on_surf, chord_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--all", action="store_true", help="use every .npz file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    if not args.all:
        idx = rng.permutation(len(files))[: args.num_samples]
        files = [files[i] for i in sorted(idx)]
    print(f"Pooling local spacing over {len(files)} samples (x_extent={X_EXTENT})")

    # per-band pooled 1-NN distances
    band_nn = {b[0]: [] for b in BANDS}
    chords = []
    all_nn = []

    for n, f in enumerate(files):
        d = np.load(f)
        pos = d["pos"].astype(np.float32)
        idcs = d["idcs_airfoil"]
        nn, d2s, on_surf, chord_x = spacing_one_sample(pos, idcs)
        chords.append(chord_x)
        all_nn.append(nn)
        d2s_c = d2s / chord_x  # chord-relative distance to surface
        for name, lo, hi in BANDS:
            if name == "surface":
                mask = on_surf
            else:
                mask = (~on_surf) & (d2s_c >= lo) & (d2s_c < hi)
            band_nn[name].append(nn[mask])
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(files)}")

    all_nn = np.concatenate(all_nn)
    chords = np.array(chords)
    band_nn = {k: np.concatenate(v) for k, v in band_nn.items()}

    # ---- numeric summary --------------------------------------------------
    print(f"\nchord_x across samples: median={np.median(chords):.3f} "
          f"[{chords.min():.3f}, {chords.max():.3f}]")

    print("\n=== voxel pitch (streamwise Dx) per ladder rung ===")
    for label, N in LADDER:
        print(f"  {label:<26} Dx = {X_EXTENT / N:.4f}")

    print("\n=== 1-NN spacing percentiles, pooled by region ===")
    print(f"{'region':<16} {'n_pts':>10} {'p05':>8} {'p25':>8} {'p50':>8} "
          f"{'p75':>8} {'p95':>8}")
    for name, _, _ in BANDS:
        v = band_nn[name]
        if v.size == 0:
            continue
        ps = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"{name:<16} {v.size:>10} "
              + " ".join(f"{x:>8.4f}" for x in ps))
    ps_all = np.percentile(all_nn, [5, 25, 50, 75, 95])
    print(f"{'ALL':<16} {all_nn.size:>10} " + " ".join(f"{x:>8.4f}" for x in ps_all))

    # grid resolution implied by the near-surface median spacing
    near_med = np.median(band_nn["near (<0.1c)"])
    far_med = np.median(band_nn["far (>0.5c)"])
    print(f"\nNear-surface median spacing  = {near_med:.4f} "
          f"-> needs N_x ~ {X_EXTENT / near_med:.0f} to match (Nyquist ~"
          f"{X_EXTENT / (2*near_med):.0f})")
    print(f"Far-field   median spacing  = {far_med:.4f} "
          f"-> needs N_x ~ {X_EXTENT / far_med:.0f}")
    print(f"density contrast (far/near) = {far_med / near_med:.1f}x")

    # ---- plot -------------------------------------------------------------
    bins = np.logspace(np.log10(max(all_nn.min(), 1e-4)),
                       np.log10(all_nn.max()), 120)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9),
                                   constrained_layout=True, sharex=True)

    # top: pooled histogram (all points) — show the dyadic peaks
    ax1.hist(all_nn, bins=bins, color="0.5", alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_ylabel("point count")
    ax1.set_title(f"1-NN point spacing, all regions pooled "
                  f"({len(files)} samples, {all_nn.size:,} points)")

    # bottom: per-region overlay (normalised densities)
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for (name, _, _), c in zip(BANDS, colors):
        v = band_nn[name]
        if v.size == 0:
            continue
        ax2.hist(v, bins=bins, density=True, histtype="step", lw=2,
                 color=c, label=f"{name}  (median {np.median(v):.4f})")
    ax2.set_xscale("log")
    ax2.set_xlabel("distance to 1st nearest neighbour  (sim units, x-extent=2.25)")
    ax2.set_ylabel("density")
    ax2.set_title("spacing by region (chord-relative distance to airfoil surface)")
    ax2.legend(loc="upper left", fontsize=9)

    # overlay voxel pitch lines on both panels (staggered labels on top panel)
    for ax in (ax1, ax2):
        for (label, N), ls in zip(LADDER, ["-.", "-", "--", ":"]):
            dx = X_EXTENT / N
            ax.axvline(dx, color="k", ls=ls, lw=1.3, alpha=0.8)
    for i, (label, N) in enumerate(LADDER):
        dx = X_EXTENT / N
        y = 0.97 - 0.11 * i  # stagger so labels don't collide
        ax1.annotate(f"{label}  Δx={dx:.4f}", (dx, y),
                     xycoords=("data", "axes fraction"),
                     va="top", ha="left", fontsize=8, color="k",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="0.7", alpha=0.85))

    fig.suptitle(
        "Non-uniform point spacing vs. voxel-grid pitch\n"
        "voxel lines = streamwise cell size of each resolution-ladder rung",
        fontsize=12,
    )
    out = OUT / "point_spacing_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
