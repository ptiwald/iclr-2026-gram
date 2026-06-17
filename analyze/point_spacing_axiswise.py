"""Per-axis point spacing + dense-population fraction.

Two questions:
 (A) Is the cloud sampled anisotropically — denser along x (flow direction)?
     For each point, among its k nearest neighbours, assign each neighbour to
     the axis of its dominant displacement component and take, per axis, the
     nearest such neighbour's |Δ_axis|. That is the local face-spacing along
     each axis. Pool and compare x vs y vs z (and the x:y:z ratio).
 (B) What fraction of points live in a densely sampled area? Reported two ways:
     by isotropic 1-NN spacing (CDF + fraction below each grid pitch) and by
     chord-relative distance to the airfoil surface (band populations).

The model grids are anisotropic: (N, N/2, N/2) over extents (2.25, 0.878,
1.281), so each axis gets its OWN pitch line:  Δ_axis = extent_axis / N_axis.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

DATA = Path("/home/paul/scratch/gram-competition/warped-ifw")
OUT = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/point_spacing")

POS_SCALE = np.array([1.125, 0.4391, 0.640625])
EXT = 2 * POS_SCALE                       # (2.25, 0.8782, 1.28125)
AXES = ["x (streamwise)", "y (spanwise)", "z (vertical)"]
# ladder rungs as (label, N_x); grid is anisotropic (N, N/2, N/2) — NOT a cube
LADDER = [("32×16×16", 32), ("64×32×32", 64), ("128×64×64", 128), ("256×128×128", 256)]
K = 16


def grid_shape(nx):
    return np.array([nx, nx // 2, nx // 2])


def per_axis_spacing(pos):
    """(N,3) nearest-neighbour spacing per axis via dominant-component binning."""
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=K + 1)
    neigh = idx[:, 1:]                          # (N,K)
    disp = np.abs(pos[neigh] - pos[:, None, :])  # (N,K,3)
    dom = np.argmax(disp, axis=2)              # (N,K) dominant axis per neighbour
    out = np.full((pos.shape[0], 3), np.nan, dtype=np.float32)
    for a in range(3):
        comp = np.where(dom == a, disp[:, :, a], np.inf)  # (N,K)
        m = comp.min(axis=1)
        out[:, a] = np.where(np.isfinite(m), m, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in sorted(rng.permutation(len(files))[: args.num_samples])]
    print(f"{len(files)} samples\n")

    ax_sp = [[], [], []]   # per-axis spacings
    iso_nn = []            # isotropic 1-NN spacing
    d2s_chord = []         # distance to surface / chord
    on_surf_all = []

    for f in files:
        d = np.load(f)
        pos = d["pos"].astype(np.float32)
        idcs = d["idcs_airfoil"]
        # isotropic 1-NN
        tree = cKDTree(pos)
        iso_nn.append(tree.query(pos, k=2)[0][:, 1])
        # per-axis
        pa = per_axis_spacing(pos)
        for a in range(3):
            v = pa[:, a]
            ax_sp[a].append(v[np.isfinite(v)])
        # distance to surface
        chord = pos[idcs, 0].max() - pos[idcs, 0].min()
        d2s = cKDTree(pos[idcs]).query(pos, k=1)[0]
        d2s_chord.append(d2s / chord)
        on = np.zeros(pos.shape[0], bool); on[idcs] = True
        on_surf_all.append(on)

    ax_sp = [np.concatenate(a) for a in ax_sp]
    iso_nn = np.concatenate(iso_nn)
    d2s_chord = np.concatenate(d2s_chord)
    on_surf_all = np.concatenate(on_surf_all)
    N = iso_nn.size

    # ---- (A) anisotropy ---------------------------------------------------
    print("=== (A) per-axis nearest-neighbour spacing ===")
    print(f"{'axis':<16} {'p25':>8} {'p50':>8} {'p75':>8}")
    med = []
    for a in range(3):
        ps = np.percentile(ax_sp[a], [25, 50, 75])
        med.append(ps[1])
        print(f"{AXES[a]:<16} " + " ".join(f"{x:>8.4f}" for x in ps))
    print(f"\nmedian spacing ratio  x:y:z = "
          f"{med[0]/med[2]:.2f} : {med[1]/med[2]:.2f} : 1.00   (normalised to z)")
    print(f"  x/y = {med[0]/med[1]:.2f}   x/z = {med[0]/med[2]:.2f}   y/z = {med[1]/med[2]:.2f}")
    print("grid pitch ratio for reference (anisotropic grid, N:N/2:N/2 over "
          f"{EXT[0]:.3f}:{EXT[1]:.3f}:{EXT[2]:.3f}):")
    gp = EXT / grid_shape(64)
    print(f"  Δx:Δy:Δz = {gp[0]/gp[2]:.2f} : {gp[1]/gp[2]:.2f} : 1.00\n")

    # ---- (B) dense fraction ----------------------------------------------
    print("=== (B) fraction of points in a densely sampled area ===")
    print("by chord-relative distance to airfoil surface:")
    for lab, lo, hi in [("on surface", -1, 0), ("<0.05c", 0, 0.05),
                        ("<0.1c", 0, 0.1), ("<0.25c", 0, 0.25),
                        ("<0.5c", 0, 0.5), (">=0.5c", 0.5, np.inf)]:
        if lab == "on surface":
            frac = on_surf_all.mean()
        else:
            frac = ((~on_surf_all) & (d2s_chord >= lo) & (d2s_chord < hi)).mean()
        print(f"  {lab:<12} {frac:6.1%}")
    print("\nby isotropic 1-NN spacing — fraction of points finer than each grid pitch")
    print("(grid pitch = geometric mean of the 3 anisotropic pitches):")
    for label, nx in LADDER:
        dx = float(np.exp(np.log(EXT / grid_shape(nx)).mean()))
        print(f"  {label:<10} ⟨Δ⟩={dx:.4f}:  {(iso_nn < dx).mean():6.1%} of points finer-spaced")
    print(f"\nisotropic 1-NN spacing percentiles: "
          + " ".join(f"p{p}={np.percentile(iso_nn,p):.4f}"
                     for p in (10, 25, 50, 75, 90)))

    # ---- plot: per-axis histograms ---------------------------------------
    lo = max(min(a.min() for a in ax_sp), 1e-4)
    hi = max(a.max() for a in ax_sp)
    bins = np.logspace(np.log10(lo), np.log10(hi), 120)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True,
                             sharex=True)
    colors = ["tab:blue", "tab:green", "tab:red"]
    for a in range(3):
        axes[a].hist(ax_sp[a], bins=bins, color=colors[a], alpha=0.8)
        axes[a].set_xscale("log")
        axes[a].set_ylabel("point count")
        axes[a].set_title(f"{AXES[a]} — nearest-neighbour spacing "
                          f"(median {med[a]:.4f})")
        for (label, nx), ls in zip(LADDER, ["-.", "-", "--", ":"]):
            dx = (EXT / grid_shape(nx))[a]
            axes[a].axvline(dx, color="k", ls=ls, lw=1.2, alpha=0.8)
            axes[a].annotate(f"{label} Δ={dx:.4f}", (dx, 0.95),
                             xycoords=("data", "axes fraction"),
                             rotation=90, va="top", ha="right", fontsize=7)
    axes[-1].set_xlabel("nearest-neighbour spacing along axis (sim units)")
    fig.suptitle("Per-axis point spacing vs. each axis's voxel pitch\n"
                 "(grid is anisotropic: N×N/2×N/2)", fontsize=12)
    out = OUT / "point_spacing_axiswise.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
