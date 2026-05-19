"""Render side-view variance heatmaps for the representative sample.

Two views:
  (a) all points projected onto x-z, coloured by var_resid (log scale)
  (b) a thin y-slice through the airfoil mid-plane (|y - y0| < SLAB_HALF),
      same colouring

The mesh is unstructured but adaptive: per-axis spans are subdivided dyadically
(max neighbour spacing along each axis is 1/32 of the span), so points "line
up" in projection even though no two points share exact coordinates. Picking
a thin slab in y reveals one quasi-2D layer of the refinement.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np

OUT_DIR = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/receptive_field_physics")
SLAB_HALF = 0.03  # y-slab half-thickness around the airfoil mid-plane


def render(rep: str, out_name: str):
    d = np.load(rep)
    pos = d["pos"]
    idcs_airfoil = d["idcs_airfoil"]
    v_in = d["velocity_in"]
    v_out = d["velocity_out"]
    resid = v_out - v_in[-1:]
    var_resid = resid.var(axis=0).sum(axis=-1)
    var_full = np.concatenate([v_in, v_out], axis=0).var(axis=0).sum(axis=-1)

    # mid-plane y0 from the airfoil
    y0 = pos[idcs_airfoil, 1].mean()
    slab = np.abs(pos[:, 1] - y0) < SLAB_HALF
    af_mask = np.zeros(pos.shape[0], dtype=bool)
    af_mask[idcs_airfoil] = True

    # color norm: log-scale across var_resid > 0
    vmin = max(np.percentile(var_resid[var_resid > 0], 1), 1e-8)
    vmax = var_resid.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = "magma"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # (a) full point cloud in x-z
    ax = axes[0]
    order = np.argsort(var_resid)  # draw hot points on top
    sc = ax.scatter(
        pos[order, 0], pos[order, 2],
        c=var_resid[order], cmap=cmap, norm=norm,
        s=1.5, linewidths=0, alpha=0.6,
    )
    # outline airfoil
    ax.scatter(pos[idcs_airfoil, 0], pos[idcs_airfoil, 2],
               s=1, c="cyan", alpha=0.4, linewidths=0, label="airfoil")
    ax.set_xlabel("x  (streamwise)")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title(f"(a) x-z projection of all 100k points\nsample {Path(rep).name}")
    ax.legend(loc="upper right", fontsize=8)

    # (b) y-slab, filled as a triangulated mesh
    ax = axes[1]
    p = pos[slab]
    v = var_resid[slab]
    # Delaunay in (x, z). Mask out long-edge triangles so we don't fill across
    # gaps in the unstructured mesh (e.g. the interior of the airfoil).
    tri = mtri.Triangulation(p[:, 0], p[:, 2])
    t_pts = p[tri.triangles]              # (T, 3, 3) — but we only need x,z
    xy = np.stack([p[:, 0], p[:, 2]], axis=-1)
    txy = xy[tri.triangles]               # (T, 3, 2)
    edge_lens = np.stack([
        np.linalg.norm(txy[:, 0] - txy[:, 1], axis=-1),
        np.linalg.norm(txy[:, 1] - txy[:, 2], axis=-1),
        np.linalg.norm(txy[:, 2] - txy[:, 0], axis=-1),
    ], axis=-1).max(axis=-1)
    # threshold = 3× the mesh's finest in-plane spacing (1/32 of domain span ≈ 0.03)
    edge_cap = 3 * 0.03125 * max(pos[:, 0].max() - pos[:, 0].min(),
                                 pos[:, 2].max() - pos[:, 2].min())
    tri.set_mask(edge_lens > edge_cap)
    sc2 = ax.tripcolor(tri, v, cmap=cmap, norm=norm, shading="gouraud")
    af_slab = af_mask & slab
    ax.scatter(pos[af_slab, 0], pos[af_slab, 2],
               s=1, c="cyan", alpha=0.6, linewidths=0, label="airfoil ∩ slab")
    ax.set_xlabel("x  (streamwise)")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title(f"(b) y-slab |y − y0| < {SLAB_HALF} (y0={y0:+.3f}) — filled "
                 f"Delaunay\n{slab.sum()} pts, {(~tri.mask).sum()} kept triangles "
                 f"(edge ≤ {edge_cap:.3f})")
    ax.legend(loc="upper right", fontsize=8)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.9, location="right", pad=0.02)
    cbar.set_label("var(v_out − v_in[-1])  (log scale)")

    fig.suptitle(
        f"Side-view variance heatmap — sample {Path(rep).name}\n"
        f"var_resid ranges over {np.log10(vmax/vmin):.1f} decades; "
        f"mesh is unstructured but dyadically refined (octree-like), "
        f"hence the apparent column alignment in projection.",
        fontsize=10,
    )
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Wrote", out)


def main():
    samples = [
        ("/home/paul/scratch/gram-competition/warped-ifw/1094_10-2.npz", "sideview_heatmap.png"),
        ("/home/paul/scratch/gram-competition/warped-ifw/1021_10-0.npz", "sideview_heatmap_2af.png"),
        ("/home/paul/scratch/gram-competition/warped-ifw/1021_1-0.npz",  "sideview_heatmap_3af.png"),
    ]
    for rep, name in samples:
        render(rep, name)

    # y-density figure for the original representative
    rep = samples[0][0]
    d = np.load(rep)
    pos = d["pos"]
    y0 = pos[d["idcs_airfoil"], 1].mean()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(pos[:, 1], bins=120, color="grey")
    ax.axvspan(y0 - SLAB_HALF, y0 + SLAB_HALF, color="tab:red", alpha=0.25, label="slab")
    ax.set_xlabel("y")
    ax.set_ylabel("point count")
    ax.set_title(f"Point density along y — sample {Path(rep).name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "y_density.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
