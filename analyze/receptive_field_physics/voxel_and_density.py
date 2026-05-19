"""(b) Local point-density map and (c) voxel-grid overlay for the side view.

For a y-slab of the representative sample:
  - Compute the local k-NN radius (distance to the 16th nearest neighbour in 3D)
    as a per-point measure of effective mesh resolution. Render it as a filled
    Delaunay heatmap on x-z so the variation is visible against the geometry.
  - Overlay the shared 64 × 32 × 32 voxel grid used by VoxelBaseline,
    VoxelUNet*, and PTPointNetA. Annotate the median chord and the
    receptive-field window we inferred from the variance analysis (~1.15 chord_x).
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import cKDTree

OUT = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/receptive_field_physics")
DATA = "/home/paul/scratch/gram-competition/warped-ifw"

# union-of-training-bboxes from norm_stats (pos_mean ± pos_scale)
POS_MEAN = np.array([1.125, 0.0022, 0.640625])
POS_SCALE = np.array([1.125, 0.4391, 0.640625])
GRID = np.array([64, 32, 32])
VOXEL = 2 * POS_SCALE / GRID  # (Δx, Δy, Δz)

SLAB_HALF = 0.03
K = 16  # neighbours for local-radius estimate


def slab_data(path: str):
    d = np.load(path)
    pos = d["pos"]
    idcs_airfoil = d["idcs_airfoil"]
    y0 = pos[idcs_airfoil, 1].mean()
    slab = np.abs(pos[:, 1] - y0) < SLAB_HALF
    return pos, idcs_airfoil, slab, y0


def local_knn_radius(pos: np.ndarray, k: int = K) -> np.ndarray:
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=k + 1)
    return d[:, -1]


def filled_tri(ax, p_xz: np.ndarray, values: np.ndarray, norm, cmap, edge_cap: float):
    tri = mtri.Triangulation(p_xz[:, 0], p_xz[:, 1])
    txy = p_xz[tri.triangles]
    e = np.stack([
        np.linalg.norm(txy[:, 0] - txy[:, 1], axis=-1),
        np.linalg.norm(txy[:, 1] - txy[:, 2], axis=-1),
        np.linalg.norm(txy[:, 2] - txy[:, 0], axis=-1),
    ], axis=-1).max(axis=-1)
    tri.set_mask(e > edge_cap)
    return ax.tripcolor(tri, values, cmap=cmap, norm=norm, shading="gouraud")


def overlay_voxel_grid(ax, color="white", alpha=0.25, lw=0.4):
    # x-lines at i * Δx for i in 0..64; z-lines at j * Δz for j in 0..32
    x0 = POS_MEAN[0] - POS_SCALE[0]
    z0 = POS_MEAN[2] - POS_SCALE[2]
    xs = x0 + np.arange(GRID[0] + 1) * VOXEL[0]
    zs = z0 + np.arange(GRID[2] + 1) * VOXEL[2]
    for x in xs:
        ax.axvline(x, color=color, alpha=alpha, lw=lw)
    for z in zs:
        ax.axhline(z, color=color, alpha=alpha, lw=lw)


def render(rep: str, out_name: str):
    pos, idcs_airfoil, slab, y0 = slab_data(rep)
    p = pos[slab]
    p_xz = np.stack([p[:, 0], p[:, 2]], axis=-1)
    # local resolution from full 3D (not just slab) so the radius reflects the
    # actual nearest-neighbour distance in space
    full_r = local_knn_radius(pos, K)
    r = full_r[slab]
    af = np.zeros(pos.shape[0], dtype=bool); af[idcs_airfoil] = True
    af_slab = af & slab
    af_x = pos[idcs_airfoil, 0]
    chord_x = af_x.max() - af_x.min()
    af_max_x = af_x.max()

    # one big figure with two panels: (left) density / k-NN radius, (right) voxel overlay
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # left: filled k-NN radius heatmap (log scale; the radius spans >1 decade)
    norm_r = mcolors.LogNorm(vmin=np.percentile(r, 1), vmax=np.percentile(r, 99))
    sc = filled_tri(axes[0], p_xz, r, norm_r, "viridis",
                    edge_cap=3 * VOXEL[0])  # mesh refinement is dyadic; 3 voxels is a safe cap
    axes[0].scatter(pos[af_slab, 0], pos[af_slab, 2],
                    s=1, c="red", alpha=0.6, linewidths=0, label="airfoil ∩ slab")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x  (streamwise)")
    axes[0].set_ylabel("z")
    axes[0].set_title(
        f"(b) local k-NN radius (k={K}) in y-slab — sample {Path(rep).name}\n"
        f"  near-airfoil ≈ {np.percentile(r[af_slab[slab]], 50):.3f}, "
        f"far-field ≈ {np.percentile(r, 99):.3f}  "
        f"(ratio ≈ {np.percentile(r,99)/np.percentile(r[af_slab[slab]],50):.0f}×)"
    )
    plt.colorbar(sc, ax=axes[0], shrink=0.85, label=f"distance to {K}-th neighbour")
    axes[0].legend(loc="upper right", fontsize=8)

    # right: scatter (faint) + voxel grid overlay
    axes[1].scatter(p_xz[:, 0], p_xz[:, 1], s=1, c="lightgrey", alpha=0.4, linewidths=0)
    axes[1].scatter(pos[af_slab, 0], pos[af_slab, 2],
                    s=2, c="red", alpha=0.8, linewidths=0, label="airfoil ∩ slab")
    overlay_voxel_grid(axes[1], color="tab:blue", alpha=0.55, lw=0.5)
    # highlight one voxel cell at the airfoil leading-edge tip for scale
    x0 = POS_MEAN[0] - POS_SCALE[0]
    z0 = POS_MEAN[2] - POS_SCALE[2]
    i0 = int(np.floor((af_x.min() - x0) / VOXEL[0]))
    j0 = int(np.floor((pos[idcs_airfoil, 2].mean() - z0) / VOXEL[2]))
    rect_x = x0 + i0 * VOXEL[0]
    rect_z = z0 + j0 * VOXEL[2]
    axes[1].add_patch(plt.Rectangle((rect_x, rect_z), VOXEL[0], VOXEL[2],
                                    fc="tab:orange", ec="tab:orange", alpha=0.9,
                                    label=f"one voxel ({VOXEL[0]:.3f} × {VOXEL[2]:.3f})"))
    # chord-length window for receptive-field context
    axes[1].add_patch(plt.Rectangle((af_x.min(), pos[idcs_airfoil, 2].min()),
                                    1.15 * chord_x,
                                    pos[idcs_airfoil, 2].max() - pos[idcs_airfoil, 2].min(),
                                    fc="none", ec="tab:green", lw=2, ls="--",
                                    label=f"1.15 chord_x ≈ {1.15*chord_x:.2f} ≈ "
                                          f"{1.15*chord_x/VOXEL[0]:.0f} voxels in x"))
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x  (streamwise)")
    axes[1].set_ylabel("z")
    axes[1].set_title(
        f"(c) 64 × 32 × 32 voxel grid (shared by VoxelBaseline / VoxelUNet / PTPointNetA)\n"
        f"  Δx={VOXEL[0]:.4f}  Δz={VOXEL[2]:.4f}  →  "
        f"median chord ≈ {1.20/VOXEL[0]:.0f} voxels in x"
    )
    axes[1].legend(loc="upper right", fontsize=8)

    # zoom inset on the airfoil region so the voxel cells are individually visible
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(axes[1], width="40%", height="55%", loc="lower right",
                       bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
                       bbox_transform=axes[1].transAxes, borderpad=2)
    axins.scatter(p_xz[:, 0], p_xz[:, 1], s=2, c="lightgrey", alpha=0.6, linewidths=0)
    axins.scatter(pos[af_slab, 0], pos[af_slab, 2],
                  s=3, c="red", alpha=0.9, linewidths=0)
    overlay_voxel_grid(axins, color="tab:blue", alpha=0.55, lw=0.6)
    pad_x = 0.15 * chord_x
    pad_z = 0.15 * (pos[idcs_airfoil, 2].max() - pos[idcs_airfoil, 2].min())
    axins.set_xlim(af_x.min() - pad_x, af_x.max() + pad_x)
    axins.set_ylim(pos[idcs_airfoil, 2].min() - pad_z, pos[idcs_airfoil, 2].max() + pad_z)
    axins.set_aspect("equal")
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title("zoom: airfoil region", fontsize=8)
    mark_inset(axes[1], axins, loc1=2, loc2=3, fc="none", ec="black", lw=0.6, alpha=0.7)

    fig.suptitle(
        f"Local mesh resolution and the model voxel grid — sample {Path(rep).name}",
        fontsize=11,
    )
    out = OUT / out_name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Wrote", out)


def main():
    for rep, name in [
        (f"{DATA}/1094_10-2.npz", "voxel_and_density_1af.png"),
        (f"{DATA}/1021_10-0.npz", "voxel_and_density_2af.png"),
        (f"{DATA}/1021_1-0.npz",  "voxel_and_density_3af.png"),
    ]:
        render(rep, name)


if __name__ == "__main__":
    main()
