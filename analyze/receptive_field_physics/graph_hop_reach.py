"""How far does an L-hop graph network reach on the actual point cloud?

For each of several samples (1 / 2 / 3 airfoil), build a k-NN graph in 3D
(k ∈ {8, 16, 32}), seed BFS from airfoil points, and measure the Euclidean
radius reached after L = 1, 2, 4, 8 hops. Express the result in chord lengths.

This translates the "8-layer graph net" description into a coverage statement.
"""
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

OUT = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/receptive_field_physics")
DATA = "/home/paul/scratch/gram-competition/warped-ifw"

K_VALUES = (8, 16, 32)
L_VALUES = (1, 2, 4, 8, 16)
N_SEEDS = 64  # seed BFS from this many random airfoil points


def hop_reach(pos: np.ndarray, seeds: np.ndarray, adj: csr_matrix, max_L: int) -> np.ndarray:
    """For each seed, return the max Euclidean distance reached after each L in 1..max_L."""
    n = pos.shape[0]
    n_seeds = len(seeds)
    # boolean (n_seeds, n) reachability — start with the seed bit
    reached = np.zeros((n_seeds, n), dtype=bool)
    for i, s in enumerate(seeds):
        reached[i, s] = True
    out = np.zeros((n_seeds, max_L))
    cur = reached.copy()
    for L in range(1, max_L + 1):
        # expand one hop: cur = cur OR (cur @ adj)
        # use sparse-matrix multiplication on float and re-threshold
        cur = (cur.astype(np.uint8) @ adj.astype(np.uint8)) > 0
        cur = cur | reached
        reached = cur
        for i, s in enumerate(seeds):
            pts = pos[reached[i]]
            d = np.linalg.norm(pts - pos[s], axis=1)
            out[i, L - 1] = d.max()
    return out  # (n_seeds, max_L)


def analyze_sample(path: str, ax):
    d = np.load(path)
    pos = d["pos"].astype(np.float32)
    af = d["idcs_airfoil"]
    chord_x = pos[af, 0].max() - pos[af, 0].min()
    tree = cKDTree(pos)

    rng = np.random.default_rng(0)
    seeds = rng.choice(af, size=N_SEEDS, replace=False)

    for k in K_VALUES:
        _, nn_idx = tree.query(pos, k=k + 1)  # includes self at column 0
        nn_idx = nn_idx[:, 1:]
        n = pos.shape[0]
        rows = np.repeat(np.arange(n), k)
        cols = nn_idx.reshape(-1)
        # symmetrize so messages can flow both ways (typical for GNNs)
        data = np.ones(rows.size, dtype=np.uint8)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        adj = adj.maximum(adj.T)

        reach = hop_reach(pos, seeds, adj, max_L=max(L_VALUES))
        # median, p25, p75 across seeds
        med = np.median(reach, axis=0)
        lo = np.percentile(reach, 25, axis=0)
        hi = np.percentile(reach, 75, axis=0)
        Ls = np.arange(1, reach.shape[1] + 1)
        ax.plot(Ls, med / chord_x, label=f"k={k}", marker="o")
        ax.fill_between(Ls, lo / chord_x, hi / chord_x, alpha=0.18)

    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="1 chord (target)")
    ax.axhline(1.15, color="tab:green", ls=":", lw=1, alpha=0.6, label="1.15 chord (p90 bbox)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("graph depth L (number of hops)")
    ax.set_ylabel("reachable radius / chord_x  (median ± IQR over seeds)")
    ax.set_title(f"{Path(path).name}\nchord_x = {chord_x:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    samples = [
        f"{DATA}/1094_10-2.npz",
        f"{DATA}/1021_10-0.npz",
        f"{DATA}/1021_1-0.npz",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, p in zip(axes, samples):
        print("analyzing", p)
        analyze_sample(p, ax)
    fig.suptitle(
        "L-hop reach of a k-NN graph network seeded from airfoil points "
        f"(BFS over symmetric kNN graph, {N_SEEDS} seeds per sample)",
        fontsize=11,
    )
    fig.savefig(OUT / "graph_hop_reach.png", dpi=140)
    plt.close(fig)
    print("Wrote", OUT / "graph_hop_reach.png")


if __name__ == "__main__":
    main()
