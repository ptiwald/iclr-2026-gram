"""Count airfoils per sample by spatial clustering of idcs_airfoil.

Each airfoil is a connected surface in 3D, well separated from other airfoils.
We build a radius graph (eps=0.02) on the airfoil surface points and count
connected components with >= 5 members (drop tiny noise components).

Outputs:
- analyze/airfoil_counts.csv: filename, n_airfoils, n_airfoil_points
- prints class distribution
"""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")
OUT_CSV = Path(__file__).resolve().parent / "airfoil_counts.csv"
EPS = 0.02
MIN_SIZE = 5


def count_airfoils(pos: np.ndarray, idcs: np.ndarray) -> int:
    pts = pos[idcs]
    if len(pts) < MIN_SIZE:
        return 1
    tree = cKDTree(pts)
    pairs = tree.query_pairs(EPS, output_type="ndarray")
    n = len(pts)
    if len(pairs) == 0:
        return n
    A = csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    A = A + A.T
    _, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    return int((sizes >= MIN_SIZE).sum())


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Found {len(files)} files\n")

    rows = []
    counts = {1: 0, 2: 0, 3: 0}
    other = 0
    for i, f in enumerate(files):
        d = np.load(f)
        n = count_airfoils(d["pos"], d["idcs_airfoil"])
        rows.append((f.name, n, int(d["idcs_airfoil"].shape[0])))
        if n in counts:
            counts[n] += 1
        else:
            other += 1
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(files)}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "n_airfoils", "n_airfoil_points"])
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV}")
    print(f"Class distribution: 1af={counts[1]}  2af={counts[2]}  3af={counts[3]}  other={other}")


if __name__ == "__main__":
    main()
