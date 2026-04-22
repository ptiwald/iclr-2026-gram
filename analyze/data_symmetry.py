"""Inspect spatial layout + freestream direction to decide the augmentation axis.

For each sample we want to answer:
  - Which axis is streamwise (= highest-magnitude mean freestream component)?
  - Which axis is spanwise (the candidate mirror-flip axis)?
  - Is the distribution of airfoil placements symmetric about the span axis?

Samples come in 1-, 2-, and 3-object variants; all stats are reported per group
(object count is recovered by clustering `idcs_airfoil` points spatially, since
the filename doesn't encode it).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")
OUT_DIR = Path("scratch/data_symmetry")


def _cluster_labels(airfoil_pos: np.ndarray) -> np.ndarray:
    """Label connected components of airfoil surface points.

    Threshold = 2 × 99th-percentile of nearest-neighbor distances. The 99th
    percentile captures the largest within-surface gap (trailing edges, sparse
    subsampled regions); doubling it gives margin without merging distinct
    airfoils that sit centimeters apart.
    """
    n = len(airfoil_pos)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    tree = cKDTree(airfoil_pos)
    dists, _ = tree.query(airfoil_pos, k=2)
    thr = np.quantile(dists[:, 1], 0.99) * 2.0
    pairs = tree.query_pairs(r=thr, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    graph = csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def cluster_airfoils(airfoil_pos: np.ndarray) -> int:
    labels = _cluster_labels(airfoil_pos)
    # Drop spurious tiny components (clustering noise); require ≥20 points per foil.
    counts = np.bincount(labels)
    return int((counts >= 20).sum())


def airfoil_centers(airfoil_pos: np.ndarray, n_comp: int) -> np.ndarray:
    """Return centroids of the substantive clusters (≥20 points)."""
    if len(airfoil_pos) < 2:
        return airfoil_pos.mean(axis=0, keepdims=True)
    labels = _cluster_labels(airfoil_pos)
    counts = np.bincount(labels)
    keep = np.where(counts >= 20)[0]
    if len(keep) == 0:
        return airfoil_pos.mean(axis=0, keepdims=True)
    centers = np.stack([airfoil_pos[labels == k].mean(axis=0) for k in keep])
    return centers


def freestream_from_farfield(pos: np.ndarray, airfoil_pos: np.ndarray,
                             velocity_in: np.ndarray) -> np.ndarray:
    """Estimate freestream as the mean velocity over points far from any airfoil.

    Uses points whose distance to the nearest airfoil point is in the top 20%.
    Averaged over the 5 input timesteps.
    """
    tree = cKDTree(airfoil_pos)
    d, _ = tree.query(pos, k=1)
    thr = np.quantile(d, 0.8)
    far_mask = d >= thr
    return velocity_in[:, far_mask].mean(axis=(0, 1))  # (3,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    if args.num_samples < len(files):
        files = list(rng.choice(files, size=args.num_samples, replace=False))
    print(f"Analyzing {len(files)} samples from {DATA_DIR}")

    per_sample = []
    for f in files:
        d = np.load(f)
        pos = d["pos"].astype(np.float32)
        idcs = d["idcs_airfoil"]
        vel = d["velocity_in"].astype(np.float32)
        airfoil_pos = pos[idcs]
        n_obj = cluster_airfoils(airfoil_pos)
        centers = airfoil_centers(airfoil_pos, n_obj)
        freestream = freestream_from_farfield(pos, airfoil_pos, vel)
        per_sample.append({
            "name": f.name,
            "n_obj": n_obj,
            "pos_min": pos.min(axis=0),
            "pos_max": pos.max(axis=0),
            "airfoil_bbox_min": airfoil_pos.min(axis=0),
            "airfoil_bbox_max": airfoil_pos.max(axis=0),
            "centers": centers,  # (n_obj, 3)
            "freestream": freestream,  # (3,)
        })
    print()

    axes = ["x", "y", "z"]

    # ----- Global: freestream direction and magnitude -----
    freestream_all = np.stack([s["freestream"] for s in per_sample])  # (N, 3)
    fs_mean = freestream_all.mean(axis=0)
    fs_std = freestream_all.std(axis=0)
    fs_mag = np.linalg.norm(fs_mean)
    print("=== Freestream (avg over samples) ===")
    for a, m, s in zip(axes, fs_mean, fs_std):
        print(f"  {a}:  mean = {m:+.3f}   std = {s:.3f}")
    print(f"  |mean| = {fs_mag:.3f}")
    stream_axis = int(np.argmax(np.abs(fs_mean)))
    print(f"  -> streamwise axis = '{axes[stream_axis]}'  (largest |mean| component)\n")

    # ----- Per-axis pos extent -----
    pos_min = np.stack([s["pos_min"] for s in per_sample]).mean(axis=0)
    pos_max = np.stack([s["pos_max"] for s in per_sample]).mean(axis=0)
    print("=== Domain extent (avg over samples) ===")
    for a, mn, mx in zip(axes, pos_min, pos_max):
        print(f"  {a}: [{mn:+.3f}, {mx:+.3f}]   center = {(mn+mx)/2:+.3f}   span = {mx-mn:.3f}")
    print()

    # ----- Group by object count -----
    groups = {k: [s for s in per_sample if s["n_obj"] == k] for k in (1, 2, 3)}
    other = [s for s in per_sample if s["n_obj"] not in (1, 2, 3)]
    print("=== Samples by inferred object count ===")
    for k in (1, 2, 3):
        print(f"  n_obj = {k}:  {len(groups[k])} samples")
    if other:
        print(f"  other (clustering artifact?): {len(other)} samples")
        ex = other[:3]
        for s in ex:
            print(f"    {s['name']}  n_obj={s['n_obj']}")
    print()

    # ----- Per-group airfoil-center distributions, per axis -----
    for k in (1, 2, 3):
        if not groups[k]:
            continue
        all_centers = np.concatenate([s["centers"] for s in groups[k]], axis=0)  # (K, 3)
        print(f"=== n_obj = {k}  |  {len(groups[k])} samples, {len(all_centers)} centers ===")
        for ax in range(3):
            c = all_centers[:, ax]
            # Symmetry test: compare mean to 0 (after centering on domain center).
            dom_center = (pos_min[ax] + pos_max[ax]) / 2
            c_rel = c - dom_center
            # Crude bimodal detection: split by sign of relative position; balanced if ~50/50.
            pos_frac = float((c_rel > 0).mean())
            # Reflection-symmetry quality: |mean| / std (should be small if centered).
            skew_proxy = abs(c_rel.mean()) / (c_rel.std() + 1e-6)
            print(f"  {axes[ax]}: center range [{c.min():+.3f}, {c.max():+.3f}]  "
                  f"rel-mean={c_rel.mean():+.3f}  rel-std={c_rel.std():.3f}  "
                  f"frac>center={pos_frac:.2f}  |mean|/std={skew_proxy:.3f}")
        print()

    # ----- Verdict -----
    # Spanwise axis: among non-streamwise axes, the one whose domain is symmetric
    # about 0 AND whose mean freestream component is near zero. Both must hold for
    # a mirror flip to preserve the physics.
    all_centers_global = np.concatenate([s["centers"] for s in per_sample], axis=0)
    non_stream = [a for a in range(3) if a != stream_axis]

    def span_score(a: int) -> float:
        # Lower = more spanwise-like.
        dom_skew = abs((pos_min[a] + pos_max[a]) / 2) / (pos_max[a] - pos_min[a])
        fs_ratio = abs(fs_mean[a]) / (abs(fs_mean[stream_axis]) + 1e-6)
        return dom_skew + fs_ratio

    span_axis = min(non_stream, key=span_score)
    vert_axis = [a for a in range(3) if a not in (stream_axis, span_axis)][0]
    print("=== Verdict ===")
    print(f"  streamwise axis:  '{axes[stream_axis]}'  (DO NOT flip; breaks physics)")
    print(f"  spanwise  axis:  '{axes[span_axis]}'   (mirror-flip CANDIDATE)")
    print(f"  vertical  axis:  '{axes[vert_axis]}'   (typically asymmetric ground/ceiling)")
    print()
    # Reflection symmetry across span:
    dom_center_span = (pos_min[span_axis] + pos_max[span_axis]) / 2
    c_span = all_centers_global[:, span_axis] - dom_center_span
    skew_span = abs(c_span.mean()) / (c_span.std() + 1e-6)
    print(f"  span-axis airfoil center symmetry (all groups combined):")
    print(f"    |mean|/std = {skew_span:.3f}  (≪1 → symmetric placement distribution)")
    fs_span = freestream_all[:, span_axis]
    print(f"  span-axis freestream component:  mean={fs_span.mean():+.3f}  std={fs_span.std():.3f}")
    print(f"    (should be ~0 for physical span-flip to hold)")
    print()

    # Save raw per-sample stats for plotting later.
    out_path = OUT_DIR / "per_sample.npz"
    np.savez(
        out_path,
        names=np.array([s["name"] for s in per_sample]),
        n_obj=np.array([s["n_obj"] for s in per_sample]),
        pos_min=np.stack([s["pos_min"] for s in per_sample]),
        pos_max=np.stack([s["pos_max"] for s in per_sample]),
        airfoil_bbox_min=np.stack([s["airfoil_bbox_min"] for s in per_sample]),
        airfoil_bbox_max=np.stack([s["airfoil_bbox_max"] for s in per_sample]),
        freestream=freestream_all,
        # centers are ragged; store as concatenated + offsets
        centers=np.concatenate([s["centers"] for s in per_sample], axis=0),
        centers_offsets=np.cumsum([0] + [len(s["centers"]) for s in per_sample]),
    )
    print(f"Saved per-sample stats to {out_path}")


if __name__ == "__main__":
    main()
