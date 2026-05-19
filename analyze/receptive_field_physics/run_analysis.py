"""Characterize the spatial distribution of temporal velocity variance in the GRaM IFW dataset.

For each sample, compute:
  - per-point temporal variance var(v)   (sum over xyz components of the per-component variance)
  - per-point persistence-residual variance var(v_out - v_in[-1])
  - thresholds at p90/p95/p99 for each signal
  - axis-aligned bounding boxes of (airfoil) and (>p_q points) for each signal
  - IoU between the >p90 sets defined by var and var_resid

Stream samples one at a time — no large in-memory aggregates beyond the per-sample table and a
sub-sampled pool used for the global histograms.
"""
from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "/home/paul/scratch/gram-competition/warped-ifw"
OUT_DIR = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/receptive_field_physics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# x is streamwise (largest extent, freestream direction); z is the vertical/other in-plane axis;
# y is the thin spanwise direction. We render the side view as x-z. Verified against per-sample
# extents inside the loop.
STREAMWISE_AXIS = 0  # x
SIDEVIEW_AXES = (0, 2)  # (x, z)

SUBSAMPLE_PER_FILE_FOR_HIST = 5_000  # points per sample contributed to the global histograms
REPR_SAMPLE_TARGET = None  # picked at runtime as the sample whose p90 wake reach is closest to the median

FNAME_RE = re.compile(r"^(?P<sim>\d+)_(?P<cfg>\d+)-(?P<win>\d+)\.npz$")


def parse_id(path: str) -> tuple[str, str, str]:
    m = FNAME_RE.match(Path(path).name)
    return m["sim"], m["cfg"], m["win"]


def axis_extent(pts: np.ndarray, axis: int) -> float:
    if pts.size == 0:
        return float("nan")
    return float(pts[:, axis].max() - pts[:, axis].min())


def aabb(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pts.min(0), pts.max(0)


def analyze_sample(path: str) -> dict:
    d = np.load(path)
    pos = d["pos"].astype(np.float32)            # (N, 3)
    idcs_airfoil = d["idcs_airfoil"].astype(np.int64)
    v_in = d["velocity_in"].astype(np.float32)   # (5, N, 3)
    v_out = d["velocity_out"].astype(np.float32) # (5, N, 3)
    N = pos.shape[0]

    # signal 1: temporal variance of the full 10-frame velocity sequence
    v_full = np.concatenate([v_in, v_out], axis=0)        # (10, N, 3)
    var = v_full.var(axis=0).sum(axis=-1)                 # (N,)

    # signal 2: variance of the persistence residual (network's actual learning target spread)
    resid = v_out - v_in[-1:]                             # (5, N, 3)
    var_resid = resid.var(axis=0).sum(axis=-1)            # (N,)

    # thresholds
    p_levels = [90, 95, 99]
    var_pcts = np.percentile(var, p_levels)
    var_resid_pcts = np.percentile(var_resid, p_levels)

    # masks at p90 for both signals
    var_p90_mask = var >= var_pcts[0]
    vr_p90_mask = var_resid >= var_resid_pcts[0]

    # IoU between the two p90 high-variance sets
    inter = np.logical_and(var_p90_mask, vr_p90_mask).sum()
    union = np.logical_or(var_p90_mask, vr_p90_mask).sum()
    iou_p90 = float(inter) / float(union) if union > 0 else float("nan")

    # geometry: airfoil
    af_pos = pos[idcs_airfoil]
    af_min, af_max = aabb(af_pos)
    af_chord_x = af_max[0] - af_min[0]
    af_chord_z = af_max[2] - af_min[2]
    af_centroid_x = float(af_pos[:, 0].mean())

    # domain bounding box (streamwise span)
    dom_min, dom_max = aabb(pos)
    dom_x_span = dom_max[0] - dom_min[0]

    def high_var_geom(mask: np.ndarray) -> dict:
        hv = pos[mask]
        if hv.shape[0] == 0:
            return {
                "highvar_bbox_x": float("nan"),
                "highvar_bbox_z": float("nan"),
                "highvar_x_extent_per_box_x": float("nan"),
                "highvar_x_extent_per_chord": float("nan"),
                "highvar_downstream_reach": float("nan"),
            }
            return
        bbox_x = float(hv[:, 0].max() - hv[:, 0].min())
        bbox_z = float(hv[:, 2].max() - hv[:, 2].min())
        return {
            "highvar_bbox_x": bbox_x,
            "highvar_bbox_z": bbox_z,
            "highvar_x_extent_per_box_x": bbox_x / dom_x_span,
            "highvar_x_extent_per_chord": bbox_x / af_chord_x,
            "highvar_downstream_reach": float(hv[:, 0].max() - af_max[0]),
        }

    var_geom = high_var_geom(var_p90_mask)
    vr_geom = high_var_geom(vr_p90_mask)

    sim, cfg, win = parse_id(path)
    row = {
        "file": Path(path).name,
        "geometry_id": f"{sim}_{cfg}",
        "sim": sim,
        "window": win,
        "n_airfoil": int(idcs_airfoil.size),
        "domain_x_span": float(dom_x_span),
        "domain_y_span": float(dom_max[1] - dom_min[1]),
        "domain_z_span": float(dom_max[2] - dom_min[2]),
        "airfoil_chord_x": float(af_chord_x),
        "airfoil_chord_z": float(af_chord_z),
        "airfoil_centroid_x": af_centroid_x,
        "airfoil_max_x": float(af_max[0]),
        "var_p90": float(var_pcts[0]),
        "var_p95": float(var_pcts[1]),
        "var_p99": float(var_pcts[2]),
        "var_resid_p90": float(var_resid_pcts[0]),
        "var_resid_p95": float(var_resid_pcts[1]),
        "var_resid_p99": float(var_resid_pcts[2]),
        "agreement_iou_p90": iou_p90,
        # var signal geometry
        "var_highvar_bbox_x": var_geom["highvar_bbox_x"],
        "var_highvar_bbox_z": var_geom["highvar_bbox_z"],
        "var_highvar_x_extent_per_box_x": var_geom["highvar_x_extent_per_box_x"],
        "var_highvar_x_extent_per_chord": var_geom["highvar_x_extent_per_chord"],
        "var_highvar_downstream_reach": var_geom["highvar_downstream_reach"],
        # var_resid signal geometry
        "vr_highvar_bbox_x": vr_geom["highvar_bbox_x"],
        "vr_highvar_bbox_z": vr_geom["highvar_bbox_z"],
        "vr_highvar_x_extent_per_box_x": vr_geom["highvar_x_extent_per_box_x"],
        "vr_highvar_x_extent_per_chord": vr_geom["highvar_x_extent_per_chord"],
        "vr_highvar_downstream_reach": vr_geom["highvar_downstream_reach"],
    }

    # sub-sample for the global histograms
    rng = np.random.default_rng(int(sim) * 1000 + int(cfg) * 10 + int(win))
    sub_idx = rng.choice(N, size=min(SUBSAMPLE_PER_FILE_FOR_HIST, N), replace=False)
    sub_var = var[sub_idx]
    sub_vr = var_resid[sub_idx]

    return row, sub_var, sub_vr, (pos, idcs_airfoil, var, var_resid, var_pcts, var_resid_pcts, af_max)


def main():
    paths = sorted(glob(f"{DATA_DIR}/*.npz"))
    print(f"Found {len(paths)} samples")

    rows = []
    hist_var = []
    hist_vr = []
    # cache one piece of data per sim so we can pick a representative geometry afterwards
    # but to avoid memory blow-up we'll do a 2-pass: first compute table, then re-load one rep
    for i, p in enumerate(paths):
        row, sv, svr, _ = analyze_sample(p)
        rows.append(row)
        hist_var.append(sv)
        hist_vr.append(svr)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(paths)}] {row['file']}  iou_p90={row['agreement_iou_p90']:.3f}  "
                  f"vr_reach={row['vr_highvar_downstream_reach']:.3f}")

    df = pd.DataFrame(rows)

    # Headline ratios (on var_resid)
    headline = {k: (float(v) if hasattr(v, "item") else v) for k, v in {
        "vr_x_extent_per_box_x_mean": df["vr_highvar_x_extent_per_box_x"].mean(),
        "vr_x_extent_per_box_x_std":  df["vr_highvar_x_extent_per_box_x"].std(),
        "vr_x_extent_per_chord_mean": df["vr_highvar_x_extent_per_chord"].mean(),
        "vr_x_extent_per_chord_std":  df["vr_highvar_x_extent_per_chord"].std(),
        "vr_downstream_reach_mean":   df["vr_highvar_downstream_reach"].mean(),
        "vr_downstream_reach_std":    df["vr_highvar_downstream_reach"].std(),
        "vr_downstream_reach_in_chords_mean": (df["vr_highvar_downstream_reach"] / df["airfoil_chord_x"]).mean(),
        "vr_downstream_reach_in_chords_std":  (df["vr_highvar_downstream_reach"] / df["airfoil_chord_x"]).std(),
        "iou_p90_mean": df["agreement_iou_p90"].mean(),
        "iou_p90_std":  df["agreement_iou_p90"].std(),
        "n_samples": int(len(df)),
        "n_distinct_sims": int(df["sim"].nunique()),
        "n_distinct_geoms": int(df["geometry_id"].nunique()),
    }.items()}
    print("Headline:")
    print(json.dumps(headline, indent=2))

    # Save tables
    df.to_csv(OUT_DIR / "summary_table.csv", index=False)
    # markdown summary table (full file is big; provide a compact view + per-row file)
    md_cols = [
        "geometry_id", "window", "airfoil_chord_x", "airfoil_centroid_x",
        "var_resid_p90", "var_resid_p99",
        "vr_highvar_bbox_x", "vr_highvar_x_extent_per_box_x",
        "vr_highvar_x_extent_per_chord", "vr_highvar_downstream_reach",
        "agreement_iou_p90",
    ]
    md_view = df[md_cols].copy()
    for c in md_view.columns:
        if md_view[c].dtype.kind == "f":
            md_view[c] = md_view[c].map(lambda x: f"{x:.4f}")
    (OUT_DIR / "summary_table.md").write_text(md_view.to_markdown(index=False))

    # Histograms (log y), CDFs
    hist_var = np.concatenate(hist_var)
    hist_vr = np.concatenate(hist_vr)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, data, label, color in [
        (axes[0, 0], hist_var, "var(v)", "tab:blue"),
        (axes[0, 1], hist_vr, "var(v_out − v_in[-1])", "tab:orange"),
    ]:
        # use log-spaced bins because the values span many decades
        positive = data[data > 0]
        bins = np.logspace(np.log10(max(positive.min(), 1e-10)), np.log10(positive.max()), 80)
        ax.hist(data, bins=bins, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.set_title(f"histogram — {label}")
    for ax, data, label, color in [
        (axes[1, 0], hist_var, "var(v)", "tab:blue"),
        (axes[1, 1], hist_vr, "var(v_out − v_in[-1])", "tab:orange"),
    ]:
        s = np.sort(data)
        cdf = np.arange(1, s.size + 1) / s.size
        ax.plot(s, cdf, color=color)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF — {label}")
        for q in (0.90, 0.95, 0.99):
            ax.axhline(q, color="k", lw=0.5, ls="--", alpha=0.4)
    fig.suptitle(
        f"Variance distributions across {len(paths)} samples "
        f"(sub-sampled {SUBSAMPLE_PER_FILE_FOR_HIST} pts / sample)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variance_histograms.png", dpi=140)
    plt.close(fig)

    # Pick a representative sample: vr_downstream_reach closest to the median
    med = df["vr_highvar_downstream_reach"].median()
    rep_idx = (df["vr_highvar_downstream_reach"] - med).abs().idxmin()
    rep_path = f"{DATA_DIR}/{df.iloc[rep_idx]['file']}"
    print(f"Representative sample: {rep_path}  (vr_reach={df.iloc[rep_idx]['vr_highvar_downstream_reach']:.3f}, median={med:.3f})")

    # Side-view figure
    _, _, _, payload = analyze_sample(rep_path)
    pos, idcs_airfoil, var, var_resid, var_pcts, var_resid_pcts, af_max = payload
    mask = var_resid >= var_resid_pcts[0]
    ax_idx_x, ax_idx_z = SIDEVIEW_AXES

    fig, ax = plt.subplots(figsize=(11, 6))
    # all points in light grey
    ax.scatter(pos[:, ax_idx_x], pos[:, ax_idx_z], s=1, c="lightgrey", alpha=0.3, linewidths=0, label="all points")
    # airfoil in red
    ax.scatter(pos[idcs_airfoil, ax_idx_x], pos[idcs_airfoil, ax_idx_z],
               s=2, c="red", alpha=0.8, linewidths=0, label="airfoil")
    # high var_resid in blue, alpha ~ var_resid
    if mask.any():
        vr_hi = var_resid[mask]
        a = vr_hi / vr_hi.max()
        ax.scatter(pos[mask, ax_idx_x], pos[mask, ax_idx_z],
                   s=3, c="tab:blue", alpha=np.clip(0.05 + 0.85 * a, 0.05, 0.95),
                   linewidths=0, label="var_resid ≥ p90")
    # bbox overlays
    dmin, dmax = pos.min(0), pos.max(0)
    ax.plot([dmin[ax_idx_x], dmax[ax_idx_x], dmax[ax_idx_x], dmin[ax_idx_x], dmin[ax_idx_x]],
            [dmin[ax_idx_z], dmin[ax_idx_z], dmax[ax_idx_z], dmax[ax_idx_z], dmin[ax_idx_z]],
            "k-", lw=1, alpha=0.6, label="domain bbox")
    if mask.any():
        hv = pos[mask]
        hvmin, hvmax = hv.min(0), hv.max(0)
        ax.plot([hvmin[ax_idx_x], hvmax[ax_idx_x], hvmax[ax_idx_x], hvmin[ax_idx_x], hvmin[ax_idx_x]],
                [hvmin[ax_idx_z], hvmin[ax_idx_z], hvmax[ax_idx_z], hvmax[ax_idx_z], hvmin[ax_idx_z]],
                color="tab:blue", lw=1.5, ls="--", alpha=0.9, label="var_resid≥p90 bbox")
    ax.set_xlabel("x  (streamwise)")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title(f"Side view (x-z) — sample {Path(rep_path).name}\n"
                 f"var_resid p90={var_resid_pcts[0]:.3g}, "
                 f"downstream reach={df.iloc[rep_idx]['vr_highvar_downstream_reach']:.2f} "
                 f"(= {df.iloc[rep_idx]['vr_highvar_downstream_reach']/df.iloc[rep_idx]['airfoil_chord_x']:.2f} chords)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sideview_representative.png", dpi=160)
    plt.close(fig)

    # Save headline json
    (OUT_DIR / "headline.json").write_text(json.dumps(headline, indent=2))
    print("Wrote outputs to", OUT_DIR)
    return df, headline, rep_path


if __name__ == "__main__":
    main()
