"""How does the 'receptive-field' answer depend on the percentile cutoff?

For each sample we sweep q ∈ {50, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9}
and recompute, on the var_resid signal:
  - bbox_x of {points with var_resid >= p_q}
  - bbox_x / airfoil_chord_x
  - downstream reach (= max x of high-var set - airfoil max x)

Then we aggregate across the 810 samples and plot ratio vs. q.
"""
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = "/home/paul/scratch/gram-competition/warped-ifw"
OUT = Path("/home/paul/gitrepos/iclr-2026-gram/analyze/receptive_field_physics")

Q_LEVELS = [50, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]


def main():
    paths = sorted(glob(f"{DATA}/*.npz"))
    print(f"{len(paths)} samples")

    ratios_chord = np.full((len(paths), len(Q_LEVELS)), np.nan)
    reaches_chord = np.full((len(paths), len(Q_LEVELS)), np.nan)
    extents_box = np.full((len(paths), len(Q_LEVELS)), np.nan)

    for i, p in enumerate(paths):
        d = np.load(p)
        pos = d["pos"]
        af = pos[d["idcs_airfoil"]]
        chord_x = af[:, 0].max() - af[:, 0].min()
        af_max_x = af[:, 0].max()
        dom_x = pos[:, 0].max() - pos[:, 0].min()
        v_in = d["velocity_in"]; v_out = d["velocity_out"]
        var_resid = (v_out - v_in[-1:]).var(0).sum(-1)
        thrs = np.percentile(var_resid, Q_LEVELS)
        for j, t in enumerate(thrs):
            mask = var_resid >= t
            if not mask.any():
                continue
            hv = pos[mask]
            bbox_x = hv[:, 0].max() - hv[:, 0].min()
            ratios_chord[i, j] = bbox_x / chord_x
            reaches_chord[i, j] = (hv[:, 0].max() - af_max_x) / chord_x
            extents_box[i, j] = bbox_x / dom_x

    # aggregate: median + IQR across samples per q
    def summarize(A):
        return (
            np.nanmedian(A, axis=0),
            np.nanpercentile(A, 25, axis=0),
            np.nanpercentile(A, 75, axis=0),
        )

    rc_m, rc_lo, rc_hi = summarize(ratios_chord)
    re_m, re_lo, re_hi = summarize(reaches_chord)
    eb_m, eb_lo, eb_hi = summarize(extents_box)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (m, lo, hi), title, ylab in [
        (axes[0], (rc_m, rc_lo, rc_hi),
         "high-var bbox_x  /  airfoil chord_x",
         "ratio  (1 = exactly chord-length)"),
        (axes[1], (re_m, re_lo, re_hi),
         "downstream reach past trailing edge  (in chords)",
         "chord lengths"),
        (axes[2], (eb_m, eb_lo, eb_hi),
         "high-var bbox_x  /  domain x-span",
         "fraction of streamwise domain"),
    ]:
        ax.plot(Q_LEVELS, m, "-o", color="tab:blue", label="median over 810 samples")
        ax.fill_between(Q_LEVELS, lo, hi, alpha=0.2, color="tab:blue", label="IQR (25–75 %)")
        ax.axvline(90, color="k", ls="--", alpha=0.3, lw=1)
        ax.set_xlabel("percentile cutoff q  (var_resid ≥ p_q)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("How the 'high-variance region size' depends on the percentile cutoff",
                 fontsize=11)
    fig.savefig(OUT / "cutoff_sensitivity.png", dpi=140)
    plt.close(fig)

    # print a table for the report
    print(f"{'q':>6} | {'bbox_x/chord (median)':>24} | {'reach/chord (median)':>22} | {'bbox_x/box (median)':>22}")
    for j, q in enumerate(Q_LEVELS):
        print(f"{q:>6.1f} | {rc_m[j]:>24.3f} | {re_m[j]:>22.3f} | {eb_m[j]:>22.3f}")
    np.savez(OUT / "cutoff_sensitivity.npz",
             q_levels=np.array(Q_LEVELS),
             bbox_per_chord=ratios_chord,
             reach_per_chord=reaches_chord,
             bbox_per_box=extents_box)


if __name__ == "__main__":
    main()
