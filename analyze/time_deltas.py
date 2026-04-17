"""Analyze temporal coordinates across samples.

Questions:
  a) Are time deltas within a sample equidistant (input and output separately)?
  b) Are time deltas the same across samples?
  c) Are the absolute time values identical across samples?
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")

files = sorted(DATA_DIR.glob("*.npz"))
print(f"Found {len(files)} samples\n")

all_t = []  # (num_samples, 10)

for f in files:
    data = np.load(f)
    all_t.append(data["t"])  # (10,)

all_t = np.array(all_t)  # (num_samples, 10)

# --- (c) Are absolute times the same across samples? ---
print("=== (c) Absolute time values ===")
t_min = all_t.min(axis=0)
t_max = all_t.max(axis=0)
t_mean = all_t.mean(axis=0)
print(f"{'Frame':<8} {'Min':<14} {'Max':<14} {'Mean':<14} {'Spread (max-min)':<14}")
print("-" * 64)
for i in range(10):
    label = f"in_{i}" if i < 5 else f"out_{i-5}"
    print(f"{label:<8} {t_min[i]:<14.6f} {t_max[i]:<14.6f} {t_mean[i]:<14.6f} {t_max[i]-t_min[i]:<14.6e}")

# Check if all samples share the exact same t vector
all_identical = np.all(all_t == all_t[0])
print(f"\nAll samples have identical t vectors: {all_identical}")
if not all_identical:
    num_unique = len(np.unique(all_t, axis=0))
    print(f"Number of unique t vectors: {num_unique}")

# --- (a) Equidistant within samples? ---
print("\n=== (a) Equidistant time deltas within samples ===")

deltas_in = np.diff(all_t[:, :5], axis=1)   # (num_samples, 4)
deltas_out = np.diff(all_t[:, 5:], axis=1)   # (num_samples, 4)
delta_gap = all_t[:, 5] - all_t[:, 4]        # gap between last input and first output

print("\nInput deltas (t[i+1] - t[i] for i in 0..3):")
print(f"{'Pair':<10} {'Min':<14} {'Max':<14} {'Mean':<14} {'Std':<14}")
print("-" * 66)
for i in range(4):
    print(f"t{i}→t{i+1}     {deltas_in[:,i].min():<14.6f} {deltas_in[:,i].max():<14.6f} "
          f"{deltas_in[:,i].mean():<14.6f} {deltas_in[:,i].std():<14.6e}")

print("\nOutput deltas (t[i+1] - t[i] for i in 5..8):")
print(f"{'Pair':<10} {'Min':<14} {'Max':<14} {'Mean':<14} {'Std':<14}")
print("-" * 66)
for i in range(4):
    print(f"t{i+5}→t{i+6}   {deltas_out[:,i].min():<14.6f} {deltas_out[:,i].max():<14.6f} "
          f"{deltas_out[:,i].mean():<14.6f} {deltas_out[:,i].std():<14.6e}")

print(f"\nInput→Output gap (t5 - t4):")
print(f"  Min: {delta_gap.min():.6f}  Max: {delta_gap.max():.6f}  "
      f"Mean: {delta_gap.mean():.6f}  Std: {delta_gap.std():.6e}")

# Check per-sample equidistance
all_deltas = np.diff(all_t, axis=1)  # (num_samples, 9)
per_sample_range = all_deltas.max(axis=1) - all_deltas.min(axis=1)
print(f"\nPer-sample max spread of all 9 deltas: "
      f"min={per_sample_range.min():.6e}, max={per_sample_range.max():.6e}")
equidistant_count = np.sum(per_sample_range < 1e-10)
print(f"Samples with perfectly equidistant frames: {equidistant_count}/{len(files)}")

# --- (b) Same deltas across samples? ---
print("\n=== (b) Same time deltas across samples ===")
print(f"{'Delta':<10} {'# unique values':<18}")
print("-" * 28)
for i in range(9):
    n_unique = len(np.unique(np.round(all_deltas[:, i], decimals=10)))
    print(f"d{i}→{i+1}      {n_unique}")

# Show a few example t vectors
print("\n=== Example t vectors (first 5 samples) ===")
for i in range(min(5, len(files))):
    print(f"  {files[i].name}: {all_t[i]}")

# --- (d) Do time windows align across geometries by slice index? ---
import re
print("\n=== (d) Time windows by slice index across geometries ===")

# Group files by geometry (prefix before the last -N)
geo_slices = {}  # geometry -> {slice_idx: t_vector}
for f, t in zip(files, all_t):
    m = re.match(r"(.+)-(\d+)\.npz$", f.name)
    if m:
        geo, sl = m.group(1), int(m.group(2))
        geo_slices.setdefault(geo, {})[sl] = t

# For each slice index, collect the t[0] (start time) across all geometries
slice_indices = sorted({sl for slices in geo_slices.values() for sl in slices})
print(f"\nSlice indices found: {slice_indices}")
print(f"Number of geometries: {len(geo_slices)}")

for sl in slice_indices:
    starts = [slices[sl][0] for slices in geo_slices.values() if sl in slices]
    starts = np.array(starts)
    all_same = np.all(np.abs(starts - starts[0]) < 1e-8)
    print(f"\n  Slice -{sl}: {len(starts)} geometries"
          f"  |  t_start: min={starts.min():.6f} max={starts.max():.6f} "
          f"spread={starts.max()-starts.min():.6e}"
          f"  |  All identical: {all_same}")

# Check if within each geometry, the 5 slices are contiguous or overlapping
print("\n=== Slice arrangement within geometries (first 10) ===")
for geo in sorted(geo_slices)[:10]:
    slices = geo_slices[geo]
    parts = []
    for sl in sorted(slices):
        t = slices[sl]
        parts.append(f"-{sl}: [{t[0]:.3f}..{t[-1]:.3f}]")
    print(f"  {geo}: {', '.join(parts)}")
