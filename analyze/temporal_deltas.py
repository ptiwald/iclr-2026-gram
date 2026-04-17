"""Compute the mean absolute delta between consecutive input timesteps
and its variance across samples.

For each sample and each pair of consecutive input frames (t, t+1),
we compute the mean absolute difference across all spatial points and
velocity components. This gives one scalar per (sample, frame-pair).

We then report:
  - The grand mean of that quantity (averaged over samples) per frame-pair
  - The variance across samples per frame-pair
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")

files = sorted(DATA_DIR.glob("*.npz"))
print(f"Found {len(files)} samples")

num_pairs = 4  # 5 input frames → 4 consecutive pairs
# Collect per-sample, per-pair mean absolute deltas
all_deltas = []  # shape will be (num_samples, num_pairs)

for f in files:
    data = np.load(f)
    vel_in = data["velocity_in"]  # (5, 100000, 3)

    # Mean absolute delta across points and velocity components per pair
    sample_deltas = []
    for i in range(num_pairs):
        delta = np.abs(vel_in[i + 1] - vel_in[i]).mean()
        sample_deltas.append(delta)
    all_deltas.append(sample_deltas)

all_deltas = np.array(all_deltas)  # (num_samples, num_pairs)

print("\n=== Mean absolute delta between consecutive input frames ===")
print(f"{'Pair':<12} {'Mean (across samples)':<24} {'Var (across samples)':<24}")
print("-" * 60)
for i in range(num_pairs):
    mean_val = all_deltas[:, i].mean()
    var_val = all_deltas[:, i].var()
    print(f"t{i}→t{i+1}       {mean_val:<24.6f} {var_val:<24.6f}")

print(f"\n{'Overall':<12} {all_deltas.mean():<24.6f} {all_deltas.var():<24.6f}")
