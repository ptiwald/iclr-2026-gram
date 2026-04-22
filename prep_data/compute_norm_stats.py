"""Compute normalization statistics from the training split and save them.

Writes a state_dict-compatible file with:
  pos_mean, pos_scale:  (3,) — position center and half-range per axis
  vel_mean, vel_std:    (3,) — per-component mean/std pooled over all points,
                               timesteps, and samples (velocity_in + velocity_out)

Loaded by ABUPT.__init__ into buffers. Run once before training:
    python prep_data/compute_norm_stats.py --data_dir <path> --out_path <path>
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/paul/scratch/gram-competition/warped-ifw/")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--out_path", default="models/ab_upt/norm_stats.pt")
    args = parser.parse_args()

    with open(args.split_file) as f:
        split = json.load(f)
    train_files = [os.path.join(args.data_dir, fn) for fn in split["train"]]
    print(f"computing stats over {len(train_files)} training samples")

    pos_min = np.full(3, np.inf)
    pos_max = np.full(3, -np.inf)

    vel_sum = np.zeros(3, dtype=np.float64)
    vel_sumsq = np.zeros(3, dtype=np.float64)
    vel_count = 0

    for i, path in enumerate(train_files):
        d = np.load(path)
        pos = d["pos"]
        vel_in = d["velocity_in"].reshape(-1, 3).astype(np.float64)
        vel_out = d["velocity_out"].reshape(-1, 3).astype(np.float64)

        pos_min = np.minimum(pos_min, pos.min(axis=0))
        pos_max = np.maximum(pos_max, pos.max(axis=0))

        vel_sum += vel_in.sum(axis=0) + vel_out.sum(axis=0)
        vel_sumsq += (vel_in ** 2).sum(axis=0) + (vel_out ** 2).sum(axis=0)
        vel_count += vel_in.shape[0] + vel_out.shape[0]

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(train_files)}")

    vel_mean = vel_sum / vel_count
    vel_std = np.sqrt(vel_sumsq / vel_count - vel_mean ** 2)

    pos_mean = 0.5 * (pos_min + pos_max)
    pos_scale = 0.5 * (pos_max - pos_min)

    print(f"pos_mean  : {pos_mean}")
    print(f"pos_scale : {pos_scale}")
    print(f"vel_mean  : {vel_mean}")
    print(f"vel_std   : {vel_std}")

    stats = {
        "pos_mean": torch.from_numpy(pos_mean).float(),
        "pos_scale": torch.from_numpy(pos_scale).float(),
        "vel_mean": torch.from_numpy(vel_mean).float(),
        "vel_std": torch.from_numpy(vel_std).float(),
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(stats, args.out_path)
    print(f"saved to {args.out_path}")


if __name__ == "__main__":
    main()
