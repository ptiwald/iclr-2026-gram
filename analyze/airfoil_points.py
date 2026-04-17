"""Understand what idcs_airfoil points represent.

1. Are velocities at airfoil indices ~0? (no-slip → surface points)
2. What's the spatial structure? (surface shell vs filled volume)
3. Where are they relative to the flow?
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")

files = sorted(DATA_DIR.glob("*.npz"))

# --- 1. Velocity magnitudes at airfoil vs non-airfoil points ---
print("=== Velocity magnitudes at airfoil vs non-airfoil points ===\n")

airfoil_vel_mags = []
non_airfoil_vel_mags = []

for f in files[:20]:  # sample 20 files for speed
    data = np.load(f)
    idcs = data["idcs_airfoil"]
    vel_in = data["velocity_in"]   # (5, 100000, 3)
    vel_out = data["velocity_out"]  # (5, 100000, 3)
    vel_all = np.concatenate([vel_in, vel_out], axis=0)  # (10, 100000, 3)

    mask = np.zeros(vel_all.shape[1], dtype=bool)
    mask[idcs] = True

    mag = np.linalg.norm(vel_all, axis=2)  # (10, 100000)
    airfoil_vel_mags.append(mag[:, mask].ravel())
    non_airfoil_vel_mags.append(mag[:, ~mask].ravel())

airfoil_vel = np.concatenate(airfoil_vel_mags)
non_airfoil_vel = np.concatenate(non_airfoil_vel_mags)

print(f"Airfoil points:     mean |v|={airfoil_vel.mean():.6f}  "
      f"std={airfoil_vel.std():.6f}  max={airfoil_vel.max():.6f}  "
      f"fraction |v|<1e-6: {(airfoil_vel < 1e-6).mean():.4f}")
print(f"Non-airfoil points: mean |v|={non_airfoil_vel.mean():.6f}  "
      f"std={non_airfoil_vel.std():.6f}  max={non_airfoil_vel.max():.6f}  "
      f"fraction |v|<1e-6: {(non_airfoil_vel < 1e-6).mean():.4f}")

print(f"\nAirfoil |v| percentiles: "
      f"p50={np.median(airfoil_vel):.6f}  "
      f"p95={np.percentile(airfoil_vel, 95):.6f}  "
      f"p99={np.percentile(airfoil_vel, 99):.6f}")

# --- 2. Spatial structure of airfoil points ---
print("\n=== Spatial structure of airfoil points (1 sample) ===\n")

data = np.load(files[0])
idcs = data["idcs_airfoil"]
pos = data["pos"]  # (100000, 3)

af_pos = pos[idcs]
other_pos = pos[~np.isin(np.arange(len(pos)), idcs)]

print(f"Sample: {files[0].name}")
print(f"Airfoil points: {len(idcs)}, Non-airfoil: {len(pos) - len(idcs)}")

for dim, label in enumerate(["x", "y", "z"]):
    print(f"\n  {label}-axis:")
    print(f"    Airfoil:     min={af_pos[:,dim].min():.4f}  max={af_pos[:,dim].max():.4f}  "
          f"mean={af_pos[:,dim].mean():.4f}  std={af_pos[:,dim].std():.4f}")
    print(f"    Non-airfoil: min={other_pos[:,dim].min():.4f}  max={other_pos[:,dim].max():.4f}  "
          f"mean={other_pos[:,dim].mean():.4f}  std={other_pos[:,dim].std():.4f}")

# --- 3. Surface vs volume: check dimensionality via PCA / thickness ---
print("\n=== Surface vs volume test ===\n")

from numpy.linalg import svd

# Center and do SVD on airfoil points
af_centered = af_pos - af_pos.mean(axis=0)
_, s, Vt = svd(af_centered, full_matrices=False)
print(f"Singular values of airfoil point cloud: {s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}")
print(f"Ratio s2/s0 = {s[2]/s[0]:.4f}  (small → surface/shell, ~1 → volume)")

# Project onto each principal axis and look at the distribution along the thinnest
proj_thin = af_centered @ Vt[2]  # projection onto thinnest axis
print(f"Thinnest axis spread: min={proj_thin.min():.4f}  max={proj_thin.max():.4f}  "
      f"std={proj_thin.std():.4f}")

# Compare to a non-airfoil subsample
other_centered = other_pos - other_pos.mean(axis=0)
_, s_other, _ = svd(other_centered, full_matrices=False)
print(f"\nSingular values of non-airfoil cloud:  {s_other[0]:.2f}, {s_other[1]:.2f}, {s_other[2]:.2f}")
print(f"Ratio s2/s0 = {s_other[2]/s_other[0]:.4f}")

# --- 4. Check if pos is the same across time slices of the same geometry ---
print("\n=== Are pos coordinates identical across time slices? ===\n")

import re
geo_files = {}
for f in files:
    m = re.match(r"(.+)-(\d+)\.npz$", f.name)
    if m:
        geo_files.setdefault(m.group(1), []).append(f)

mismatches = 0
checked = 0
for geo, gfiles in sorted(geo_files.items())[:20]:
    positions = [np.load(f)["pos"] for f in gfiles]
    for i in range(1, len(positions)):
        checked += 1
        if not np.allclose(positions[0], positions[i]):
            mismatches += 1
            print(f"  MISMATCH: {geo} slice 0 vs {i}")

print(f"Checked {checked} pairs across 20 geometries: {mismatches} mismatches")

# --- 5. Flow direction: what's the dominant velocity direction? ---
print("\n=== Dominant flow direction (freestream) ===\n")

# Look at non-airfoil points far from the airfoil
data = np.load(files[0])
vel_in = data["velocity_in"]  # (5, 100000, 3)
idcs = data["idcs_airfoil"]
mask = np.ones(len(pos), dtype=bool)
mask[idcs] = True

# Mean velocity of all non-airfoil points at first timestep
mean_vel = vel_in[0, ~mask].mean(axis=0)
print(f"Mean velocity of non-airfoil points (t=0): [{mean_vel[0]:.4f}, {mean_vel[1]:.4f}, {mean_vel[2]:.4f}]")
print(f"  → Dominant flow direction: {'x' if abs(mean_vel[0]) > abs(mean_vel[1]) and abs(mean_vel[0]) > abs(mean_vel[2]) else 'y' if abs(mean_vel[1]) > abs(mean_vel[2]) else 'z'}-axis")
print(f"  → Flow magnitude: {np.linalg.norm(mean_vel):.4f}")

# Points downstream = those with x > airfoil centroid (assuming x is flow dir)
af_centroid = af_pos.mean(axis=0)
print(f"\nAirfoil centroid: [{af_centroid[0]:.4f}, {af_centroid[1]:.4f}, {af_centroid[2]:.4f}]")

# Check a few more samples to confirm flow direction consistency
print("\n=== Flow direction consistency (first 10 samples) ===")
for f in files[:10]:
    data = np.load(f)
    idcs = data["idcs_airfoil"]
    mask = np.zeros(100000, dtype=bool)
    mask[idcs] = True
    mean_v = data["velocity_in"][0, ~mask].mean(axis=0)
    print(f"  {f.name}: mean_vel = [{mean_v[0]:.3f}, {mean_v[1]:.3f}, {mean_v[2]:.3f}]")
