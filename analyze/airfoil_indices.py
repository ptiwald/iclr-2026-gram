"""Analyze uniqueness of idcs_airfoil across samples."""

import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/paul/scratch/gram-competition/warped-ifw")

files = sorted(DATA_DIR.glob("*.npz"))
print(f"Found {len(files)} samples\n")

# Collect hashable representations of each idcs_airfoil
index_sets = {}  # frozenset -> list of filenames
lengths = []

for f in files:
    data = np.load(f)
    idcs = data["idcs_airfoil"]
    lengths.append(len(idcs))
    key = tuple(sorted(idcs.tolist()))
    index_sets.setdefault(key, []).append(f.name)

print(f"Number of unique idcs_airfoil vectors: {len(index_sets)}")
print(f"\nLength stats: min={min(lengths)}, max={max(lengths)}, "
      f"mean={np.mean(lengths):.1f}, unique lengths={len(set(lengths))}")

print(f"\n=== Clusters (samples sharing the same idcs_airfoil) ===")
clusters = sorted(index_sets.values(), key=len, reverse=True)
for i, members in enumerate(clusters):
    if len(members) > 1:
        print(f"\n  Cluster {i+1} ({len(members)} samples):")
        for name in sorted(members):
            print(f"    {name}")

singletons = sum(1 for m in clusters if len(m) == 1)
if singletons:
    print(f"\n  + {singletons} unique (single-sample) vectors")
