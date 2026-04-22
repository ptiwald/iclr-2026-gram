"""Build a y-mirrored training dataset in a fresh directory.

Leaves the source dataset untouched. Writes to `--dst_dir`:
  - every file from the source split (train + test) copied verbatim
  - for each *train* file, an additional `_ymirror.npz` with y-flipped pos and
    y-flipped velocity_in/velocity_out (t, pressure, idcs_airfoil unchanged)
  - a new split JSON where `train = originals + mirrors` and `test` is unchanged.

Config-wise you then just point `data_dir` at --dst_dir and `split_file` at the
new split to A/B against the original. Idempotent: skips files already present
unless --force is passed.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def mirror_sample(src_npz: Path, dst_npz: Path) -> None:
    """Write a y-mirrored copy. pos.y, v_in.y, v_out.y negated; other keys passthrough."""
    with np.load(src_npz) as d:
        out = {k: np.array(d[k]) for k in d.files}
    out["pos"][:, 1] *= -1
    out["velocity_in"][..., 1] *= -1
    out["velocity_out"][..., 1] *= -1
    np.savez(dst_npz, **out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", type=Path, required=True,
                   help="Source .npz directory (read-only).")
    p.add_argument("--dst_dir", type=Path, required=True,
                   help="Destination directory (created if missing).")
    p.add_argument("--src_split", type=Path, default=Path("split.json"),
                   help="Source split JSON with 'train' and 'test' lists.")
    p.add_argument("--dst_split", type=Path, default=None,
                   help="Destination split JSON path. Defaults to "
                        "<dst_dir>/split.json so the split travels with the data.")
    p.add_argument("--suffix", type=str, default="_ymirror",
                   help="Filename suffix for mirrored copies (before .npz).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files in dst_dir.")
    args = p.parse_args()

    dst_split = args.dst_split or (args.dst_dir / "split.json")
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    with open(args.src_split) as f:
        split = json.load(f)
    train_files = list(split["train"])
    test_files = list(split["test"])
    print(f"Source: {args.src_dir}")
    print(f"Destination: {args.dst_dir}")
    print(f"Train: {len(train_files)}   Test: {len(test_files)}")

    def copy_if_needed(name: str) -> None:
        src = args.src_dir / name
        dst = args.dst_dir / name
        if dst.exists() and not args.force:
            return
        shutil.copyfile(src, dst)

    # 1) Copy originals (train + test).
    n_copied = 0
    for name in train_files + test_files:
        dst = args.dst_dir / name
        pre_existing = dst.exists() and not args.force
        copy_if_needed(name)
        if not pre_existing:
            n_copied += 1
    print(f"Copied {n_copied}/{len(train_files) + len(test_files)} originals "
          f"(remaining already in place).")

    # 2) Write mirrors for train.
    mirror_names = []
    n_mirrored = 0
    for name in train_files:
        stem = name[:-4] if name.endswith(".npz") else name
        mirror_name = f"{stem}{args.suffix}.npz"
        mirror_names.append(mirror_name)
        dst = args.dst_dir / mirror_name
        if dst.exists() and not args.force:
            continue
        mirror_sample(args.src_dir / name, dst)
        n_mirrored += 1
        if n_mirrored % 50 == 0:
            print(f"  mirrored {n_mirrored}/{len(train_files)}")
    print(f"Mirrored {n_mirrored}/{len(train_files)} train samples "
          f"(remaining already in place).")

    # 3) Write new split (train = originals + mirrors, test = originals only).
    new_split = {
        "train": train_files + mirror_names,
        "test": test_files,
    }
    dst_split.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_split, "w") as f:
        json.dump(new_split, f, indent=2)
    print(f"Wrote split to {dst_split}: "
          f"{len(new_split['train'])} train, {len(new_split['test'])} test.")


if __name__ == "__main__":
    main()
