import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WarpedIFWDataset(Dataset):
    """Lazily loads individual .npz sample files from disk."""

    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        return {
            "t": torch.from_numpy(data["t"]),                         # (10,)
            "pos": torch.from_numpy(data["pos"]),                     # (100000, 3)
            "idcs_airfoil": torch.from_numpy(data["idcs_airfoil"]),   # (variable,)
            "velocity_in": torch.from_numpy(data["velocity_in"]),     # (5, 100000, 3)
            "velocity_out": torch.from_numpy(data["velocity_out"]),   # (5, 100000, 3)
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that keeps idcs_airfoil as a list of variable-length tensors."""
    return {
        "t": torch.stack([s["t"] for s in batch]),
        "pos": torch.stack([s["pos"] for s in batch]),
        "idcs_airfoil": [s["idcs_airfoil"] for s in batch],
        "velocity_in": torch.stack([s["velocity_in"] for s in batch]),
        "velocity_out": torch.stack([s["velocity_out"] for s in batch]),
    }


def _geometry_key(path: str) -> str:
    """Extract geometry identifier from a file path.

    E.g. '/data/1021_1-3.npz' -> '1021_1' (everything before the last '-N.npz').
    """
    name = os.path.splitext(os.path.basename(path))[0]
    return name.rsplit("-", 1)[0]


def load_split(split_file: str) -> dict[str, list[str]]:
    """Load the canonical geometry-level train/test split.

    The split is committed to the repo and must not be regenerated implicitly —
    regenerating on a different machine (or with different files in the data
    dir) would silently produce a different test set and invalidate
    cross-machine comparisons. To regenerate intentionally, call
    `make_split(...)` from a script.
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"{split_file} not found. The split is canonical and committed to "
            f"the repo; run `make_split` explicitly if you really intend to "
            f"regenerate it."
        )
    with open(split_file) as f:
        return json.load(f)


def make_split(
    data_dir: str,
    split_file: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate a geometry-level train/test split and write it to disk.

    Splits by geometry so that all time windows of a given geometry land in
    the same split, matching competition conditions where test geometries are
    unseen. The split stores bare filenames (no directory prefix) so the same
    file is portable across machines with different data dirs. Only call this
    when you intentionally want a new split — the loaders use `load_split`
    and will not regenerate.
    """
    paths = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    names = [os.path.basename(p) for p in paths]
    geo_to_names: dict[str, list[str]] = {}
    for n in names:
        geo_to_names.setdefault(_geometry_key(n), []).append(n)

    geometries = sorted(geo_to_names.keys())
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(geometries))
    n_train = int(len(geometries) * train_ratio)

    train_names: list[str] = []
    test_names: list[str] = []
    for i in indices[:n_train]:
        train_names.extend(geo_to_names[geometries[i]])
    for i in indices[n_train:]:
        test_names.extend(geo_to_names[geometries[i]])

    split = {"train": train_names, "test": test_names}

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)

    n_test_geo = len(geometries) - n_train
    print(f"Created split: {n_train} geometries ({len(train_names)} samples) train, "
          f"{n_test_geo} geometries ({len(test_names)} samples) test")
    return split


def make_dataloaders(
    data_dir: str,
    split_file: str = "split.json",
    batch_size: int = 2,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create train and test DataLoaders using the canonical committed split.

    `data_dir` is the machine-local location of the .npz files; it is
    prepended to the bare filenames stored in the split.
    """
    split = load_split(split_file)

    loaders = {}
    for name, filenames in split.items():
        paths = [os.path.join(data_dir, fn) for fn in filenames]
        dataset = WarpedIFWDataset(paths)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    return loaders
