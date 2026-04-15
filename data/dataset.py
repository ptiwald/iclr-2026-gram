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


def make_split(
    data_dir: str,
    split_file: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create or load a train/test split of .npz file paths.

    If split_file exists, loads it. Otherwise scans data_dir, shuffles,
    splits, and saves the result to split_file for reproducibility.
    """
    if os.path.exists(split_file):
        with open(split_file) as f:
            return json.load(f)

    paths = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))
    n_train = int(len(paths) * train_ratio)

    split = {
        "train": [paths[i] for i in indices[:n_train]],
        "test": [paths[i] for i in indices[n_train:]],
    }

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Created split: {len(split['train'])} train, {len(split['test'])} test")
    return split


def make_dataloaders(
    data_dir: str,
    split_file: str = "split.json",
    batch_size: int = 2,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train and test DataLoaders."""
    split = make_split(data_dir, split_file, train_ratio, seed)

    loaders = {}
    for name, paths in split.items():
        dataset = WarpedIFWDataset(paths)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
