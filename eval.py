import argparse
import importlib

import torch
import yaml

from data import make_dataloaders

DEFAULTS = {
    "model": "MLP",
    "checkpoint": None,
    "split": "test",
    "data_dir": "/home/paul/scratch/gram-competition/warped-ifw/",
    "split_file": "split.json",
    "batch_size": 2,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {**DEFAULTS, **cfg}


def get_model_class(name: str):
    module = importlib.import_module("models")
    return getattr(module, name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    per_sample = []
    batch_losses = []

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        pred = model(t, pos, idcs_airfoil, velocity_in)
        # main.py metric: per-sample L2 norm averaged over timesteps and points.
        per_sample.append((pred - velocity_out).norm(dim=3).mean(dim=(1, 2)))
        # train.py loss: mean L2 norm over the whole batch.
        batch_losses.append((pred - velocity_out).norm(dim=-1).mean().item())

    return torch.cat(per_sample), sum(batch_losses) / len(batch_losses)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the Warped-IFW split")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    loaders = make_dataloaders(
        data_dir=cfg["data_dir"],
        split_file=cfg["split_file"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    loader = loaders[cfg["split"]]

    ModelClass = get_model_class(cfg["model"])
    model = ModelClass().to(cfg["device"])

    if cfg["checkpoint"] is not None:
        state_dict = torch.load(cfg["checkpoint"], map_location=cfg["device"], weights_only=True)
        model.load_state_dict(state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']} | Params: {n_params:,} | Device: {cfg['device']} | "
          f"Split: {cfg['split']} ({len(loader.dataset)} samples)"
          + (f" | Checkpoint: {cfg['checkpoint']}" if cfg["checkpoint"] else ""))

    metric, val_loss = evaluate(model, loader, cfg["device"])
    print(f"Metric: {metric.mean():.4f} +- {metric.std():.4f}")
    print(f"Val loss (train.py-style): {val_loss:.4f}")


if __name__ == "__main__":
    main()
