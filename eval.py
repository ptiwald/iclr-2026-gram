import argparse
import importlib
import os

import torch

from data import make_dataloaders


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
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional state_dict path to load after construction")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--data-dir", type=str, default="/home/paul/scratch/gram-competition/warped-ifw/")
    parser.add_argument("--split-file", type=str, default="split.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    loaders = make_dataloaders(
        data_dir=args.data_dir,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = loaders[args.split]

    ModelClass = get_model_class(args.model)
    model = ModelClass().to(args.device)

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params:,} | Device: {args.device} | "
          f"Split: {args.split} ({len(loader.dataset)} samples)"
          + (f" | Checkpoint: {args.checkpoint}" if args.checkpoint else ""))

    metric, val_loss = evaluate(model, loader, args.device)
    print(f"Metric: {metric.mean():.4f} +- {metric.std():.4f}")
    print(f"Val loss (train.py-style): {val_loss:.4f}")


if __name__ == "__main__":
    main()
