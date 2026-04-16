import argparse
import csv
import importlib
import os
import time

import torch

from data import make_dataloaders


def get_model_class(name: str):
    """Import a model class by name from the models package."""
    module = importlib.import_module("models")
    return getattr(module, name)


def train_one_epoch(model, loader, optimizer, device, epoch, max_steps=None, step_logger=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        pred = model(t, pos, idcs_airfoil, velocity_in)
        loss = (pred - velocity_out).norm(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        if step_logger is not None:
            step_logger(epoch, n_batches, loss_val)

        if max_steps is not None and n_batches >= max_steps:
            break

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        pred = model(t, pos, idcs_airfoil, velocity_in)
        loss = (pred - velocity_out).norm(dim=-1).mean()

        total_loss += loss.item()
        n_batches += 1

        if max_steps is not None and n_batches >= max_steps:
            break

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train a model on Warped-IFW data")
    parser.add_argument("--model", type=str, default="MLP", help="Model class name from models/")
    parser.add_argument("--data-dir", type=str, default="/home/paul/scratch/gram-competition/warped-ifw/")
    parser.add_argument("--split-file", type=str, default="split.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N training steps per epoch (default: full epoch)")
    args = parser.parse_args()

    print(f"Model: {args.model} | Device: {args.device} | Batch size: {args.batch_size}")

    # Data
    loaders = make_dataloaders(
        data_dir=args.data_dir,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(loaders['train'].dataset)} samples | Test: {len(loaders['test'].dataset)} samples")

    # Model — construct fresh (ignores pretrained weights for training)
    ModelClass = get_model_class(args.model)
    model = ModelClass().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss logging — persistent file handles, flushed per write for live tailing
    os.makedirs(args.log_dir, exist_ok=True)
    steps_path = os.path.join(args.log_dir, f"{args.model.lower()}_steps.csv")
    epochs_path = os.path.join(args.log_dir, f"{args.model.lower()}_epochs.csv")
    steps_file = open(steps_path, "w", newline="")
    epochs_file = open(epochs_path, "w", newline="")
    steps_writer = csv.writer(steps_file)
    epochs_writer = csv.writer(epochs_file)
    steps_writer.writerow(["epoch", "step_in_epoch", "train_loss"])
    epochs_writer.writerow(["epoch", "train_loss", "test_loss", "time_s"])
    steps_file.flush()
    epochs_file.flush()

    def log_step(epoch, step_in_epoch, loss_val):
        steps_writer.writerow([epoch, step_in_epoch, f"{loss_val:.6f}"])
        steps_file.flush()

    # Training loop
    best_test_loss = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, args.device, epoch,
                max_steps=args.max_steps, step_logger=log_step,
            )
            test_loss = evaluate(model, loaders["test"], args.device, args.max_steps)
            elapsed = time.time() - t0

            epochs_writer.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}", f"{elapsed:.2f}"])
            epochs_file.flush()

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, f"{args.model.lower()}_best.pt")
                torch.save(model.state_dict(), path)
                print(f"  -> Saved best checkpoint to {path}")
    finally:
        steps_file.close()
        epochs_file.close()

    print(f"Done. Best test loss: {best_test_loss:.4f}")
    print(f"Logs: {steps_path} | {epochs_path}")


if __name__ == "__main__":
    main()
