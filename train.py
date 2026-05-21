import argparse
import contextlib
import csv
import importlib
import os
import time

import torch
import yaml

from data import make_dataloaders

DEFAULTS = {
    "model": "MLP",
    "data_dir": "/workspace/data/warped-ifw/",
    "split_file": "split.json",
    "batch_size": 8,
    "num_workers": 2,
    "lr": 1e-3,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "/workspace/checkpoints",
    "log_dir": "/workspace/logs/",
    "max_steps": None,
    "bf16": False,
    "lr_schedule": None,  # None or "cosine"
    "min_lr": 0.0,
    "model_kwargs": {},
    # EMA: None disables; a float in (0,1) (e.g. 0.9999) enables. Shadow weights
    # are used for eval + saved as the checkpoint.
    "ema_decay": None,
    "ema_warmup_steps": 1000,
}


class EMA:
    """Maintain an exponential moving average of a model's state_dict.

    Non-float tensors (e.g. integer buffers) are copied rather than averaged.
    Decay is ramped up linearly so early steps don't get locked into noisy init:
        effective_decay = min(decay, step / (step + warmup))
    """

    def __init__(self, model: torch.nn.Module, decay: float, warmup_steps: int = 1000):
        self.decay = decay
        self.warmup = max(1, warmup_steps)
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = min(self.decay, self.step / (self.step + self.warmup))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return self.shadow

    @contextlib.contextmanager
    def swap(self, model: torch.nn.Module):
        """Temporarily load EMA weights into `model`; restore on exit."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(backup)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    merged = {**DEFAULTS, **cfg}
    return merged


def get_model_class(name: str):
    """Import a model class by name from the models package."""
    module = importlib.import_module("models")
    return getattr(module, name)


def train_one_epoch(model, loader, optimizer, device, epoch, max_steps=None, step_logger=None, bf16=False, ema=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=bf16):
            pred = model(t, pos, idcs_airfoil, velocity_in)

            # L2 norm per point (across 3 velocity components), exclude airfoil surface
            per_point_err = (pred - velocity_out).norm(dim=3)  # (B, 5, N)
            mask = torch.ones_like(per_point_err, dtype=torch.bool)
            for i, idcs in enumerate(idcs_airfoil):
                mask[i, :, idcs] = False
            loss = per_point_err[mask].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        if step_logger is not None:
            step_logger(epoch, n_batches, loss_val)

        if max_steps is not None and n_batches >= max_steps:
            break

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None, bf16=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=bf16):
            pred = model(t, pos, idcs_airfoil, velocity_in)

            per_point_err = (pred - velocity_out).norm(dim=3)  # (B, 5, N)
            mask = torch.ones_like(per_point_err, dtype=torch.bool)
            for i, idcs in enumerate(idcs_airfoil):
                mask[i, :, idcs] = False
            loss = per_point_err[mask].mean()

        total_loss += loss.item()
        n_batches += 1

        if max_steps is not None and n_batches >= max_steps:
            break

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train a model on Warped-IFW data")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Model: {cfg['model']} | Device: {cfg['device']} | Batch size: {cfg['batch_size']}")

    # Data
    loaders = make_dataloaders(
        data_dir=cfg["data_dir"],
        split_file=cfg["split_file"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["device"].startswith("cuda"),
    )
    print(f"Train: {len(loaders['train'].dataset)} samples | Test: {len(loaders['test'].dataset)} samples")

    # Model — construct fresh (ignores pretrained weights for training)
    ModelClass = get_model_class(cfg["model"])
    model = ModelClass(**cfg["model_kwargs"]).to(cfg["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    ema = None
    if cfg["ema_decay"] is not None:
        ema = EMA(model, decay=cfg["ema_decay"], warmup_steps=cfg["ema_warmup_steps"])
        print(f"EMA: decay={cfg['ema_decay']} warmup_steps={cfg['ema_warmup_steps']}")

    scheduler = None
    if cfg["lr_schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"],
        )
    elif cfg["lr_schedule"] is not None:
        raise ValueError(f"Unknown lr_schedule: {cfg['lr_schedule']}")

    # Loss logging — persistent file handles, flushed per write for live tailing
    os.makedirs(cfg["log_dir"], exist_ok=True)
    model_tag = cfg.get("run_name", cfg["model"].lower())
    steps_path = os.path.join(cfg["log_dir"], f"{model_tag}_steps.csv")
    epochs_path = os.path.join(cfg["log_dir"], f"{model_tag}_epochs.csv")
    steps_file = open(steps_path, "w", newline="")
    epochs_file = open(epochs_path, "w", newline="")
    steps_writer = csv.writer(steps_file)
    epochs_writer = csv.writer(epochs_file)
    steps_writer.writerow(["epoch", "step_in_epoch", "train_loss"])
    epochs_writer.writerow(["epoch", "train_loss", "test_loss", "test_loss_raw", "time_s"])
    steps_file.flush()
    epochs_file.flush()

    def log_step(epoch, step_in_epoch, loss_val):
        steps_writer.writerow([epoch, step_in_epoch, f"{loss_val:.6f}"])
        steps_file.flush()

    # Training loop
    best_test_loss = float("inf")
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, cfg["device"], epoch,
                max_steps=cfg["max_steps"], step_logger=log_step, bf16=cfg["bf16"], ema=ema,
            )
            if ema is not None:
                test_loss_raw = evaluate(model, loaders["test"], cfg["device"], cfg["max_steps"], bf16=cfg["bf16"])
                with ema.swap(model):
                    test_loss = evaluate(model, loaders["test"], cfg["device"], cfg["max_steps"], bf16=cfg["bf16"])
            else:
                test_loss = evaluate(model, loaders["test"], cfg["device"], cfg["max_steps"], bf16=cfg["bf16"])
                test_loss_raw = test_loss
            if scheduler is not None:
                scheduler.step()
            elapsed = time.time() - t0

            epochs_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{test_loss:.6f}",
                f"{test_loss_raw:.6f}", f"{elapsed:.2f}",
            ])
            epochs_file.flush()

            if ema is not None:
                print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"Train loss: {train_loss:.4f} | Test loss (EMA): {test_loss:.4f} | "
                      f"Test loss (raw): {test_loss_raw:.4f} | Time: {elapsed:.1f}s")
            else:
                print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
                path = os.path.join(cfg["checkpoint_dir"], f"{model_tag}_best.pt")
                sd = ema.state_dict() if ema is not None else model.state_dict()
                torch.save(sd, path)
                print(f"  -> Saved best checkpoint to {path}")
    finally:
        steps_file.close()
        epochs_file.close()

    print(f"Done. Best test loss: {best_test_loss:.4f}")
    print(f"Logs: {steps_path} | {epochs_path}")


if __name__ == "__main__":
    main()
