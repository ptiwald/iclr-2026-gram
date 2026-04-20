import importlib
import sys

import torch
import yaml

from data import make_dataloaders
from models import LastFrame

DEFAULTS = {
    "model": "MLP",
    "checkpoint": None,
    "split": "test",
    "data_dir": "/home/paul/scratch/gram-competition/warped-ifw/",
    "split_file": "split.json",
    "batch_size": 15,
    "num_workers": 4,
    "device": "cpu",
    "bf16": False,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {**DEFAULTS, **cfg}


def get_model_class(name: str):
    module = importlib.import_module("models")
    return getattr(module, name)


@torch.no_grad()
def evaluate(model, loader, device, bf16=False):
    model.eval()
    last_frame = LastFrame().to(device).eval()

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    per_sample = []
    batch_losses = []
    lf_all_sum, m_all_sum = 0.0, 0.0
    lf_wake_sum, lf_rest_sum = 0.0, 0.0
    m_wake_sum, m_rest_sum = 0.0, 0.0
    n_batches = 0

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=bf16):
            pred = model(t, pos, idcs_airfoil, velocity_in)
            lf_pred = last_frame(t, pos, idcs_airfoil, velocity_in)
        pred = pred.float()
        lf_pred = lf_pred.float()

        # main.py metric: per-sample L2 norm averaged over timesteps and points.
        per_sample.append((pred - velocity_out).norm(dim=3).mean(dim=(1, 2)))
        # train.py loss: mean L2 norm over the whole batch.
        batch_losses.append((pred - velocity_out).norm(dim=-1).mean().item())

        # Wake/rest split: per-point error averaged over time, wake = top 10% by LastFrame.
        lf_pp = (lf_pred - velocity_out).norm(dim=-1).mean(dim=1)  # (B, N)
        m_pp = (pred - velocity_out).norm(dim=-1).mean(dim=1)      # (B, N)
        k = int(round(0.1 * lf_pp.shape[1]))
        thresh = lf_pp.topk(k, dim=1).values[:, -1:]
        wake = lf_pp >= thresh
        rest = ~wake

        lf_all_sum += lf_pp.mean().item()
        m_all_sum += m_pp.mean().item()
        lf_wake_sum += lf_pp[wake].mean().item()
        lf_rest_sum += lf_pp[rest].mean().item()
        m_wake_sum += m_pp[wake].mean().item()
        m_rest_sum += m_pp[rest].mean().item()
        n_batches += 1

    n = n_batches
    return {
        "metric": torch.cat(per_sample),
        "val_loss": sum(batch_losses) / n,
        "lf_all": lf_all_sum / n,
        "m_all": m_all_sum / n,
        "lf_wake": lf_wake_sum / n,
        "lf_rest": lf_rest_sum / n,
        "m_wake": m_wake_sum / n,
        "m_rest": m_rest_sum / n,
    }


def main(cfg):
    loaders = make_dataloaders(
        split_file=cfg["split_file"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["device"].startswith("cuda"),
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

    r = evaluate(model, loader, cfg["device"], bf16=cfg["bf16"])
    print(f"Metric: {r['metric'].mean():.4f} +- {r['metric'].std():.4f}")
    print(f"Val loss (train.py-style): {r['val_loss']:.4f}")
    print()
    print("Wake = top 10% of points per sample by LastFrame per-point L2 error.")
    print(f"{'Model':<20} {'All':>10} {'Wake (10%)':>12} {'Rest (90%)':>12}")
    print(f"{'LastFrame':<20} {r['lf_all']:>10.4f} {r['lf_wake']:>12.4f} {r['lf_rest']:>12.4f}")
    print(f"{cfg['model']:<20} {r['m_all']:>10.4f} {r['m_wake']:>12.4f} {r['m_rest']:>12.4f}")


if __name__ == "__main__":
    main(load_config(sys.argv[1]))
