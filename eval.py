import importlib
import sys

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data import make_dataloaders
from models import LastFrame

AIRFOIL_CLUSTER_EPS = 0.02
AIRFOIL_CLUSTER_MIN_SIZE = 5


def count_airfoils(pos: np.ndarray, idcs: np.ndarray) -> int:
    pts = pos[idcs]
    if len(pts) < AIRFOIL_CLUSTER_MIN_SIZE:
        return 1
    pairs = cKDTree(pts).query_pairs(AIRFOIL_CLUSTER_EPS, output_type="ndarray")
    n = len(pts)
    if len(pairs) == 0:
        return n
    A = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(A + A.T, directed=False)
    return int((np.bincount(labels) >= AIRFOIL_CLUSTER_MIN_SIZE).sum())

DEFAULTS = {
    "model": "MLP",
    "checkpoint": None,
    "split": "test",
    "data_dir": "/workspace/data/warped-ifw/",
    "split_file": "split.json",
    "batch_size": 15,
    "num_workers": 4,
    "device": "cuda",
    "bf16": False,
    "model_kwargs": {},
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {**DEFAULTS, **cfg}


def get_model_class(name: str):
    module = importlib.import_module("models")
    return getattr(module, name)


def _rel_l2_masked(pred: torch.Tensor, gt: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 over points selected by `mask_bn` (B, N).

    Numerator = sum of squared errors over (T, selected N, 3); denominator
    likewise over the targets. Returns (B,).
    """
    mask = mask_bn[:, None, :, None].to(pred.dtype)  # (B, 1, N, 1)
    sq_err = (pred - gt) ** 2 * mask
    sq_gt = gt ** 2 * mask
    num = sq_err.sum(dim=(1, 2, 3))
    denom = sq_gt.sum(dim=(1, 2, 3))
    return (num / denom.clamp(min=1e-12)).sqrt()


@torch.no_grad()
def evaluate(model, loader, device, bf16=False):
    model.eval()
    last_frame = LastFrame().to(device).eval()

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    # Per-sample metrics (concatenated across batches): mean ± std at the end.
    metric_all, metric_wake, metric_rest = [], [], []
    rel_l2_all, rel_l2_wake, rel_l2_rest = [], [], []
    lf_rel_l2_all, lf_rel_l2_wake, lf_rel_l2_rest = [], [], []
    n_airfoils_per_sample: list[int] = []

    # Per-batch scalars (averaged at the end): per-point L2 norm in each region.
    lf_all_sum, m_all_sum = 0.0, 0.0
    lf_wake_sum, lf_rest_sum = 0.0, 0.0
    m_wake_sum, m_rest_sum = 0.0, 0.0
    val_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        pos_np = batch["pos"].numpy()
        for b_idx, idx_t in enumerate(batch["idcs_airfoil"]):
            n_airfoils_per_sample.append(count_airfoils(pos_np[b_idx], idx_t.numpy()))

        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=bf16):
            pred = model(t, pos, idcs_airfoil, velocity_in)
            lf_pred = last_frame(t, pos, idcs_airfoil, velocity_in)
        pred = pred.float()
        lf_pred = lf_pred.float()

        B, _, N, _ = velocity_out.shape

        # Wake/rest split: per-point error averaged over time, wake = top 10% by LastFrame.
        lf_pp = (lf_pred - velocity_out).norm(dim=-1).mean(dim=1)  # (B, N)
        m_pp = (pred - velocity_out).norm(dim=-1).mean(dim=1)      # (B, N)
        k = int(round(0.1 * N))
        thresh = lf_pp.topk(k, dim=1).values[:, -1:]
        wake = lf_pp >= thresh           # (B, N) bool
        rest = ~wake                     # (B, N) bool
        all_mask = torch.ones_like(wake)

        # Organizer's per-sample relative L2 metric, restricted to each region.
        rel_l2_all.append(_rel_l2_masked(pred, velocity_out, all_mask))
        rel_l2_wake.append(_rel_l2_masked(pred, velocity_out, wake))
        rel_l2_rest.append(_rel_l2_masked(pred, velocity_out, rest))
        lf_rel_l2_all.append(_rel_l2_masked(lf_pred, velocity_out, all_mask))
        lf_rel_l2_wake.append(_rel_l2_masked(lf_pred, velocity_out, wake))
        lf_rel_l2_rest.append(_rel_l2_masked(lf_pred, velocity_out, rest))

        # Per-sample per-point L2 norm (mean over T and points), restricted to each region.
        # `m_pp` and `lf_pp` are already (B, N) per-point errors averaged over T.
        def _masked_per_sample_mean(pp: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            # pp: (B, N), m: (B, N) bool. Returns (B,) per-sample mean over m.
            mf = m.to(pp.dtype)
            return (pp * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)

        metric_all.append(_masked_per_sample_mean(m_pp, all_mask))
        metric_wake.append(_masked_per_sample_mean(m_pp, wake))
        metric_rest.append(_masked_per_sample_mean(m_pp, rest))

        # Per-batch scalars (kept for readability; equivalent to means of the per-sample tensors).
        lf_all_sum += lf_pp.mean().item()
        m_all_sum += m_pp.mean().item()
        lf_wake_sum += lf_pp[wake].mean().item()
        lf_rest_sum += lf_pp[rest].mean().item()
        m_wake_sum += m_pp[wake].mean().item()
        m_rest_sum += m_pp[rest].mean().item()

        # train.py loss: mean L2 norm over the whole batch (no region split).
        val_loss_sum += (pred - velocity_out).norm(dim=-1).mean().item()
        n_batches += 1

    n = n_batches
    return {
        # Concatenated per-sample tensors (mean ± std reported at print time).
        "metric_all": torch.cat(metric_all),
        "metric_wake": torch.cat(metric_wake),
        "metric_rest": torch.cat(metric_rest),
        "rel_l2_all": torch.cat(rel_l2_all),
        "rel_l2_wake": torch.cat(rel_l2_wake),
        "rel_l2_rest": torch.cat(rel_l2_rest),
        "lf_rel_l2_all": torch.cat(lf_rel_l2_all),
        "lf_rel_l2_wake": torch.cat(lf_rel_l2_wake),
        "lf_rel_l2_rest": torch.cat(lf_rel_l2_rest),
        "n_airfoils": torch.tensor(n_airfoils_per_sample, dtype=torch.int64),
        # Per-batch averages for the per-point L2 norm summary (kept for compat with prior format).
        "val_loss": val_loss_sum / n,
        "lf_all": lf_all_sum / n,
        "m_all": m_all_sum / n,
        "lf_wake": lf_wake_sum / n,
        "lf_rest": lf_rest_sum / n,
        "m_wake": m_wake_sum / n,
        "m_rest": m_rest_sum / n,
    }


def _row(label: str, *vals: torch.Tensor | float, fmt: str = "{:.4f}") -> str:
    cells = [f"{label:<20}"]
    for v in vals:
        if isinstance(v, torch.Tensor):
            cells.append(f"{fmt.format(v.mean().item()):>12}")
        else:
            cells.append(f"{fmt.format(v):>12}")
    return " ".join(cells)


def main(cfg):
    loaders = make_dataloaders(
        data_dir=cfg["data_dir"],
        split_file=cfg["split_file"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["device"].startswith("cuda"),
    )
    loader = loaders[cfg["split"]]

    ModelClass = get_model_class(cfg["model"])
    model = ModelClass(**cfg["model_kwargs"]).to(cfg["device"])

    if cfg["checkpoint"] is not None:
        state_dict = torch.load(cfg["checkpoint"], map_location=cfg["device"], weights_only=True)
        model.load_state_dict(state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']} | Params: {n_params:,} | Device: {cfg['device']} | "
          f"Split: {cfg['split']} ({len(loader.dataset)} samples)"
          + (f" | Checkpoint: {cfg['checkpoint']}" if cfg["checkpoint"] else ""))

    r = evaluate(model, loader, cfg["device"], bf16=cfg["bf16"])
    print(f"Val loss (train.py-style): {r['val_loss']:.4f}")
    print()

    print("Wake = top 10% of points per sample by LastFrame per-point L2 error.")
    header = f"{'Model':<20} {'All':>12} {'Wake (10%)':>12} {'Rest (90%)':>12}"

    print("Rel-L2 (leaderboard metric, per-sample, mean over batch):")
    print(header)
    print(_row("LastFrame", r["lf_rel_l2_all"], r["lf_rel_l2_wake"], r["lf_rel_l2_rest"]))
    print(_row(cfg["model"], r["rel_l2_all"], r["rel_l2_wake"], r["rel_l2_rest"]))
    print()

    print("Per-point L2 norm (mean across points and time, mean over batch):")
    print(header)
    print(_row("LastFrame", r["lf_all"], r["lf_wake"], r["lf_rest"]))
    print(_row(cfg["model"], r["m_all"], r["m_wake"], r["m_rest"]))
    print()

    # Per-airfoil-count subgroup breakdown (Rel-L2).
    n_af = r["n_airfoils"]
    print("Rel-L2 by airfoil count (per-sample mean ± std; n = sample count):")
    sub_header = f"{'Subgroup':<20} {'n':>6} {'All':>12} {'Wake (10%)':>12} {'Rest (90%)':>12}"
    print(sub_header)
    for k in sorted(set(n_af.tolist())):
        mask = n_af == k
        n_k = int(mask.sum())
        a = r["rel_l2_all"][mask]
        w = r["rel_l2_wake"][mask]
        rr = r["rel_l2_rest"][mask]
        line = (
            f"{f'{k}af':<20} {n_k:>6} "
            f"{a.mean().item():>6.4f}±{a.std().item():.4f} "
            f"{w.mean().item():>6.4f}±{w.std().item():.4f} "
            f"{rr.mean().item():>6.4f}±{rr.std().item():.4f}"
        )
        print(line)


if __name__ == "__main__":
    main(load_config(sys.argv[1]))
