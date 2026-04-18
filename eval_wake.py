"""Evaluate VanillaPointNet on wake points, identified as the top 10% of points
with largest per-point error under the LastFrame persistence baseline.
"""
import argparse

import torch

from data import make_dataloaders
from models import LastFrame, VanillaPointNet


def per_point_err(pred, truth):
    # (B, T, N, 3) -> (B, N): mean over time of per-point L2 norm
    return (pred - truth).norm(dim=-1).mean(dim=1)


@torch.no_grad()
def run(args):
    device = args.device
    loaders = make_dataloaders(
        data_dir=args.data_dir,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = loaders[args.split]

    last_frame = LastFrame().to(device).eval()
    pointnet = VanillaPointNet().to(device).eval()
    if args.checkpoint is not None:
        pointnet.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    lf_wake_sum, lf_rest_sum = 0.0, 0.0
    pn_wake_sum, pn_rest_sum = 0.0, 0.0
    pn_all_sum, lf_all_sum = 0.0, 0.0
    n_batches = 0

    for batch in loader:
        t = batch["t"].to(device)
        pos = batch["pos"].to(device)
        idcs_airfoil = [idx.to(device) for idx in batch["idcs_airfoil"]]
        velocity_in = batch["velocity_in"].to(device)
        velocity_out = batch["velocity_out"].to(device)

        lf_pred = last_frame(t, pos, idcs_airfoil, velocity_in)
        pn_pred = pointnet(t, pos, idcs_airfoil, velocity_in)

        lf_pp = per_point_err(lf_pred, velocity_out)  # (B, N)
        pn_pp = per_point_err(pn_pred, velocity_out)  # (B, N)

        # Top-10% mask per sample from last_frame errors.
        k = int(round(0.1 * lf_pp.shape[1]))
        thresh = lf_pp.topk(k, dim=1).values[:, -1:]  # (B, 1)
        wake = lf_pp >= thresh                        # (B, N)
        rest = ~wake

        lf_wake_sum += lf_pp[wake].mean().item()
        lf_rest_sum += lf_pp[rest].mean().item()
        pn_wake_sum += pn_pp[wake].mean().item()
        pn_rest_sum += pn_pp[rest].mean().item()
        lf_all_sum += lf_pp.mean().item()
        pn_all_sum += pn_pp.mean().item()
        n_batches += 1

    n = n_batches
    print(f"Split: {args.split} ({len(loader.dataset)} samples, {n} batches)")
    print(f"Wake = top 10% of points per sample by LastFrame per-point L2 error.")
    print()
    print(f"{'Model':<16} {'All':>10} {'Wake (10%)':>12} {'Rest (90%)':>12}")
    print(f"{'LastFrame':<16} {lf_all_sum/n:>10.4f} {lf_wake_sum/n:>12.4f} {lf_rest_sum/n:>12.4f}")
    print(f"{'VanillaPointNet':<16} {pn_all_sum/n:>10.4f} {pn_wake_sum/n:>12.4f} {pn_rest_sum/n:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--data-dir", type=str, default="/home/paul/scratch/gram-competition/warped-ifw/")
    parser.add_argument("--split-file", type=str, default="split.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str,
                        default="scratch/checkpoints_runpod/vanillapointnet_best.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
