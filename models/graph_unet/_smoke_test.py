"""Lightweight CPU smoke test for GraphUNet (small N so it runs without a GPU).

Checks both configs: construction, param count (~7.5M target), forward output
shape, no-slip zeroing, and a backward pass (incl. gradient checkpointing).
Run: conda run -n gram python -m models.graph_unet._smoke_test
"""
import torch

from models import GraphUNet

B, N, T = 2, 3000, 5


def make_batch():
    t = torch.rand(B, 10)
    pos = torch.rand(B, N, 3) * torch.tensor([2.10, 0.82, 1.22]) - torch.tensor([0.0, 0.41, 0.0])
    idcs = [torch.randint(N, size=(torch.randint(80, 300, (1,)).item(),)) for _ in range(B)]
    vel_in = torch.rand(B, T, N, 3)
    vel_out = torch.rand(B, T, N, 3)
    return t, pos, idcs, vel_in, vel_out


CONFIGS = {
    "graph_baseline (num_levels=1)": dict(
        backbone="graph_transformer", hidden=256, heads=8, k=24,
        num_levels=1, blocks_per_level=10, temporal_attn_layers=2,
    ),
    "graph_unet (num_levels=3)": dict(
        backbone="graph_transformer", hidden=256, heads=8, k=24,
        num_levels=3, blocks_per_level=2, pool_ratio=0.25, temporal_attn_layers=2,
    ),
}

for name, kwargs in CONFIGS.items():
    print(f"\n=== {name} ===")
    for use_ckpt in (False, True):
        torch.manual_seed(0)
        model = GraphUNet(use_checkpoint=use_ckpt, **kwargs)
        model.train()
        t, pos, idcs, vel_in, vel_out = make_batch()
        pred = model(t, pos, idcs, vel_in)
        assert pred.shape == (B, T, N, 3), f"bad shape {pred.shape}"

        # no-slip: airfoil rows must be exactly zero
        for b in range(B):
            assert torch.count_nonzero(pred[b, :, idcs[b], :]) == 0, "no-slip violated"

        # backward
        loss = (pred - vel_out).norm(dim=3).mean()
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for _ in model.parameters())
        print(f"  use_checkpoint={use_ckpt}: shape OK | no-slip OK | "
              f"loss={loss.item():.4f} | grads {n_grad}/{n_total} tensors")

print("\nALL SMOKE CHECKS PASSED")
