"""CPU smoke test + overfit-one-batch learnability gate for GraphUNet.

Part 1 (shape gate): construct all three regimes, check forward shape, no-slip
zeroing, and a backward pass (with and without gradient checkpointing):
  A  flat baseline   num_levels=1
  B  flat + skips     num_levels=3, pool_ratio=1.0  (skips, no coarsening -> same reach)
  C  graph-U-Net      num_levels=3, pool_ratio=0.25 (pooling buys reach)

Part 2 (learnability gate): take ONE fixed random batch and run a short Adam
loop on the flat baseline (the config that stalled under post-LN). A healthy
pre-LN net should drive the loss down by a large factor in ~150 steps. This is
the cheap local proof — run before spending any GPU time.

Run: conda run -n gram python -m models.graph_unet._smoke_test
"""
import torch

from models import GraphUNet

B, N, T = 2, 3000, 5

NARROW = dict(backbone="graph_transformer", hidden=128, heads=4, k=16, temporal_attn_layers=2)


def make_batch(b=B, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(b, 10, generator=g)
    pos = torch.rand(b, n, 3, generator=g) * torch.tensor([2.10, 0.82, 1.22]) - torch.tensor([0.0, 0.41, 0.0])
    idcs = [torch.randint(n, size=(torch.randint(80, 300, (1,), generator=g).item(),), generator=g) for _ in range(b)]
    vel_in = torch.rand(b, T, n, 3, generator=g)
    vel_out = torch.rand(b, T, n, 3, generator=g)
    return t, pos, idcs, vel_in, vel_out


CONFIGS = {
    "A flat baseline (num_levels=1, blocks=8)": dict(num_levels=1, blocks_per_level=8, **NARROW),
    "B flat+skips (num_levels=3, pool_ratio=1.0)": dict(num_levels=3, blocks_per_level=2, pool_ratio=1.0, **NARROW),
    "C graph-unet (num_levels=3, pool_ratio=0.25)": dict(num_levels=3, blocks_per_level=2, pool_ratio=0.25, **NARROW),
}

print("=" * 70)
print("PART 1 — shape / no-slip / backward gate")
print("=" * 70)
for name, kwargs in CONFIGS.items():
    print(f"\n=== {name} ===")
    for use_ckpt in (False, True):
        torch.manual_seed(0)
        model = GraphUNet(use_checkpoint=use_ckpt, **kwargs)
        model.train()
        t, pos, idcs, vel_in, vel_out = make_batch()
        pred = model(t, pos, idcs, vel_in)
        assert pred.shape == (B, T, N, 3), f"bad shape {pred.shape}"

        for b in range(B):
            assert torch.count_nonzero(pred[b, :, idcs[b], :]) == 0, "no-slip violated"

        loss = (pred - vel_out).norm(dim=3).mean()
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for _ in model.parameters())
        print(f"  use_checkpoint={use_ckpt}: shape OK | no-slip OK | "
              f"loss={loss.item():.4f} | grads {n_grad}/{n_total} tensors")

print("\n" + "=" * 70)
print("PART 2 — overfit-one-batch learnability gate (flat baseline, pre-LN)")
print("=" * 70)
torch.manual_seed(0)
model = GraphUNet(num_levels=1, blocks_per_level=8, **NARROW)
model.train()
# small fixed batch so CPU can run many steps
t, pos, idcs, vel_in, vel_out = make_batch(b=1, n=1500, seed=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

mask = torch.ones(vel_out.shape[2], dtype=torch.bool)
mask[idcs[0]] = False  # exclude no-slip (forced-zero) rows from the loss

init_loss = None
for step in range(150):
    opt.zero_grad()
    pred = model(t, pos, idcs, vel_in)
    loss = (pred[0, :, mask] - vel_out[0, :, mask]).norm(dim=-1).mean()
    loss.backward()
    opt.step()
    if init_loss is None:
        init_loss = loss.item()
    if step % 25 == 0 or step == 149:
        print(f"  step {step:3d}: loss {loss.item():.5f}")

ratio = loss.item() / init_loss
print(f"\n  init {init_loss:.5f} -> final {loss.item():.5f}  ({ratio:.2%} of init)")
assert ratio < 0.5, f"FAILED: flat baseline did not overfit one batch (ratio {ratio:.2%}) — recheck training path"
print("  LEARNABILITY OK — pre-LN flat baseline drives loss down on a fixed batch")

print("\nALL CHECKS PASSED")
