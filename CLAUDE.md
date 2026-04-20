# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a submission portal for the GRaM (Geometry-grounded Representation Learning and Generative Modeling) workshop competition at ICLR 2026. Submissions are made as pull requests containing model implementations.

- **Competition page:** https://gram-competition.github.io/
- **Dataset:** https://huggingface.co/datasets/gram-competition/warped-ifw
- **Deadline:** April 22, 2026 (AoE)
- **Prize:** MCML Award of €500; valid submissions offered co-authorship in workshop proceedings

## Competition Task

Predict airflow dynamics around Formula 1 front wing geometries: given velocity fields at 5 earlier timesteps, predict velocity fields at 5 future timesteps. This is a neural operator problem conditioned on airfoil geometry.

Key details:
- **Dataset:** 181 simulations with 1–3 differently-sized airfoils (from Imperial Front Wing geometry) at random positions/pitch angles, constant freestream velocity, ~100k subsampled points per sample
- **Input data:** temporal coordinates `t`, spatial positions `pos`, airfoil surface indices, and 5 input velocity fields
- **Constraint:** models must satisfy no-slip boundary conditions at the airfoil surface
- **Difficulty:** capturing high-frequency turbulent components alongside low-frequency laminar flow
- **Evaluation:** accuracy of predicted 3D velocity fields vs ground truth on a held-out test set (exact metric undisclosed; `main.py` hints at L2 norm averaged over points and timesteps)

## Data Location

The default `data_dir` in `train.py` / `eval.py` is `/workspace/data/warped-ifw/` — the path on the remote training machine. On the local box the dataset lives at `/home/paul/scratch/gram-competition/warped-ifw/`; use `configs/local.yaml` (which sets `data_dir` accordingly) when running locally. `split.json` stores bare filenames, so the same split is portable across both machines. 810 `.npz` files total, ~17MB each. Each file contains one sample (unbatched) with keys:
- `t`: (10,) — temporal coordinates
- `pos`: (100000, 3) — spatial positions
- `idcs_airfoil`: (variable,) — indices into `pos` marking airfoil surface points
- `pressure`: (10, 100000) — pressure field
- `velocity_in`: (5, 100000, 3) — input velocity fields
- `velocity_out`: (5, 100000, 3) — ground truth velocity fields

## Development Environment

Use the `gram` conda environment: `conda activate gram`

## Running the Evaluation

```bash
python main.py
```

This instantiates the model, feeds dummy test data through it, and checks output shape/metric. The real test data has batch_size=95, 100k spatial points, 5 input timesteps, and 5 output timesteps.

## Architecture

- `main.py` — evaluation script that imports a model and runs inference with the expected tensor signature
- `models/__init__.py` — single import entry point; each model is re-exported here (e.g., `from .mlp import MLP`)
- `models/<model_name>/` — self-contained model directory with implementation, weights, and optional training docs

### Model Contract

Every model must:
1. Be constructable with no arguments: `Model()`
2. Load its own weights during `__init__`
3. Implement `__call__` / `forward` with signature:
   ```
   (t: [B,10], pos: [B,100k,3], idcs_airfoil: list[Tensor], velocity_in: [B,5,100k,3]) -> velocity_out: [B,5,100k,3]
   ```
   `idcs_airfoil` elements are variable-length index tensors into the `pos` dimension.

### Adding a New Model

1. Create `models/<name>/` with model class and weights
2. Add import in `models/__init__.py`
3. Update the `from models import ... as Model` line in `main.py` to use the new model
