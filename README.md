# Train-AE

Generative model of Euclid Q1 galaxy images, developed during an internship at [CosmoStat](https://www.cosmostat.org/) (CEA).

Two stages, trained on 64×64 postage stamps:

1. **Autoencoder**: reconstructs the galaxy. Its output is convolved with the PSF (via `jax-galsim`) before being compared with the observed image.
2. **Normalizing flow**: fitted on the latent space of the frozen autoencoder, to sample new galaxies.

The modeling code in [pshear/](pshear/) builds on prior work by Benjamin Rémy (CosmoStat). Training runs were done on the [Jean Zay](http://www.idris.fr/docs/category/jean-zay) supercomputer (IDRIS/CNRS).

## Data

Two Hugging Face datasets. Each sample contains a science image, its PSF (full and partial), a noise map and a mask.

| Dataset | Galaxies | Used by |
|---|---|---|
| [euclid-Q1-VF](https://huggingface.co/datasets/VincentB03/euclid-Q1-VF) | ~50k | `train_test.py`, `train_test_partial.py`, `evaluate_residuals.py` |
| [Euclid-Q1-postage-stamps](https://huggingface.co/datasets/VincentB03/Euclid-Q1-postage-stamps) | ~260k | `train_partial_parallel.py`, `train_flow.py`, `verification.py` |

The model was developed on **euclid-Q1-VF** (just over 50k galaxies). It was then also trained on **Euclid-Q1-postage-stamps** (about 260k galaxies), which seems to give better results.

## Repository structure

```
pshear/                       Core library (JAX / Equinox)
  galaxy.py                   Galaxy autoencoder with PSF convolution, and its losses
  nn/                         Autoencoder, flow and network blocks
  utils.py                    Checkpoint save/load, W&B checkpoint download
experiments/
  train_partial_parallel.py   Autoencoder training, multi-GPU (main script)
  train_test_partial.py       Autoencoder training, partial PSF, single GPU
  train_test.py               Autoencoder training, full PSF, single GPU
  lr_range_test.py            Learning-rate range test for train_partial_parallel.py
  train_flow.py               Flow training on the frozen autoencoder's latents
  verification.py             PQMass test: generated vs real distribution
  evaluate_residuals.py       Residual diagnostics to compare autoencoder checkpoints
download_wandb_weights.py     Pre-downloads W&B checkpoints (for offline compute nodes)
test/                         Environment and multi-GPU sanity checks
```

## Installation

JAX and PyTorch (only its `DataLoader` is used) are installed separately, with the CUDA build that matches the machine:

```
pip install -U "jax[cuda12]"
pip install torch
pip install -r requirements.txt
pip install "pqm>=0.6"            # only for verification.py
```

## Usage

Scripts are run from the repository root. Their hyperparameters are in the `CONFIG` dict at the top of each file. Runs are logged to [Weights & Biases](https://wandb.ai/), and outputs go to `$SCRATCH/pshear/cosmos/runs/` if `$SCRATCH` is set, `./runs/` otherwise.

```
python test/test_multi_gpu.py                   # check that JAX sees the GPUs
python -m experiments.train_partial_parallel    # 1. train the autoencoder
python -m experiments.train_flow                # 2. train the flow (set ae_run_dir / ae_epoch in CONFIG)
python -m experiments.verification              # 3. check the generated samples
```

### Full vs. partial PSF

- `train_test.py` uses the **full** PSF. Deconvolving with it is ill-posed, so the loss adds a total-variation term to suppress pixelization artifacts.
- `train_test_partial.py` and `train_partial_parallel.py` use the **partial** PSF, also provided in the dataset. No regularization term is needed.

### Multi-GPU

`train_partial_parallel.py` trains on all the GPUs of one node from a single process (`Mesh` + `shard_map`). Model and optimizer state are replicated, `batch_size` is the global batch split across GPUs, and gradients are averaged with `pmean`. Submit it with **one task** for the whole node (e.g. `--ntasks=1 --gres=gpu:4`).

### Evaluation

- `verification.py` uses [PQMass](https://github.com/Ciela-Institute/PQM) to test whether the flow's samples follow the real data distribution, both in latent space and in image space. Real-vs-real calibration tests serve as the reference. Figures are written to `PQM_results/`.
- `evaluate_residuals.py` compares several autoencoder checkpoints on the same test images: noise floor, residuals binned by signal-to-noise, and how many latent dimensions are used.

### Checkpoints

`download_wandb_weights.py` downloads the autoencoder and/or flow checkpoints of a W&B run into `wandb_weights/<run_id>/epoch_<n>/`. Run it on a node with network access. On an offline compute node, `fetch_wandb_checkpoint` then reads this cache without calling W&B.

```
python download_wandb_weights.py                           # runs set in the CONFIG block
python download_wandb_weights.py --only flow --flow-run-id <id> --flow-epoch <n>
python download_wandb_weights.py --cache-dir <dir>         # other destination
```

The W&B entity and run IDs are set at the top of `download_wandb_weights.py` and `verification.py`. The [galaxy-morphometrics](https://github.com/VincentB03/galaxy-morphometrics) repository reads the same `wandb_weights/` layout, so `--cache-dir` can point directly at its checkpoint directory.
