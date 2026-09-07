#!/usr/bin/env python
"""Download (and cache) the weights of a Weights & Biases run for the AE and/or
flow models, in the layout expected by `pshear.utils.load_galaxy_autoencoder` /
`load_flow`:

    wandb_weights/<run_id>/config.yaml
    wandb_weights/<run_id>/epoch_<n>/model_checkpoint_<n>.eqx
    wandb_weights/<run_id>/epoch_<n>/config.yaml   (run config, "de-wandbified")

Run this from a login node (with network access): the resulting cache can then be
reused as-is on a compute node without network, where `fetch_wandb_checkpoint`
skips the WandB API entirely.

Usage:
    python download_wandb_weights.py                 # use the CONFIG block below
    python download_wandb_weights.py --only flow
    python download_wandb_weights.py --flow-run-id 4q23te9a --flow-epoch 420
    python download_wandb_weights.py --cache-dir /path/to/other/dir   # change the download destination
"""
import argparse
from pathlib import Path

from pshear.utils import fetch_wandb_checkpoint

# =============================================================================
# CONFIGURATION — the only parameters to edit
# =============================================================================
WANDB_ENTITY = "vincentb03-imt-atlantique"

# --- Autoencoder ---
WANDB_PROJECT_AE = "Test-AE-partial-3"
AE_RUN_ID = "i1pf186a"
AE_EPOCH_TO_LOAD = 2000

# --- Flow ---
WANDB_PROJECT_FLOW = "pshear-euclid-flow"
FLOW_RUN_ID = "4q23te9a"
FLOW_EPOCH_TO_LOAD = 420

# Cache root, relative to the repo (independent of $SCRATCH), same convention as
# experiments/verification.py.
CACHE_DIR = Path(".") / "wandb_weights"
# =============================================================================


def download(name, run_path, epoch, cache_dir):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  run   : {run_path}")
    print(f"  epoch : {epoch}")
    print(f"{'=' * 60}")

    epoch_dir = fetch_wandb_checkpoint(run_path, epoch, cache_dir=cache_dir)

    run_root_dir = epoch_dir.parent
    print("Files in place:")
    print(f"  run config      : {run_root_dir / 'config.yaml'}")
    print(f"  config (epoch)  : {epoch_dir / 'config.yaml'}")
    print(f"  checkpoint      : {epoch_dir / f'model_checkpoint_{int(epoch)}.eqx'}")
    return epoch_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download WandB weights (AE and/or flow) into wandb_weights/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=["ae", "flow", "both"],
        default="both",
        help="Which model(s) to download.",
    )
    parser.add_argument("--entity", default=WANDB_ENTITY)

    parser.add_argument("--ae-project", default=WANDB_PROJECT_AE)
    parser.add_argument("--ae-run-id", default=AE_RUN_ID)
    parser.add_argument("--ae-epoch", type=int, default=AE_EPOCH_TO_LOAD)

    parser.add_argument("--flow-project", default=WANDB_PROJECT_FLOW)
    parser.add_argument("--flow-run-id", default=FLOW_RUN_ID)
    parser.add_argument("--flow-epoch", type=int, default=FLOW_EPOCH_TO_LOAD)

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Destination directory for the download; the <run_id>/epoch_<n>/ "
        "layout is created underneath it. Point verification.py at the same "
        "directory if you move it away from the default.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("ae", "both"):
        if not args.ae_run_id:
            raise ValueError("Set --ae-run-id (or AE_RUN_ID in the CONFIG block).")
        download(
            "Autoencoder",
            f"{args.entity}/{args.ae_project}/{args.ae_run_id}",
            args.ae_epoch,
            args.cache_dir,
        )

    if args.only in ("flow", "both"):
        if not args.flow_run_id:
            raise ValueError("Set --flow-run-id (or FLOW_RUN_ID in the CONFIG block).")
        download(
            "Flow",
            f"{args.entity}/{args.flow_project}/{args.flow_run_id}",
            args.flow_epoch,
            args.cache_dir,
        )

    print(f"\nDone. Cache: {args.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
