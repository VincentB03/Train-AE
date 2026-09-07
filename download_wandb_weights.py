#!/usr/bin/env python
"""Télécharge (et met en cache) les poids d'un run Weights & Biases pour les
modèles AE et/ou flow, dans la disposition attendue par
`pshear.utils.load_galaxy_autoencoder` / `load_flow` :

    wandb_weights/<run_id>/config.yaml
    wandb_weights/<run_id>/epoch_<n>/model_checkpoint_<n>.eqx
    wandb_weights/<run_id>/epoch_<n>/config.yaml   (config du run, "dé-wandbifié")

À lancer depuis un nœud de login (avec accès réseau) : le cache produit peut
ensuite être réutilisé tel quel sur un nœud de calcul sans réseau, où
`fetch_wandb_checkpoint` saute alors entièrement l'API WandB.

Usage :
    python download_wandb_weights.py                 # utilise le bloc CONFIG ci-dessous
    python download_wandb_weights.py --only flow
    python download_wandb_weights.py --flow-run-id 4q23te9a --flow-epoch 420
"""
import argparse
from pathlib import Path

from pshear.utils import fetch_wandb_checkpoint

# =============================================================================
# CONFIGURATION — seuls paramètres à modifier
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

# Racine du cache, relative au repo (indépendante de $SCRATCH), même convention
# que experiments/verification.py.
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
    print("Fichiers en place :")
    print(f"  config du run   : {run_root_dir / 'config.yaml'}")
    print(f"  config (epoch)  : {epoch_dir / 'config.yaml'}")
    print(f"  checkpoint      : {epoch_dir / f'model_checkpoint_{int(epoch)}.eqx'}")
    return epoch_dir


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les poids WandB (AE et/ou flow) dans wandb_weights/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=["ae", "flow", "both"],
        default="both",
        help="Quel(s) modèle(s) télécharger.",
    )
    parser.add_argument("--entity", default=WANDB_ENTITY)

    parser.add_argument("--ae-project", default=WANDB_PROJECT_AE)
    parser.add_argument("--ae-run-id", default=AE_RUN_ID)
    parser.add_argument("--ae-epoch", type=int, default=AE_EPOCH_TO_LOAD)

    parser.add_argument("--flow-project", default=WANDB_PROJECT_FLOW)
    parser.add_argument("--flow-run-id", default=FLOW_RUN_ID)
    parser.add_argument("--flow-epoch", type=int, default=FLOW_EPOCH_TO_LOAD)

    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("ae", "both"):
        if not args.ae_run_id:
            raise ValueError("Renseigne --ae-run-id (ou AE_RUN_ID dans le bloc CONFIG).")
        download(
            "Autoencoder",
            f"{args.entity}/{args.ae_project}/{args.ae_run_id}",
            args.ae_epoch,
            args.cache_dir,
        )

    if args.only in ("flow", "both"):
        if not args.flow_run_id:
            raise ValueError("Renseigne --flow-run-id (ou FLOW_RUN_ID dans le bloc CONFIG).")
        download(
            "Flow",
            f"{args.entity}/{args.flow_project}/{args.flow_run_id}",
            args.flow_epoch,
            args.cache_dir,
        )

    print(f"\nTerminé. Cache : {args.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
