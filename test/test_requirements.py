import sys

print("=== TEST DE L'ENVIRONNEMENT ===")
print(f"Python version : {sys.version.split()[0]}")

print("\n--- TEST JAX & GPU ---")
try:
    import jax
    print("\n[✓] JAX imported successfully.")
    
    devices = jax.devices()
    print(f"[✓] JAX devices detected : {devices}")
    
    if "cpu" in str(devices[0]).lower():
        print("\n Jax uses the CPU")
    else:
        print("\n[✓] JAX is properly configured to use the GPU !")

    import optax
    import equinox
    import jax_galsim
    import numpy as np
    
    print("\n[✓] Optax, Equinox et jax-galsim imported successfully.")
    print(f"    - JAX version : {jax.__version__}")
    print(f"    - Numpy version : {np.__version__}")
    
    print("\n No problems detected. The environment is ready for training the autoencoder.")
    
except Exception as e:
    print(f"\n[X] Error : {e}")

print ("\n=== TEST API HUGGINGFACE ===")
try:
    from datasets import load_dataset
    import os
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("No API key")
    
    print("Loading Dataset from Hugging Face")
    
    dataset_name = "VincentB03/euclid-Q1-V2"
    
    ds = load_dataset(dataset_name, split="train", token=True)
    
    print(f"[✓] Hugging Face OK ")
    print(f"    - Size dataset : {len(ds)}")
    
except Exception as e:
    print(f"[X] Error Hugging Face : {e}")

print ("\n=== TEST API WANDB ===")
try:
    import wandb
    
    # Vérification de la clé API
    if not os.environ.get("WANDB_API_KEY"):
        print("[i] No WANDB_API_KEY environment variable found. Please set it to use Weights & Biases.")
    
    print("Initialization run with WandB")
    
    run = wandb.init(project="test-env-cluster", name="test-init", reinit=True)
    
    # Simulation of use of wandb
    wandb.log({"test_loss": 0.042, "test_accuracy": 0.99})
    
    print(f"[✓] WandB OK ! Run '{run.name}' initialized successfully with ID: {run.id}")
    
    wandb.finish(quiet=True)
    print("[✓] No problems detected with WandB. Run finished successfully.")
    
except Exception as e:
    print(f"[X] Error WandB : {e}")

print("\n=== FIN DU TEST ===")