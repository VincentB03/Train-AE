import sys
import os

print("=== FULL TEST ENVIRONMENT ===")
print(f"Python version : {sys.version.split()[0]}")

# ---------------------------------------------------------
# TEST 1 : JAX & GPU
# ---------------------------------------------------------
print("\n--- TEST JAX & GPU ---")
try:
    import jax
    import jax.numpy as jnp
    print("\n[✓] JAX imported successfully.")
    
    devices = jax.devices()
    print(f"[✓] JAX devices detected : {devices}")
    
    if "cpu" in str(devices[0]).lower():
        print("\n[!] Jax uses the CPU")
    else:
        print("\n[✓] JAX is properly configured to use the GPU !")

    print("    Launching a test matrix multiplication to confirm GPU usage...")
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (5000, 5000))
    y = jnp.dot(x, x).block_until_ready()
    print("    [✓] Computation completed successfully.")
    
    # CORRECTION ICI : Utilisation de la nouvelle API JAX
    print(f"    [i] Computing device : {y.devices()}")

    import optax
    import equinox
    import jax_galsim
    import numpy as np
    
    print("\n[✓] Optax, Equinox et jax-galsim imported successfully.")
    print(f"    - JAX version : {jax.__version__}")
    print(f"    - Numpy version : {np.__version__}")
    
    print("\n No problems detected. The environment is ready for training.")
    
except Exception as e:
    print(f"\n[X] Error JAX : {e}")


print ("\n=== TEST API HUGGINGFACE (OFFLINE) ===")
try:
    from datasets import load_dataset
    
    print("Loading Dataset from local cache...")
    dataset_name = "VincentB03/euclid-Q1-V2"

    ds = load_dataset(dataset_name, split="train")
    
    print(f"[✓] Hugging Face OK , dataset loaded successfully.")
    print(f"    - Size dataset : {len(ds)}")
    
except Exception as e:
    print(f"[X] Error Hugging Face : {e}")

print ("\n=== TEST API WANDB (OFFLINE) ===")
try:
    import wandb
    
    # On vérifie si on est bien en mode hors-ligne
    if os.environ.get("WANDB_MODE") == "offline":
        print("[i] WandB is offline mode, data will be saved locally in the 'wandb/' directory.")
    else:
        print("[!] WandB is not in offline mode. Please set the environment variable 'WANDB_MODE=offline' to avoid connection issues.")
    
    print("Initialization run with WandB...")
    
    run = wandb.init(project="test-env-cluster", name="test-init", reinit=True)
    
    wandb.log({"test_loss": 0.042, "test_accuracy": 0.99})
    
    print(f"[✓] WandB OK ! Run '{run.name}' initialized successfully.")
    
    wandb.finish(quiet=True)
    print("[✓] Data saved locally in the 'wandb/' directory.")
    
except Exception as e:
    print(f"[X] Error WandB : {e}")

print("\n=== END TEST ENVIRONMENT===")