import sys

print("=== TEST DE L'ENVIRONNEMENT ===")
print(f"Python version : {sys.version.split()[0]}")

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
    
    print("\n[✓] Optax, Equinox et jax-galsim ont été importés sans erreur.")
    print(f"    - JAX version : {jax.__version__}")
    print(f"    - Numpy version : {np.__version__}")
    
    print("\n No problems detected. The environment is ready for training the autoencoder.")
    
except Exception as e:
    print(f"\n[X] Error : {e}")