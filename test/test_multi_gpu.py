# Checks that JAX sees the GPUs of the node and can all-reduce across them (NCCL).
# Run inside the job, with 1 task: python test/test_multi_gpu.py
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print("jax", jax.__version__, "| SLURM_NTASKS =", os.environ.get("SLURM_NTASKS"),
      "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(f"process {jax.process_index()}/{jax.process_count()}")
print("jax.devices()       :", jax.devices())
print("jax.local_devices() :", jax.local_devices())

n = jax.device_count()
mesh = Mesh(np.array(jax.devices()), ("data",))
x = jax.device_put(jnp.arange(4 * n, dtype=jnp.float32), NamedSharding(mesh, P("data")))
print("shards:", [(str(s.device), s.data.tolist()) for s in x.addressable_shards])

# summing a sharded array forces a collective between the devices
total = float(jax.jit(jnp.sum)(x))
assert total == sum(range(4 * n)), total
print(f"OK: {n} device(s), platform={jax.devices()[0].platform}, all-reduce sum={total}")
