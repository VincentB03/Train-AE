import jax.random as jr


def split(key, N):
    if key is None:
        return [None for _ in range(N)]
    else:
        return jr.split(key, N)