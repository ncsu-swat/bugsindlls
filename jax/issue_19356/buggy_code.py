from jax import numpy as jnp

array = jnp.zeros(10, dtype=bool)
mask = jnp.arange(array.size) < 7

assert array.any(where=mask) == jnp.any(array, where=mask)
assert array.all(where=mask) == jnp.all(array, where=mask)