import jax 
import jax.numpy as jnp
from jax.scipy.special import sph_harm as jnp_sph
from scipy.special import sph_harm
import pytest

def test_f():
    # Generate 200 3D points
    seed = 23
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    data = jax.random.normal(subkey, shape=(200,3))
    r = jnp.linalg.norm(data, ord=2, axis=1)
    phi = jnp.array(jnp.arccos(data[:,2]/r))
    theta = jnp.array(jnp.arctan2(data[:,1],data[:,0]))

    # Calculate spa_harm value of Jax and scipy
    m = 0
    n = 1
    scipy_result = sph_harm(jnp.array([m]), jnp.array([n]), theta, phi)
    jax_result = jnp_sph(jnp.array([m]), jnp.array([n]), theta, phi, n_max=n)
    assert(jnp.max(jnp.abs(scipy_result - jax_result))) == 0.8381599 # This should value should be close to 0
