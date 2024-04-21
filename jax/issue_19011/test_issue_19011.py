import jax
from jax import numpy as jnp
from jax import lax as lax

def f(shape):
    return jax.random.uniform(jax.random.PRNGKey(0), shape, minval=-1, maxval=1)

def test_f():
    issue_no = '19011'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    test = f((2,))
    # The value shouldn't be nan
    assert not jnp.isnan(jnp.min(test.astype(jnp.float8_e4m3fn)))
    assert not jnp.isnan(jnp.max(test.astype(jnp.float8_e4m3fn)))
