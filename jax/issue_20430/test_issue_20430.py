import jax
import pytest
from jax import numpy as jnp
from jax.scipy.special import ndtri


def f():
    return ndtri(jnp.asarray(-1.)), ndtri(jnp.asarray(2.))

def test_f():
    issue_no = '20340'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    result1, result2 = f()
    # The value shouldn't be nan
    assert result1 != jnp.array(jnp.nan) # Should be jnp.nan rather than Array(-inf, dtype=float32)
    assert result2 != jnp.array(jnp.nan) # Should be jnp.nan rather than Array(inf, dtype=float32)
