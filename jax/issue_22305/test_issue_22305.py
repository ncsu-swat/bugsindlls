import jax
from jax import numpy as jnp
import pytest

def test_f():
    issue_no = '22305'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with jax.numpy_rank_promotion('raise'): # does not raise ValueError on catching rank promotions inside jnp.vectorize
        my_sum = jnp.vectorize(lambda x, y: x + y, signature='(n),(n)->(n)')
        my_sum(jnp.zeros((10, 10)), jnp.zeros((10,)))
