import jax
import pytest
from jax import numpy as jnp

def test_f():
    issue_no = '19150'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    p = 0.8
    n = 5
    x = jnp.linspace(-1, 10, 1000)
    xxf = jax.scipy.stats.binom.pmf(k=x, n=n, p=p)
    outside_range = (x < 0) | (x > n)
    with pytest.raises(AssertionError) as e_info:
        assert jnp.all(xxf[outside_range] == 0), "PMF is not zero outside of its range"
    print(f'{e_info.type.__name__}: {e_info.value}')
