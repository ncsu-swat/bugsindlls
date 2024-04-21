import jax
import jax.numpy as jnp
import pytest

def f():
    x = jnp.arange(10)
    z = jnp.diff(x, prepend=x[0])

def test_f():
    issue_no = '19326'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(IndexError) as e_info:
        f()
    print(e_info.value)
    # ValueError: Zero-dimensional arrays cannot be concatenated.
