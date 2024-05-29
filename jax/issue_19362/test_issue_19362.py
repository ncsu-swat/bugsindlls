import jax
import jax.numpy as jnp
import pytest

def f():
    x = jnp.arange(10)
    z = jnp.diff(x, prepend=x[0])

def test_f():
    issue_no = '19362'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(ValueError) as e_info:
        f()
    print(f'{e_info.type.__name__}: {e_info.value}')
    # ValueError: Zero-dimensional arrays cannot be concatenated.
