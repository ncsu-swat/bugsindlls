import jax
import jax.numpy as jnp
from jax import jit
import pytest

@jit
def f(x):
    x.imag.shape

def test_f():
    issue_no = '20624'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(AttributeError) as e_info:
        f(jnp.asarray(0.0)) # AttributeError as x.imag is a float object, so it doesn't have the attribute shape.
    print(f'{e_info.type.__name__}: {e_info.value}')
