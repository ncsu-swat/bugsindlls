from jax import config
import jax.numpy as jnp
import jax
import pytest

config.update("jax_debug_nans", True)
config.update('jax_disable_jit', True)

@jax.jit
def f(x):
    z = jnp.std(x)
    return z

def test_f():
    issue_no = '22308'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    x = jnp.arange(0, 0.5, step=0.1)

    with pytest.raises(FloatingPointError) as e_info:
        f(x) # raises Floating Point Error
    print(f'{e_info.type.__name__}: {e_info.value}')
