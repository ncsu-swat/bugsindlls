import jax
import pytest
import jax.numpy as jnp

def test_f():
    issue_no = '18542'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    x = jnp.arange(6).reshape(2, 3)
    mask = jnp.array([True, True])
    
    with pytest.raises(IndexError) as e_info:
        x[jnp.newaxis, mask] #boolean index did not match shape of indexed array in index 1: got (2,), expected (3,)
    print(f'{e_info.type.__name__}: {e_info.value}')
