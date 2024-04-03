import jax
import jax.numpy as jnp
import pytest

def f(x):
    @jax.jit
    def inner(a, x):
        return a, jnp.exp(x)
    return inner(0., x)[0]

def test_f():
    issue_no = '20267'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(IndexError) as e_info:
        jax.grad(f)(1.)
    print(e_info.value)
    # IndexError: list index out of range