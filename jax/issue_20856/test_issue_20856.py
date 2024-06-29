import jax
import jax.numpy as jnp
import pytest

def f(x):
    return jax.nn.softmax(x, axis=0)

def test_f():
    issue_no = '20856'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    x = jax.random.normal(jax.random.key(0), (3, 1, 1))
    y1 = f(x)
    y2 = jax.jit(f)(x)
    print(y1, y2, sep="\n")

    assert not jnp.allclose(y1, y2) # Output y1 and y2 do not match, due to inconsistency in function jax.nn.softmax
