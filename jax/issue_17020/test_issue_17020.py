import jax
import jax.numpy as jnp
import pytest

def f(x, y):
    if y is None:
        return x
    return x * y

f = jnp.vectorize(f, signature="(),()->()")

def test_f():
    issue_no = '17020'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    res1 = f(1., None)
    print(res1)
    assert jnp.isnan(res1) # returns NaN
    
    res2 = f(jnp.ones(6), None)
    print(res2)
    assert jnp.all(jnp.isnan(res2)) # returns array of NaNs
