import jax
import jax.numpy as jnp
import scipy
import pytest

def f_gammainc(a, x):
    y1 = scipy.special.gammainc(a, x)
    y2 = jax.scipy.special.gammainc(a, x)
    return y1, y2

def f_gammaincc(a, x):
    y1 = scipy.special.gammaincc(a, x)
    y2 = jax.scipy.special.gammaincc(a, x)
    return y1, y2

def test_f():
    issue_no = '20507'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    y1, y2 = f_gammainc(0, 1)
    print(y1)
    print(y2)
    y3, y4 = f_gammaincc(0, 0)
    print(y3)
    print(y4)

    assert not jnp.allclose(y1, y2)
    assert not jnp.allclose(y3, y4) # Output of scipy.special.{gammainc, gammaincc} functions of jax library is inconsistent with same functions of scipy library for edge cases
