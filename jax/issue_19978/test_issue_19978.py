from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib.xla_extension import XlaRuntimeError

import pytest

def my_fun(x):
    print(f"provided callback with {x.shape}")
    return x

@jax.jit
def my_jax_fun(x):
    res = jax.pure_callback(my_fun, 
                      jax.ShapeDtypeStruct(x.shape, x.dtype),
                      x,
                      vectorized=True)
    return res

def test_f():
    issue_no = '19978'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(XlaRuntimeError) as e_info:
        s = jnp.ones((2,3,4))
        vmap_fun = jax.vmap(my_jax_fun, in_axes=1, out_axes=1)
        vmap_fun(s)
    print(f'{e_info.type.__name__}: {e_info.value}')