import jax
import pytest
from jax import numpy as jnp
from jaxlib.xla_extension import XlaRuntimeError

@jax.jit
def f():
  return jnp.broadcast_to(0.1, (4,))

def test_f():
    issue_no = '19334'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    # The value shouldn't be nan
    with pytest.raises(XlaRuntimeError) as e_info:
        with jax.transfer_guard('disallow'):
            f()
    print(e_info.value)
