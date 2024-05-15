import jax
import pytest
import jax.numpy as jnp
from jax._src.interpreters import mlir
from jax.lax import linalg
import jax.test_util as jtu

def test_f():
    issue_no = '19076'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    x = jnp.zeros((3, 3))
    assert x.device().platform == 'cpu'
    del mlir._platform_specific_lowerings['cpu'][linalg.lu_p]

    #
    with pytest.raises(AssertionError) as e_info:
        jtu.check_grads(jnp.linalg.det, (x,), order=1)

    
    print(e_info.value)
