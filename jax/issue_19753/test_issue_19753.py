import numpy as np
import jax
from jax import numpy as jnp

def test_f():
    issue_no = '19753'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    buggy_output = np.array([1.e-30 + 0j], dtype=np.complex64)

    # proper output would be: 1e-30-1e-20j
    assert buggy_output == jnp.sin(jnp.array(1e-30-1e-20j, dtype=jnp.complex64))
