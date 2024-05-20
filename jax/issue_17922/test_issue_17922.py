import jax
from jax._src.random import _gamma_one
import jax.numpy as jnp

def test_f():
    issue_no = '17922'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    key=jax.random.wrap_key_data(jnp.array([1057748167, 1356978999], dtype='uint32'))
    result=_gamma_one(key, 0.0, log_space=True)
    print(result)
    assert jnp.isnan(result) # Gives NaN
