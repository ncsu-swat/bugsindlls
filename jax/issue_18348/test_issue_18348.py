import jax
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="ml_dtypes.float8_e4m3b11 is deprecated.*")
import tensorflow as tf
import jax.numpy as jnp

from jax.experimental import jax2tf
from jax.experimental.export.shape_poly import InconclusiveDimensionOperation


def f(x):
    inds= jnp.array([0, 2, 4], dtype=jnp.int32)
    updates = jnp.array([100., 101., 102.], dtype=jnp.float32)
    x = x.at[jnp.array(inds)].set(updates)
    return x

def test_f():
    issue_no = '19011'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    
    b = 10
    x = tf.random.uniform((b, 5))

    with pytest.raises(InconclusiveDimensionOperation) as e_info:

        with tf.GradientTape() as tape:
            tape.watch(x)
            func_convert = jax2tf.convert(jax.vmap(f), polymorphic_shapes=['(b, _)'])
            res = func_convert(x)
            loss = tf.reduce_sum(res)
        
        grads = tape.gradient(loss, x)
    print(e_info.value)
