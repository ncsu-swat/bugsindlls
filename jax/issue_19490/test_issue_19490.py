import jax
import jax.numpy as jnp
import math

def f(x, mask):    
    return jax.nn.log_softmax(x, where=mask, initial=x[jnp.argmax(mask)])[0]

def test_f():
    issue_no = '19490'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    mask = jnp.array([True,  False])
    val1 = jax.value_and_grad(f)(jnp.array([36., 36.]), mask)
    val2 = jax.value_and_grad(f)(jnp.array([36., 10000.]), mask)    

    # Buggy behavior:
    # 0.0 [0. 0.]
    # 0.0 [ 0. nan]

    assert math.isnan(val2[1][1])