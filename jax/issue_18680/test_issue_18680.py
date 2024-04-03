from jax import config
config.update("jax_debug_nans", True)
config.update('jax_disable_jit', True)

import jax.numpy as jnp
import jax
import pytest

xp = jnp.array([0.        , 0.16      , 0.35      , 0.39999998, 0.53999996,
        0.62      , 0.78999996, 0.95      , 1.25      , 1.36      ,
        1.43      , 1.5799999 , 1.77      , 1.86      , 1.9499999 ,
        2.02      , 2.09      , 2.29      , 2.52      , 2.74      ,
        2.8899999 , 3.1499999 , 3.35      , 3.4499998 , 3.59      ,
        3.8       , 3.86      , 4.06      , 4.2       , 4.46      ,
        4.62      , 4.8599997 , 4.97      , 5.15      , 5.15      ,
      ])

x = jnp.arange(0, 5.0, step=0.01)
@jax.jit
def f(x, xp):
    return jnp.searchsorted(xp, x, side="right")

# throws error only if both config flags are enabled
def test_f():
    with pytest.raises(FloatingPointError) as e_info:
        issue_no = '18680'
        print('Jax issue no.', issue_no)
        jax.print_environment_info()
        f(x, xp)
    print(e_info.value)