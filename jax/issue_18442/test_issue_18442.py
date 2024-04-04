import jax
from jax import custom_vjp, grad
from jax.random import key
import pytest

@custom_vjp
def find_fixed_point(theta, state):
    return state

def _ffp_fwd(theta, state):
    return state, None

def _ffp_bwd(residuals, state_bar):
    assert False

find_fixed_point.defvjp(_ffp_fwd, _ffp_bwd)

def fixed_point_using_while_of_theta(theta) -> float:
    state = (8.0, key(123))
    x, _ = find_fixed_point(theta, state)
    return x
def test_f():    
    issue_no = '18442'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(ValueError) as e_info:
        grad(fixed_point_using_while_of_theta)(3.0)
    
    print(e_info.value)