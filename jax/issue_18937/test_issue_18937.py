import jax
import pytest


def f(size):
    return jax.numpy.ones([size]) # <-- size is a float

def test_f():
    issue_no = '18937'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    # The value shouldn't be nan
    with pytest.raises(TypeError) as e_info:
        f(1.0)
    print(f'{e_info.type.__name__}: {e_info.value}')
