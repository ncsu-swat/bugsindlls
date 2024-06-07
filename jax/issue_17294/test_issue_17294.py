import jax
import pytest


def test_f():
    issue_no = '17294'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(TypeError) as e_info:
        jax.jacrev(jax.jit(jax.lax.pow))(jax.numpy.arange(4.0), 2)

    print(f'{e_info.type.__name__}: {e_info.value}')
