import jax
import pytest

def f():
    jax.lax.abs(jax.numpy.uint32(1))

def test_f():
    issue_no = '17958'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(ValueError) as e_info:
        f()
    print(e_info.value)
    # ValueError: `abs` operation is not implemented for uint32 type.
