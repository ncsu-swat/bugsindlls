import jax
import pytest

def test_f():
    issue_no = '17761'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(NotImplementedError) as e_info:
        x = jax.random.key(0).itemsize # Abstract method itemsize is Not Implemented
    print(f'{e_info.type.__name__}: {e_info.value}')
