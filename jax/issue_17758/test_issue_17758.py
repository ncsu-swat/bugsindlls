import jax
import pytest

def test_f():
    issue_no = '17758'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    b = jax.random.key(1)
    with pytest.raises(ValueError) as e_info:
        jax.random.wrap_key_data(jax.random.key_data(b), impl=b.impl)
    
    print(f'{e_info.type.__name__}: {e_info.value}')