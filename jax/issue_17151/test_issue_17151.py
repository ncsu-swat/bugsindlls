import jax
import numpy as np
from jax.experimental import jax2tf
import pytest

from tensorflow.python.framework.errors_impl import InvalidArgumentError

def test_f():
    issue_no = '17151'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    
    x = np.ones(4, dtype=bool)
    y = np.ones(4, dtype='int32')

    print(jax.lax.dot(x, y)) # 4
    
    with pytest.raises(InvalidArgumentError) as e_info:
        print(jax2tf.convert(jax.lax.dot)(x, y)) # InvalidArgumentError: Value for attr 'LhsT' of bool is not in the list of allowed values.
    print(f'{e_info.type.__name__}: {e_info.value}')
