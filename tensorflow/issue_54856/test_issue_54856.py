import pytest
import tensorflow as tf
import numpy as np

def test_f():
    numpy_res=np.floor_divide(0,0)
    print(f'numpy result: {numpy_res}')
    assert numpy_res==0
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.experimental.numpy.floor_divide(0,0)
    print(f'{e_info.type.__name__}: {e_info.value}')
