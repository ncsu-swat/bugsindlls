import pytest
import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework.errors_impl import NotFoundError

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    x = tf.constant(np.ones((5, 6)), dtype=tf.qint8)
    with pytest.raises(NotFoundError) as e_info:
        enter = tf.raw_ops.Enter(data=x, frame_name="test", is_constant=True)
    print(f'{e_info.type.__name__}: {e_info.value}')
