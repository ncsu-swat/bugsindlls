import tensorflow as tf
from copy import deepcopy
import sys
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    with pytest.raises(RuntimeError) as e_info:
        optimizer = tf.keras.optimizers.Adam()
        deepcopy(optimizer)
    print(f'{e_info.type.__name__}: {e_info.value}')
