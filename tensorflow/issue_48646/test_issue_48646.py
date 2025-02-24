import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    model = keras.Sequential([
    keras.layers.ReLU(max_value=1, threshold=-1, negative_slope=1, input_shape=(4,))])
    x = tf.constant([[1.5, 0.5,-0.5, -1.5]])
    res = model.predict(x,steps=1)

    with pytest.raises(AssertionError) as e_info:
        print(res)
        np.testing.assert_array_equal(res, [[1, 0.5, -0.5, -0.5]])
    print(f'{e_info.type.__name__}: {e_info.value}')
