import pytest
import tensorflow as tf
import numpy as np
import sys

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    tf_result = tf.experimental.numpy.vander(
        [-1., -1.], N=0, increasing=False
    ).numpy()

    np_result = np.vander(
        [-1., -1.], N=0, increasing=False
    )

    assert np.array_equal(tf_result, np_result)
    