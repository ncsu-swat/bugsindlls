import pytest
import tensorflow as tf
import sys


def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    oneplus = 1 + 1e-15  # this 1.0 in float32 representation, but correctly represented in float64
    oneplus_cast32 = tf.cast(oneplus, dtype=tf.float32)  # 1.0, as expected
    oneplus_cast64 = tf.cast(oneplus, dtype=tf.float64)  # 1.0, but should be larger
    oneplus_converted64 = tf.convert_to_tensor(oneplus, dtype=tf.float64)

    oneplus_cast32_as64 = tf.cast(oneplus_cast32, dtype=tf.float64)  # casting just for the equal below to work
    # note that it is truncated due to being a float32 that is represented in a float64

    assert oneplus_cast32_as64 == oneplus_cast64  # works, but should not
    assert oneplus_converted64 != oneplus_cast64  # fails, but should work
