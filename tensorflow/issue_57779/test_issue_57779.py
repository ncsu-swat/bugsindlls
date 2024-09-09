import tensorflow as tf
import pytest

def test_f():
    tf.print(tf.cast(0.2, tf.float64))
    tf.print(tf.math.real(tf.cast(0.2, tf.complex128)))
