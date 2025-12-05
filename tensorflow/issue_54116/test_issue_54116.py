import tensorflow as tf
import pytest

def test_f():
    x = [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]]]

    adjusted = tf.image.adjust_brightness(x, delta=1.1)
    random_adj = tf.image.random_brightness(x, max_delta=1.1)

    assert adjusted.shape == (2, 2, 3)
    assert random_adj.shape == (2, 2, 3)
