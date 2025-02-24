import sys
import tensorflow as tf
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    with pytest.raises(ValueError) as e_info:
        y_true = tf.ragged.constant([[0, 1], [2]])
        y_pred = tf.ragged.constant([[[.9, .05, .05], [.5, .89, .6]], [[.05, .01, .94]]], ragged_rank=1, dtype=tf.float32)
        print(y_true.shape, y_pred.shape)
        print(tf.losses.SparseCategoricalCrossentropy()(y_true, y_pred))
    print(f'{e_info.type.__name__}: {e_info.value}')
