import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    data = tf.complex(
        tf.random.uniform([3, 4], dtype=tf.float64),
        tf.random.uniform([3, 4], dtype=tf.float64)
    )
    segment_ids = [0, 0, 1]

    r1 = tf.math.segment_sum(data=data, segment_ids=segment_ids)

    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.sparse.segment_sum(
            data=data,
            indices=tf.constant([0, 1, 2]),
            segment_ids=segment_ids
        )
