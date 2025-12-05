import tensorflow as tf
import pytest

def test_f():
    data = tf.complex(
        tf.random.uniform([3, 4], dtype=tf.float64),
        tf.random.uniform([3, 4], dtype=tf.float64)
    )
    segment_ids = [0, 0, 1]

    r1 = tf.math.segment_sum(data=data, segment_ids=segment_ids)
    print(f'r1: {r1}')
    assert True
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.sparse.segment_sum(
            data=data,
            indices=tf.constant([0, 1, 2]),
            segment_ids=segment_ids
        )
    print(f'{e_info.type.__name__}: {e_info.value}')