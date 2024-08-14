import tensorflow as tf
import numpy as np
import sys
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    data=tf.constant(value=np.random.randint(0,100,size=(3, 2)), shape=(3, 2), dtype=tf.int32)
    segment_ids=3
    segment_ids = tf.convert_to_tensor(segment_ids)
    num_segments=3597855484

    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.raw_ops.UnsortedSegmentSum(data=data, segment_ids=segment_ids, num_segments=num_segments) # raises Invalid Argument Error due to integer overflow
    print(f'{e_info.type.__name__}: {e_info.value}')
