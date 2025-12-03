import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    data = ['a', 'b', 'c', 'd', 'e']
    partitions = [3, -2, 2, -1, 2]  
    num_partitions = 5

    # dynamic_partition 
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions))

    # stack_dynamic_partitions 
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)

