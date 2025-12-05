import tensorflow as tf
import pytest

def test_f():
    data = ['a', 'b', 'c', 'd', 'e']
    partitions = [3, -2, 2, -1, 2]  
    num_partitions = 5

    t1 = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions) # Succeed
    print(t1)
    assert True
    # dynamic_partition 
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions))
    print(f'{e_info.type.__name__}: {e_info.value}')
