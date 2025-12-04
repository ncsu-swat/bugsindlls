import pytest
import tensorflow as tf

def test_f():
    a = tf.random.uniform([1, 20]) # All good
    print(a)
    with pytest.raises(TypeError) as e_info:
        s = [[1,2]]
        tf.random.set_seed(s) # This will fail
    print(f'{e_info.type.__name__}: {e_info.value}')
    print(tf.add(2,3)) # OK
    with pytest.raises(AttributeError) as e_info:
        b = tf.random.uniform([1, 20]) # AttributeError: 'Context' object has no attribute '_rng'
    print(f'{e_info.type.__name__}: {e_info.value}')