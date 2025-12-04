import pytest
import tensorflow as tf

def test_f():
    seed = None

    seed_res=tf.random.set_seed(seed)
    assert True
    a = tf.random.uniform([1, 2])

    seed_res=tf.random.set_seed(seed)
    assert True
    b = tf.random.uniform([1, 2])

    assert not tf.reduce_all(tf.equal(a, b)), "Tensors are equal"
