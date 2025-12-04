import tensorflow as tf
import pytest

def test_f():
    logits = tf.random.uniform([16, 1, 10], dtype=tf.float16)
    r1 = tf.nn.softmax(logits, axis=-1)
    assert True
    
    logits_sp = tf.sparse.from_dense(logits)

    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.sparse.softmax(logits_sp)
    print(f'{e_info.type.__name__}: {e_info.value}')
