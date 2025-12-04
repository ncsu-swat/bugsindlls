import tensorflow as tf
import pytest

def test_f():
    x = [0.5, 1.0, 2.0, 4.0]
    axis = 0
    exclusive = -1
    reverse = -1
    
    res_1 = tf.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)
    print(res_1) # tf.Tensor([7. 6. 4. 0.], shape=(4,), dtype=float32)
    assert list(res_1.numpy()) == [7.0, 6.0, 4.0, 0.0]
    
    with pytest.raises(TypeError) as e_info:
        res_2 = tf.raw_ops.Cumsum(x=x, axis=axis, exclusive=exclusive, reverse=reverse)
        print(res_2) # TypeError: Expected bool for argument 'exclusive' not -1.
    print(f'{e_info.type.__name__}: {e_info.value}')
