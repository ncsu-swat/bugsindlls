import tensorflow as tf
import pytest

def test_f():
    res=print(tf.keras.losses.get(None))
    assert res is None
    with pytest.raises(ValueError) as e_info:
        tf.keras.metrics.get(None)
    print(f'{e_info.type.__name__}: {e_info.value}')
