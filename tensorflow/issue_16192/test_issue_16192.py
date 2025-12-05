import tensorflow as tf
import pytest

def test_f():
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    from_logits = True
    axis = True  

    loss1 = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss2 = tf.keras.losses.categorical_crossentropy(y_true, y_pred, axis=axis) # Pass
    print(f'loss1: {loss1.numpy()}, loss2: {loss2.numpy()}')
    assert True
    with pytest.raises(AssertionError) as e_info:
        loss3 = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis) #AssertionError
    print(f'{e_info.type.__name__}: {e_info.value}')