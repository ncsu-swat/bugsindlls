import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

    from_logits = True
    axis = True  


    tf.keras.losses.categorical_crossentropy(y_true, y_pred, axis=axis)

 
    with pytest.raises(AssertionError):
        tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)
