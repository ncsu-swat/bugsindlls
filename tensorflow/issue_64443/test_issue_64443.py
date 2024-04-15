import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TextVectorization
from tensorflow.keras.models import Model
import pytest

def test_f():
    vocabulary = ['hello', 'there', 'random', 'vocab', 'check']
    length = 64

    text_input = Input(dtype=tf.string, shape=(1,), name='text_input')
    vectorize_layer = TextVectorization(vocabulary=vocabulary, output_mode='int', name='vectorization_layer',
                                        output_sequence_length=length)(text_input)
    model = Model(inputs={'text_input': text_input}, outputs=[vectorize_layer])

    input_data = np.array([['hello here and there']], dtype=str)
    model({'text_input': input_data})

    tf.saved_model.save(model, 'my_model')
    loaded_model = tf.keras.layers.TFSMLayer('my_model', call_endpoint="serving_default")
    with pytest.raises(ValueError):
        loaded_model(input_data)