import tensorflow as tf
import pytest
import sys

def create_cascaded_model(_child_model):
    inputs = tf.keras.layers.Input((8, 8, 3), name='input_1')
    x = tf.keras.layers.Conv2D(3, 3, padding='same', activation='relu', name='conv_1')(inputs)
    x = _child_model(x)
    x = tf.keras.layers.Dense(1, name='dense_2')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    inputs = tf.keras.layers.Input((8, 8, 3), name='input_1')
    x = tf.keras.layers.Conv2D(6, 3, activation='relu', name='conv_1')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2, name='dense_1')(x)
    x = tf.keras.layers.Activation('softmax', dtype=tf.float32, name='output_1')(x)
    child_model = tf.keras.models.Model(inputs=inputs, outputs=x)

    child_model.save('temp.keras')
    loaded_child_model = tf.keras.models.load_model('temp.keras')

    model = create_cascaded_model(child_model)

    with pytest.raises(ValueError) as e_info:
        model = create_cascaded_model(loaded_child_model)
    print(e_info)
