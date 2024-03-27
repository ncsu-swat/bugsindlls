from typing import Dict
import tensorflow as tf
import pickle
import sys
import numpy as np
import pytest

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
params = [
]
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        trans = tf.transpose(inp, perm=[1, 0])
        rev = tf.reverse(trans, axis=[0, 1])
        subtract = tf.math.subtract(trans, trans)
        add = tf.add(rev, subtract)
        return add,

class Model2(tf.keras.Model):
    def __call__(self, inp):
        trans = tf.transpose(inp, perm=[1, 0])
        rev = tf.reverse(trans, axis=[0, 1])
        substract = tf.math.subtract(trans, trans)
        add = tf.add(substract, rev)
        return add,  

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    inputs = [
    tf.random.uniform(shape=[16, 16], dtype=tf.float64),
    ]
    model1 = Model1()
    model2 = Model2()
    device = "gpu"
    with tf.device(device):
        tf.config.run_functions_eagerly(True)
        out1 = model1(*inputs)
        out2 = model2(*inputs)
        print(f'=========eager_output(version:{tf.__version__})================')
        
        for i in range(min(len(out1),len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
        print("XLA_eager does not trigger assertion")
        
        with pytest.raises(AssertionError) as e_info:
            tf.config.run_functions_eagerly(False)
            out1 = model1(*inputs)
            out2 = model2(*inputs)
            print(f'=========compiled_output(version:{tf.__version__})================')
            for i in range(min(len(out1),len(out2))):
                    np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
        
        print("XLA_complie triggers assertion")
        print(e_info.value)
            