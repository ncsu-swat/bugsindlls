import tensorflow as tf
import numpy as np
import sys
from scipy.special import zeta

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    x = 5.
    q = -9.59183673

    with tf.device('gpu'):
        gpu_output = tf.math.zeta(x=x, q=q)

    with tf.device('cpu'): 
        cpu_output = tf.math.zeta(x=x, q=q)

    scipy_output = zeta(x,q=q)

    print("GPU:\n", gpu_output)
    print("CPU:\n", cpu_output)
    print("scipy:\n", scipy_output)

    # Assert they are NOT equal
    assert False == np.array_equal(scipy_output, gpu_output)
    assert False == np.array_equal(scipy_output, cpu_output)
    assert False == np.array_equal(gpu_output, cpu_output)