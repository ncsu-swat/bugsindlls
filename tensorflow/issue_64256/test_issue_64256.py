import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pytest
from tensorflow.python.framework.errors_impl import NotFoundError
import sys

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    dtype_list_approxeq_not_found_error = [
        'bfloat16', 'complex128', 'complex64', 'float16',
        'half', 'int16', 'int32', 'int64', 'int8',
        'uint16', 'uint32', 'uint64', 'uint8'
    ]
    dtype_list_erfinv_not_found_error = [
        'bfloat16', 'float16', 'half'
    ]

    for dtype in dtype_list_approxeq_not_found_error:
        x = tf.constant(np.random.randint(-50, 50, ()), dtype=dtype)
        y = tf.constant(np.random.randint(-50, 50, ()), dtype=dtype)
        with pytest.raises(NotFoundError) as e_info:
            out = tf.raw_ops.ApproximateEqual(x=x,y=y)
        print(e_info.value)

    for dtype in dtype_list_erfinv_not_found_error:
        x = tf.constant(np.random.randint(-50, 50, ()), dtype=dtype)
        with pytest.raises(NotFoundError) as e_info:
            out = tf.raw_ops.Erfinv(x=x)
        print(e_info.value)
