import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf
import numpy as np

def test_f():
    print(np.floor_divide(0,0))
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.experimental.numpy.floor_divide(0,0)
