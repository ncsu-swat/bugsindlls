import pytest
import tensorflow as tf
import sys
import os


def test_f():

    if sys.platform == 'darwin':
        pytest.skip('This bug is Linux-specific.')
        exit(2)

    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    with pytest.raises(TypeError) as e_info:
        p=tf.keras.utils.get_file(fname="auto-mpg.csv",
            origin="http://archive.ics.uci.edu/ml/"+
            "machine-learning-databases/auto-mpg/auto-mpg.data")
    print(e_info.value)
