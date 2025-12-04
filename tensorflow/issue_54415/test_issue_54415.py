import tensorflow as tf
import pytest

def test_f():
    nbins = -16
    value_range = [0.0, 5.0]
    new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    
    # The bug
    indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=nbins)
    print("Indices:", indices.numpy())
    # Assert the bug is reproduced
    assert indices.numpy().tolist() == [0, 0, 0, 0, 0, 0]
