import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf

def test_f():
    splits = [-16, 4, 2, 5, 5, 7]
    
    result = tf.ragged.row_splits_to_segment_ids(splits)
    
    # The bug: splits starts with negative number, but no error is raised
    assert result is not None
