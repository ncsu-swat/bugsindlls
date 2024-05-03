import tensorflow as tf
import numpy as np
import pytest
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.framework.errors_impl import InvalidArgumentError


def test_f():

    ragged_values_0_tensor = tf.convert_to_tensor(np.ones([3], dtype=str))
    ragged_values_0 = tf.identity(ragged_values_0_tensor)
    ragged_values = [ragged_values_0,]
    ragged_row_splits_0_tensor = tf.random.uniform([4], minval=-256, maxval=257, dtype=tf.int64)
    ragged_row_splits_0 = tf.identity(ragged_row_splits_0_tensor)
    ragged_row_splits = [ragged_row_splits_0,]

    sparse_indices = []
    sparse_values = []
    sparse_shape = []

    dense_inputs = []
    input_order = "R"
    hashed_output = False
    num_buckets = 0
    hash_key = 956888297470

    out_values_type = 7
    out_row_splits_type = 9

    with pytest.raises(InvalidArgumentError) as e_info:
        out = gen_ragged_array_ops.ragged_cross(
            ragged_values=ragged_values,
            ragged_row_splits=ragged_row_splits,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            sparse_shape=sparse_shape,
            dense_inputs=dense_inputs,
            input_order=input_order,
            hashed_output=hashed_output,
            num_buckets=num_buckets,
            hash_key=hash_key,
            out_values_type=out_values_type,
            out_row_splits_type=out_row_splits_type
        )
    print(e_info.value)
