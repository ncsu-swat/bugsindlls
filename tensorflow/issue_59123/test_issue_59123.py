import tensorflow as tf
import pytest
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.framework.errors_impl import InvalidArgumentError

def test_f():

    with pytest.raises(InvalidArgumentError) as e_info:

        file_pattern = "/tmp/record_input_test3nvh1t09/tmp3gauzk6b/basic.*"
        file_buffer_size = -1
        file_parallelism = -1
        file_shuffle_shift_ratio = -2
        batch_size = -1
        file_random_seed = -2
        compression_type = ""

        out = gen_data_flow_ops.record_input(
            file_pattern=file_pattern,
            file_buffer_size=file_buffer_size,
            file_parallelism=file_parallelism,
            file_shuffle_shift_ratio=file_shuffle_shift_ratio,batch_size=batch_size,
            file_random_seed=file_random_seed,
            compression_type=compression_type
        )
    print(e_info.value)
