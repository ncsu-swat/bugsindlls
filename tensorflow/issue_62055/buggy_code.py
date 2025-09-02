import tensorflow as tf 

args = {
    'axis': -1, 
    'input': tf.random.uniform(shape=[]),
    'input_max': tf.random.uniform(shape=[1,8,5]),
    'input_min': tf.random.uniform(shape=[0,5]),
    'name': 'not defined', 
    'narrow_range': True, 
    'num_bits': 2, 
    'range_given': True,
    'round_mode': 'HALF_TO_EVEN',
    'signed_input': True
}

res = tf.raw_ops.QuantizeAndDequantizeV4(**args)
print(res)