import tensorflow as tf
import numpy as np

# Create resource variables with extreme values
large_val = 5.24393461e+36

var = tf.Variable([[large_val, large_val, large_val],
                   [large_val, large_val, large_val]], 
                  dtype=tf.float32, name="var")

accum = tf.Variable([[large_val, large_val, large_val],
                     [large_val, large_val, large_val]], 
                    dtype=tf.float32, name="accum")

# Scalar tensors with large values
lr = tf.constant(large_val, dtype=tf.float32)
l1 = tf.constant(large_val, dtype=tf.float32) 
l2 = tf.constant(large_val, dtype=tf.float32)

# Gradient and indices
grad = tf.constant([7.90505e+31], dtype=tf.float32)
indices = tf.constant([0], dtype=tf.int32)

# This call causes the crash
result = tf.raw_ops.ResourceSparseApplyProximalAdagrad(
    var=var.handle,
    accum=accum.handle,
    lr=lr,
    l1=l1,
    l2=l2,
    grad=grad,
    indices=indices,
    use_locking=False
)