import tensorflow as tf

# Create variable and accumulators
var = tf.Variable([[0.0, 0.0]] * 10, dtype=tf.float32)
gradient_accumulator = tf.Variable([[0.0, 0.0]] * 10, dtype=tf.float32)
gradient_squared_accumulator = tf.Variable([[0.0, 0.0]] * 10, dtype=tf.float32)

# Problematic parameters - scalar grad with 2D indices
grad = tf.constant(0.0, dtype=tf.float32)  # Scalar gradient
indices = tf.constant([0, 0], dtype=tf.int32)  # 2D indices

# Other parameters
lr = tf.constant(0.0, dtype=tf.float32)
l1 = tf.constant(0.0, dtype=tf.float32)
l2 = tf.constant(0.0, dtype=tf.float32)
global_step = tf.constant(1, dtype=tf.int64)

# This will crash
tf.raw_ops.ResourceSparseApplyAdagradDA(
    var=var.handle,
    gradient_accumulator=gradient_accumulator.handle,
    gradient_squared_accumulator=gradient_squared_accumulator.handle,
    grad=grad,
    indices=indices,
    lr=lr,
    l1=l1,
    l2=l2,
    global_step=global_step,
    use_locking=True
)