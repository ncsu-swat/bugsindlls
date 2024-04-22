import tensorflow as tf

tf.raw_ops.DrawBoundingBoxesV2(
    images=tf.random.normal([1,1,1]),
    boxes=tf.random.normal([1]),
    colors=0.0,
    name=None
)
