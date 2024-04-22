import tensorflow as tf
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

tf.raw_ops.DrawBoundingBoxesV2(
    images=tf.random.normal([1,1,1]),
    boxes=tf.random.normal([1]),
    colors=0.0,
    name=None
)
