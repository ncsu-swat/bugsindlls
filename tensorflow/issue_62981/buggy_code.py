import tensorflow as tf
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

seed = 62981

tf.random.set_seed(seed)
print('Using seed', seed) 

tf.raw_ops.DrawBoundingBoxesV2(
    images=tf.random.normal([1,1,1]),
    boxes=tf.random.normal([1]),
    colors=0.0,
    name=None
)
