import tensorflow as tf

with tf.device('/cpu:0'):
    try:
        input_tensor = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32, shape=[5,1,1,1])
        grad = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32, shape=[5,1,1,1])
        argmax = tf.constant([0, 25288767438848, -1099511627776, -1, 4294967295], dtype=tf.int64, shape=[5,1,1,1])
        ksize = [1, 1, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        include_batch_in_index = True
        
        result = tf.raw_ops.MaxPoolGradGradWithArgmax(
            input=input_tensor,
            grad=grad,
            argmax=argmax,
            ksize=ksize,
            strides=strides,
            padding=padding,
            include_batch_in_index=include_batch_in_index,
            name=None
        )
        print("MaxPoolGradGradWithArgmax executed successfully on CPU.")
    except Exception as e:
        print(f"Exception on CPU: {e}")

with tf.device('/gpu:0'):
    try:
        input_tensor = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32, shape=[5,1,1,1])
        grad = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32, shape=[5,1,1,1])
        argmax = tf.constant([0, 25288767438848, -1099511627776, -1, 4294967295], dtype=tf.int64, shape=[5,1,1,1])
        ksize = [1, 1, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        include_batch_in_index = True
        
        result = tf.raw_ops.MaxPoolGradGradWithArgmax(
            input=input_tensor,
            grad=grad,
            argmax=argmax,
            ksize=ksize,
            strides=strides,
            padding=padding,
            include_batch_in_index=include_batch_in_index,
            name=None
        )
        print("MaxPoolGradGradWithArgmax executed successfully on GPU.")
    except Exception as e:
        print(f"Exception on GPU: {e}")