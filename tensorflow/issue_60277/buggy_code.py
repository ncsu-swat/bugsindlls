import tensorflow as tf
with tf.device("CPU"):
    superdiag = tf.random.uniform([5, 13, 5, 0, 6291163412844499803, 7], dtype=tf.float64, minval=-1024, maxval=1024)
    maindiag = tf.random.uniform([], dtype=tf.float64, minval=-1024, maxval=1024)
    subdiag = tf.random.uniform([9, 3, 3], dtype=tf.float64, minval=-1024, maxval=1024)
    rhs = tf.random.uniform([0, 3, 2, 8, 12, 8], dtype=tf.float64, minval=-1024, maxval=1024)
    res = tf.raw_ops.TridiagonalMatMul(
        superdiag=superdiag,
        maindiag=maindiag,
        subdiag=subdiag,
        rhs=rhs,
    )