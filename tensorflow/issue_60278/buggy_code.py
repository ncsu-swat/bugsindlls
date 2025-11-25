import tensorflow as tf
with tf.device("CPU"):
    adjoint = False
    matrix = tf.random.uniform([0, 3, 15, 7847250026211090813, 11], dtype=tf.float64, minval=-1024, maxval=1024)
    rhs = tf.random.uniform([11], dtype=tf.float64, minval=-1024, maxval=1024)
    res = tf.raw_ops.MatrixSolve(
        adjoint=adjoint,
        matrix=matrix,
        rhs=rhs,
    )