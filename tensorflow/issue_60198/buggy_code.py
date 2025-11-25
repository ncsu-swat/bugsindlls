import tensorflow as tf
with tf.device("GPU:0"):
    partial_pivoting = True
    perturb_singular = False
    diagonals = tf.complex(tf.random.uniform([2, 13, 0, 13, 857005098598382018], dtype=tf.float32, minval=-1024, maxval=1024),tf.random.uniform([2, 13, 0, 13, 857005098598382018], dtype=tf.float32, minval=-1024, maxval=1024))
    rhs = tf.complex(tf.random.uniform([15, 15, 12, 1, 10, 16], dtype=tf.float32, minval=-1024, maxval=1024),tf.random.uniform([15, 15, 12, 1, 10, 16], dtype=tf.float32, minval=-1024, maxval=1024))
    res = tf.raw_ops.TridiagonalSolve(
        partial_pivoting=partial_pivoting,
        perturb_singular=perturb_singular,
        diagonals=diagonals,
        rhs=rhs,
    )