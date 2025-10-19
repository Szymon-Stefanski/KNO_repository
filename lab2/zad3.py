import tensorflow as tf


def solve_linear_system(A, b):

    A = tf.constant(A, dtype=tf.float32)
    b = tf.constant(b, dtype=tf.float32)

    if len(b.shape) == 1:
        b = tf.reshape(b, [-1, 1])

    x = tf.linalg.solve(A, b)

    return tf.reshape(x, [-1])


A = [[2.0, 1.0], [1.0, -1.0]]

b = [5.0, 1.0]

solution = solve_linear_system(A, b)

print("Zadanie 3:", solution.numpy())
