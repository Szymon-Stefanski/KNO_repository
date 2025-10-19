import tensorflow as tf


def rotate_point_tensorflow(x, y, angle_deg):

    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    angle_deg = tf.constant(angle_deg, dtype=tf.float32)

    angle_rad = angle_deg * (tf.constant(tf.experimental.numpy.pi) / 180.0)

    rotation_matrix = tf.stack(
        [
            [tf.cos(angle_rad), -tf.sin(angle_rad)],
            [tf.sin(angle_rad), tf.cos(angle_rad)],
        ]
    )

    point = tf.stack([x, y])

    rotated_point = tf.linalg.matvec(rotation_matrix, point)

    return rotated_point


rotated = rotate_point_tensorflow(1.0, 0.0, 90.0)
print("Obrót punktu (1,0) o 90°:", rotated.numpy())
