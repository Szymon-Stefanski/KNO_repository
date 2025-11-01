import argparse
import tensorflow as tf


@tf.function
def rotate_point_tf(x, y, angle_deg):
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    angle_rad = angle_deg * (tf.constant(tf.experimental.numpy.pi) / 180.0)
    rotation_matrix = tf.stack(
        [
            [tf.cos(angle_rad), -tf.sin(angle_rad)],
            [tf.sin(angle_rad), tf.cos(angle_rad)],
        ]
    )
    point = tf.stack([x, y])
    return tf.linalg.matvec(rotation_matrix, point)


def main():
    parser = argparse.ArgumentParser(description="Obrót punktu (x, y) o zadany kąt.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--angle", type=float, required=True)
    args = parser.parse_args()

    rotated = rotate_point_tf(args.x, args.y, args.angle)
    print(f"Obrót punktu ({args.x}, {args.y}) o {args.angle}° → {rotated.numpy()}")


if __name__ == "__main__":
    main()
