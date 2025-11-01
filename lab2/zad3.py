import argparse
import tensorflow as tf


@tf.function
def solve_linear_system(A, b):
    A = tf.constant(A, dtype=tf.float32)
    b = tf.constant(b, dtype=tf.float32)
    if len(b.shape) == 1:
        b = tf.reshape(b, [-1, 1])
    x = tf.linalg.solve(A, b)
    return tf.reshape(x, [-1])


def main():
    parser = argparse.ArgumentParser(description="Rozwiązywanie układów równań liniowych Ax = b")
    parser.add_argument("--A", nargs="+", type=float, required=True, help="Wartości macierzy A (kolejno wierszami)")
    parser.add_argument("--b", nargs="+", type=float, required=True, help="Wartości wektora b")
    args = parser.parse_args()

    n = int(len(args.b))
    A = [args.A[i * n : (i + 1) * n] for i in range(n)]
    b = args.b

    solution = solve_linear_system(A, b)
    print(f"Rozwiązanie układu: {solution.numpy()}")


if __name__ == "__main__":
    main()
