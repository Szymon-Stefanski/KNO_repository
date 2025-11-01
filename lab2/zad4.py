import argparse
import sys

import numpy as np
import tensorflow as tf


def solve_system(a, b):
    a = tf.convert_to_tensor(a, dtype=tf.float64)
    b = tf.convert_to_tensor(b, dtype=tf.float64)
    if abs(tf.linalg.det(a).numpy()) < 1e-8:
        raise ValueError("Brak unikalnego rozwiązania.")

    @tf.function
    def solver(a, b):
        return tf.linalg.solve(a, tf.reshape(b, (-1, 1)))

    x = solver(a, b)
    return tf.reshape(x, (-1,)).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--values", type=str, help="Lista liczb")
    args = p.parse_args()

    if not args.values:
        print("Podaj dane przez --values lub użyj --example")
        sys.exit(1)

    vals = [float(x) for x in args.values.replace(",", " ").split()]
    total = len(vals)
    n = int((-1 + np.sqrt(1 + 4 * total)) // 2)
    if n * n + n != total:
        print("Zła liczba wartości")
        sys.exit(2)

    A = np.array(vals[: n * n]).reshape((n, n))
    b = np.array(vals[n * n :])
    try:
        x = solve_system(A, b)
        print(x)
    except Exception as e:
        print("Błąd:", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
