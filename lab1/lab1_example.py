import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if os.path.exists("mnist_model.keras"):
    print("Model już istnieje. Wczytuję")
    model = tf.keras.models.load_model("mnist_model.keras")
else:
    print("Nowy model")
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=5)
    model.save("mnist_model.keras")
    print(f"Model został zapisany do pliku mnist_model.keras")
