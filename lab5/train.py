import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def build_dense_model(hp=None, input_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_model(hp=None, input_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def make_augmentation_layer():
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05)
    ])


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(10), CLASS_NAMES, rotation=45)
    plt.yticks(np.arange(10), CLASS_NAMES)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return cm


def train(args):
    (x_train, y_train), (x_test, y_test) = load_data()

    batch_size = args.batch_size
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)

    if args.augment:
        aug = make_augmentation_layer()
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    if args.arch == "dense":
        model = build_dense_model()
    else:
        model = build_cnn_model()

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # Zapis modelu
    model.save(args.output_model)
    print(f"Model zapisany do {args.output_model}")

    # Metryki
    hist = history.history
    metrics = {
        "loss": hist["loss"][-1],
        "accuracy": hist["accuracy"][-1],
        "val_loss": hist["val_loss"][-1],
        "val_accuracy": hist["val_accuracy"][-1],
        "architecture": args.arch,
        "augment": args.augment,
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metryki zapisane w {args.output_metrics}")

    # Macierz pomyłek
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = plot_confusion_matrix(y_test, y_pred, args.output_cm)
    metrics["confusion_matrix"] = cm.tolist()

    # Update metrics.json
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["dense", "cnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--output_model", default="fashion_model.keras")
    parser.add_argument("--output_metrics", default="metrics.json")
    parser.add_argument("--output_cm", default="confusion_matrix.png")

    args = parser.parse_args()
    train(args)
