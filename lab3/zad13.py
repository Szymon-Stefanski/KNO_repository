import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
CSV_LOCAL = "wine.data.csv"
LOG_DIR = "logs/wine"
BEST_MODEL_PATH = "best_model.h5"
SCALER_PATH = "scaler.pkl"

epochs = 80
batch_size = 16
learning_rate = 0.001
test_size = 0.2

if not os.path.exists(CSV_LOCAL):
    print("Pobieram CSV lokalnie...")
    r = requests.get(URL)
    r.raise_for_status()
    with open(CSV_LOCAL, "wb") as f:
        f.write(r.content)
    print("Pobrano ->", CSV_LOCAL)

columns = [
    "Class",
    "Alcohol",
    "Malic_acid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280/OD315_of_diluted_wines",
    "Proline",
]
df = pd.read_csv(CSV_LOCAL, header=None, names=columns)

df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

X = df.drop(columns=["Class"]).values.astype("float32")
y_int = df["Class"].values.astype(int) - 1  # z 1-3 -> 0-2
y = to_categorical(y_int, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y_int
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)


def first_model(input_shape=(13,)):
    model = models.Sequential(name="First_Model")
    model.add(layers.Input(shape=input_shape, name="input"))
    model.add(layers.Dense(64, activation="relu", name="dense1_relu"))
    model.add(layers.Dense(32, activation="relu", name="dense2_relu"))
    model.add(layers.Dense(3, activation="softmax", name="output_softmax"))
    return model


def second_model(input_shape=(13,)):
    model = models.Sequential(name="Second_Model")
    model.add(layers.Input(shape=input_shape, name="input"))
    model.add(
        layers.Dense(
            128, activation="elu", kernel_initializer="he_normal", name="dense1_elu"
        )
    )
    model.add(layers.Dense(64, activation="elu", name="dense2_elu"))
    model.add(layers.Dropout(0.3, name="dropout"))
    model.add(layers.Dense(3, activation="softmax", name="output_softmax"))
    return model


model_a = first_model()
model_b = second_model()

opt_a = optimizers.Adam(learning_rate=learning_rate)
opt_b = optimizers.Adam(learning_rate=learning_rate * 0.5)

model_a.compile(optimizer=opt_a, loss="categorical_crossentropy", metrics=["accuracy"])
model_b.compile(optimizer=opt_b, loss="categorical_crossentropy", metrics=["accuracy"])

# callbacks: early stopping + tensorboard
es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
tb_a = callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, "model_a"), histogram_freq=1)
tb_b = callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, "model_b"), histogram_freq=1)

hist_a = model_a.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es, tb_a],
    verbose=2,
)

hist_b = model_b.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es, tb_b],
    verbose=2,
)

loss_a, acc_a = model_a.evaluate(X_test, y_test, verbose=0)
loss_b, acc_b = model_b.evaluate(X_test, y_test, verbose=0)
print(f"First Model: loss={loss_a:.4f}, acc={acc_a:.4f}")
print(f"Second Model: loss={loss_b:.4f}, acc={acc_b:.4f}")

best_model = model_a if acc_a >= acc_b else model_b
print("Wybrano najlepszy model:", best_model.name)
best_model.save(BEST_MODEL_PATH)


def plot_history(histories):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, h in histories.items():
        plt.plot(h["loss"], label=f"{name} train")
        plt.plot(h["val_loss"], linestyle="--", label=f"{name} val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for name, h in histories.items():
        plt.plot(h["accuracy"], label=f"{name} train")
        plt.plot(h["val_accuracy"], linestyle="--", label=f"{name} val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_history({"First": hist_a.history, "Second": hist_b.history})

parser = argparse.ArgumentParser(description="Train or predict with wine model")
parser.add_argument(
    "--predict", action="store_true", help="Run prediction using saved model"
)
parser.add_argument(
    "--features", type=float, nargs=13, help="13 cech wina (podaj 13 wartości)"
)
parser.add_argument("--model_path", type=str, default=BEST_MODEL_PATH)
args = parser.parse_args()

if args.predict:
    if not args.features:
        raise SystemExit("Aby robić predykcję podaj --features (13 liczb).")
    model = models.load_model(args.model_path)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    x = np.array(args.features, dtype=np.float32).reshape(1, -1)
    x = scaler.transform(x)
    probs = model.predict(x)
    pred_class = int(np.argmax(probs, axis=1)[0]) + 1  # 1..3
    print("Predykowana klasa:", pred_class)
    print("Prawdopodobieństwa:", probs[0].tolist())
