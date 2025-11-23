import numpy as np
import pandas as pd
import keras_tuner as kt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

cols = [
    "Class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
    "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
    "OD280/OD315_of_diluted_wines","Proline"
]

df = pd.read_csv("wine.data.csv", header=None, names=cols)

X = df.drop(columns=["Class"]).astype("float32").values
y_int = df["Class"].astype(int).values - 1
y = to_categorical(y_int, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int
)

normalizer = layers.Normalization()
normalizer.adapt(X_train)

def baseline_model():
    m = models.Sequential([
        layers.Input(shape=(13,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    m.compile(
        optimizer=optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

base = baseline_model()
base.fit(X_train, y_train, epochs=30, verbose=0, validation_data=(X_test, y_test))
base_loss, base_acc = base.evaluate(X_test, y_test, verbose=0)

def build_model(hp):
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    units1 = hp.Int("units1", min_value=32, max_value=256, step=32)
    dropout = hp.Float("drop", min_value=0.0, max_value=0.5, step=0.1)

    model = models.Sequential([
        layers.Input(shape=(13,)),
        normalizer,
        layers.Dense(units1, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="tuner_logs",
    project_name="wine_tuning"
)

tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)

best_model = tuner.get_best_models(1)[0]
best_model.save("best_tuned_model.keras")

pred = best_model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("BASELINE ACC:", base_acc)
print("TUNED ACC:", acc)
print("CONFUSION MATRIX:\n", cm)
best_model.summary()
