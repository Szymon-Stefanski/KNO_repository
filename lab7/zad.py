import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
import keras_tuner as kt

WINDOW_SIZE = 20
EPOCHS = 50
BATCH_SIZE = 32


def load_data(path):
    return pd.read_csv(path)["Close"].values.astype("float32")


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val), min_val, max_val


def denormalize(series_norm, min_val, max_val):
    return series_norm * (max_val - min_val) + min_val


def enrich_data(series):
    t = np.arange(len(series))
    return np.stack(
        [series, np.sin(2 * np.pi * t / 30), np.cos(2 * np.pi * t / 30)], axis=1
    )


def make_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size][0])
    return np.array(X), np.array(y)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(WINDOW_SIZE, 3)))
    model.add(layers.LSTM(units=hp.Int("units", 16, 64, 16)))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3])), loss="mse"
    )
    return model


def train_model(X, y):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=1,
        directory="tuner",
        project_name="ts",
    )
    tuner.search(
        X, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
    )
    return tuner.get_best_models(1)[0]


def predict_future(model, data, n):
    window = data[-WINDOW_SIZE:].copy()
    preds = []
    for _ in range(n):
        x = np.expand_dims(window, 0)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        t = len(data) + len(preds)
        window = np.vstack(
            [window[1:], [p, np.sin(2 * np.pi * t / 30), np.cos(2 * np.pi * t / 30)]]
        )
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    # Ustawiłem ostatnie 365 dni - kurs bitcoina przez ostatnie 12 lat się wahał i predykcje były dalekie od tego
    # ile obecnie mniej więcej jest wart
    series = load_data(args.history)[-365:]

    series_norm, min_val, max_val = normalize(series)

    enriched = enrich_data(series_norm)
    x, y = make_windows(enriched, WINDOW_SIZE)

    model = train_model(x, y)

    preds_norm = predict_future(model, enriched, args.n)

    preds = denormalize(np.array(preds_norm), min_val, max_val)

    pd.DataFrame({"prediction": preds}).to_csv(args.result, index=False)


if __name__ == "__main__":
    main()
