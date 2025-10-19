import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)

    X = df.drop("species", axis=1).values
    y = df["species"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model(learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(epochs, batch_size, learning_rate):
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nDokładność modelu na danych testowych: {test_acc:.4f}")
    return model, history

if __name__ == "__main__":
    model, history = train_model(epochs=10, batch_size=8, learning_rate=0.001)
