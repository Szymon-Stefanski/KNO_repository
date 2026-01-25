import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

x_axis = np.linspace(0, 50, 1000)
y_axis = np.sin(x_axis)

n_steps = 20

X, y = [], []

for i in range(len(y_axis) - n_steps):
    X.append(y_axis[i : i + n_steps])
    y.append(y_axis[i + n_steps])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


model = Sequential()
model.add(LSTM(50, activation="tanh", input_shape=(n_steps, 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

print("TRENING")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

predictions = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.title("Przewidywanie funkcji Sinus za pomocą LSTM")
plt.plot(y_axis, label="FUNKCJA", alpha=0.5, color="green")

time_axis_test = range(split_index + n_steps, len(y_axis))

plt.plot(time_axis_test, predictions, label="PREDYKCJA LSTM", color="brown")
plt.legend()
plt.grid(True)
plt.show()
