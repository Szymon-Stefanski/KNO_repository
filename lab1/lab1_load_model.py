import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.keras")
model.summary()
