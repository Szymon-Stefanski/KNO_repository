import argparse
import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    img_array = np.array(img)
    return img_array.reshape(1, 28, 28)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()
    model = tf.keras.models.load_model("mnist_model.keras")
    img = preprocess_image(args.image)
    prediction = model.predict(img)
    print("Predicted digit:", np.argmax(prediction))
if __name__ == "__main__":
    main()
