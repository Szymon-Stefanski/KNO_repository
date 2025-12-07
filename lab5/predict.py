import argparse
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)

    arr = np.array(img).astype("float32")
    arr = 255.0 - arr      # NEGATYW
    arr = arr / 255.0      # normalizacja

    arr = np.expand_dims(arr, -1)   # (28,28,1)
    arr = np.expand_dims(arr, 0)    # (1,28,28,1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="ścieżka do obrazka")
    parser.add_argument("--model", default="fashion_model.keras")
    args = parser.parse_args()

    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Nie mogę wczytać modelu: {e}", file=sys.stderr)
        sys.exit(1)

    x = preprocess_image(args.image)
    preds = model.predict(x)
    cls = int(np.argmax(preds))
    prob = float(np.max(preds))

    print(f"Klasa: {CLASS_NAMES[cls]}")
    print(f"Pewność: {prob:.4f}")


if __name__ == "__main__":
    main()
