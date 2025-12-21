import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
LATENT_DIM = 2
EPOCHS = 100


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "img",
    label_mode=None,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


dataset = dataset.map(lambda x: (x / 255.0, x / 255.0))


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

class AutoencoderCNN(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 3)),
            data_augmentation,
            layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(latent_dim, name="latent_space")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32 * 32 * 64, activation="relu"),
            layers.Reshape((32, 32, 64)),
            layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(3, 3, padding="same", activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AutoencoderCNN(LATENT_DIM)
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.encoder.summary()
autoencoder.decoder.summary()

autoencoder.fit(
    dataset,
    epochs=EPOCHS
)

def show_reconstructions(model, dataset):
    originals = []
    reconstructions = []

    for x, _ in dataset:
        decoded = model(x)
        originals.append(x)
        reconstructions.append(decoded)

    originals = tf.concat(originals, axis=0)
    reconstructions = tf.concat(reconstructions, axis=0)

    n = originals.shape[0]

    plt.figure(figsize=(2 * n, 4))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(originals[i])
        plt.axis("off")
        if i == 0:
            plt.title("Oryginał")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i])
        plt.axis("off")
        if i == 0:
            plt.title("Rekonstrukcja")

    plt.show()



show_reconstructions(autoencoder, dataset)

autoencoder.encoder.save("model_encoder.keras")
autoencoder.decoder.save("model_decoder.keras")
