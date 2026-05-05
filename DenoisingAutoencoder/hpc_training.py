 # ** YOUR STUFF HERE **
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import os


#Load the noisy documents dataset
def load_noisy_documents(count, reverse=False, feature_only=False):
    X, Y = [], []
    c = 0
    for fname in sorted(os.listdir("/data/zhanglab/zweiss1/AI-Project-4/DenoisingAutoencoder/Noisy_Documents/noisy/540x420/"), reverse=reverse):
        # load feature image
        x = Image.open(os.path.join("/data/zhanglab/zweiss1/AI-Project-4/DenoisingAutoencoder/Noisy_Documents/noisy/540x420", fname)).convert("L")
        x = np.array(x).astype("float32") / 255.0

        # load label image
        y = Image.open(os.path.join("/data/zhanglab/zweiss1/AI-Project-4/DenoisingAutoencoder/Noisy_Documents/clean/540x420", fname)).convert("L")
        y = np.array(y).astype("float32") / 255.0

        # add channel dimension
        X.append(x[..., None])
        Y.append(y[..., None])
        c += 1
        #Shrink the dataset for testing
        if c > count:
            break

    X = np.array(X)
    Y = np.array(Y)
    if (feature_only): return X
    else: return X, Y

def build_autoencoder():
    model = Sequential()
    # Encoder
    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(420, 540, 1)))
    model.add(Conv2D(64, 3, strides=2, activation='relu', padding='same'))
    # Latent Representation (really just a bottleneck)
    model.add(Conv2D(128, 3, strides=2, activation='relu', padding='same'))
    # Decoder
    model.add(UpSampling2D())
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    #Output layer
    model.add(Conv2D(1, 3, activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(model, noisy_source, clean_source, epochs=10):
    model.fit(x=noisy_source, y=clean_source, epochs=10, steps_per_epoch=12, batch_size=12)


def main():
    epochs = 100
    model = build_autoencoder()
    x, y = load_noisy_documents(137)
    model.fit(
        x=x,
        y=y,
        epochs=epochs,
    )
    model.save("inferencetest_100e/autoencoder_540x420_138imgs_100epochs_v1.keras")
    return model
    
model = main()

#model = tf.keras.models.load_model("HPC_autoencoder_138imgs_5epochs_540x420.keras")
features, labels = load_noisy_documents(6, reverse=True, feature_only=False)
predictions = model.predict(features)
predictionImgs = predictions.reshape(features.shape[0], features.shape[1], features.shape[2])

# pools = [features, labels, predictionImgs]
pools = [
    np.squeeze(features, axis=-1),
    np.squeeze(labels, axis=-1),
    predictionImgs
]

for i in range(0, len(predictionImgs)):
    for t in range(0,3):
        img = pools[t][i]
        img = (img * 255).clip(0, 255).astype("uint8")
        im = Image.fromarray(img)
        im.save("./inferencetest_100e/inferencetest_img"+str(i)+"_type"+str(t)+".png")
