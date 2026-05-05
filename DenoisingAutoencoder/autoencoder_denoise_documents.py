 # ** YOUR STUFF HERE **
import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import os


#Load the noisy documents dataset
def load_noisy_documents(count, reverse=False, feature_only=False):
    X, Y = [], []
    c = 0
    for fname in sorted(os.listdir("Noisy_Documents/noisy/540x420/"), reverse=reverse):
        # load feature image
        x = Image.open(os.path.join("Noisy_Documents/noisy/540x420", fname)).convert("L")
        x = np.array(x).astype("float32") / 255.0

        # load label image
        y = Image.open(os.path.join("Noisy_Documents/clean/540x420", fname)).convert("L")
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
    epochs = input("number of epochs (default 10) ")
    if epochs == "":
        epochs = 10
    else:
        epochs = int(epochs)
    model = build_autoencoder()
    x, y = load_noisy_documents()
    model.fit(
        x=x,
        y=y,
        epochs=epochs,
        steps_per_epoch=12
    )
    model.save("autoencoder_540x420_70imgs_5epochs_v1.keras")
    return model
    
model = main()

features, labels = load_noisy_documents(6, reverse=True, feature_only=False)
predictions = model.predict(features)
predictionImgs = predictions.reshape(features.shape[0], features.shape[1], features.shape[2])
