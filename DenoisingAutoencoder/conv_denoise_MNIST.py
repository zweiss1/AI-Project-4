import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape
from matplotlib import pyplot as plt
import numpy as np
import random
import os

os.makedirs("visuals", exist_ok=True)

## ** YOUR CODE HERE **

(xRef, yRef), _ = mnist.load_data()

xTrain = xRef[..., np.newaxis] / 255.0        # (60000, 28, 28, 1)
noise = np.random.normal(0, 0.4, xTrain.shape)
noisedSet = xTrain + noise

model = Sequential()
# Encoder
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
# Latent Representation
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# Decoder
model.add(Dense(28*28*32, activation='relu'))
model.add(Reshape((28, 28, 32)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='mse')
model.fit(noisedSet, xTrain, batch_size=32, epochs=10)

denoised = model.predict(noisedSet)

imgs = denoised.reshape(xRef.shape[0], xRef.shape[1], xRef.shape[2])
originals = xTrain.reshape(xRef.shape[0], xRef.shape[1], xRef.shape[2])
noisy = noisedSet.reshape(xRef.shape[0], xRef.shape[1], xRef.shape[2])

fig, axes = plt.subplots(3, 10, figsize=(15, 6))
row_titles = ["Original Images", "Input with Noise Added", "Denoised Output"]
for ax, title in zip(axes[:, 0], row_titles):
    ax.set_ylabel(title, fontsize=12, rotation=90)

digs = set()
i = 0
while len(digs) < 10:
    label = yRef[i]
    if label not in digs:
        axes[0, label].imshow(originals[i], cmap='gray')
        axes[0, label].set_title(str(label))
        axes[1, label].imshow(noisy[i], cmap='gray')
        axes[2, label].imshow(imgs[i], cmap='gray')
        for row in range(3):
            if label == 0:
                axes[row, label].set_xticks([])
                axes[row, label].set_yticks([])
            else:
                axes[row, label].axis('off')
        digs.add(label)
    i += 1

plt.tight_layout()
plt.savefig("visuals/conv_denoise_mnist.png", dpi=150, bbox_inches='tight')
print("Saved visuals/conv_denoise_mnist.png")
plt.show()
