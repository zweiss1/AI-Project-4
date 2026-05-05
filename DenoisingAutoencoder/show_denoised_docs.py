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

def load_images_from_folder(folder, count=6):
    images = []
    for i, fname in enumerate(sorted(os.listdir(folder))):
        if i >= count:
            break
        img = Image.open(os.path.join(folder, fname)).convert("L")
        images.append(np.array(img))
    return images

# Load images
noisy_imgs = load_images_from_folder("denoised_document_examples/noisy", 3)
clean_imgs = load_images_from_folder("denoised_document_examples/denoised", 3)

# Plot
fig, axes = plt.subplots(2, len(noisy_imgs), figsize=(15, 5))

for i in range(len(noisy_imgs)):
    axes[0, i].imshow(noisy_imgs[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Noisy {i+1}")

    axes[1, i].imshow(clean_imgs[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f"Clean {i+1}")

axes[0, 0].set_ylabel("Noisy")
axes[1, 0].set_ylabel("Clean")

plt.tight_layout()
plt.show()