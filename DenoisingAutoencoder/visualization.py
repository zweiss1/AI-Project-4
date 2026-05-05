import os
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array

# Create visuals folder for saving plots
os.makedirs("visuals", exist_ok=True)


def display_MNIST_samples():
    (xRef,yRef), _ =mnist.load_data()
    fig, axes = plt.subplots(1,10,figsize=(15,2))
    digs = set()

    i=0
    while len(digs) < 10:
        label = yRef[i]
        if label not in digs:
            ax = axes[label]
            ax.imshow(xRef[i], cmap='gray')
            ax.set_title(str(label))
            ax.axis('off')
            digs.add(label)
        i+=1
    plt.savefig("visuals/mnist_samples.png", dpi=150, bbox_inches='tight')
    print("Saved visuals/mnist_samples.png")
    plt.show()

def display_NoisyOffice_samples():
    cleanPath = "Noisy_Documents/clean/540x420"
    dirtPath = "Noisy_Documents/noisy/540x420"
    fig, axes = plt.subplots(3,2,figsize=(6,10))

    axes[0,0].set_title("Clean")
    axes[0,1].set_title("Noisy")
    for i in range(3):
        clean = load_img(cleanPath + "/" + str(i) + ".png")
        dirty = load_img(dirtPath + "/" + str(i) + ".png")
        axes[i,0].imshow(clean, cmap='gray')
        axes[i,1].imshow(dirty, cmap='gray')
        axes[i,0].axis('off')
        axes[i,1].axis('off')
    plt.savefig("visuals/noisy_office_samples.png", dpi=150, bbox_inches='tight')
    print("Saved visuals/noisy_office_samples.png")
    plt.show()
    

display_MNIST_samples()
display_NoisyOffice_samples()