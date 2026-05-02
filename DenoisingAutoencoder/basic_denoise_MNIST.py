import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import random

(xRef,yRef), _ = mnist.load_data() # x is image, y is label
noisedSet = xRef.reshape(xRef.shape[0],xRef.shape[2]*xRef.shape[1])
noisedSet = noisedSet/255.0

noise = np.random.normal(0, 0.5, noisedSet.shape)

noisedSet = noisedSet + noise

xRef = noisedSet.reshape(xRef.shape[0],xRef.shape[1],xRef.shape[2])
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
plt.show()