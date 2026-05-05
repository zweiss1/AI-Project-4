import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import random
import os

os.makedirs("visuals", exist_ok=True)

(xRef,yRef), _ = mnist.load_data() # x is image, y is label
shapedSet = xRef.reshape(xRef.shape[0],xRef.shape[1]*xRef.shape[2])
noisedSet = shapedSet/255.0

noise = np.random.normal(0, 0.1, noisedSet.shape)

noisedSet = noisedSet + noise
noisedImgs = noisedSet.reshape(xRef.shape)

model = Sequential()
model.add(Dense(xRef.shape[1]*xRef.shape[2]))
model.compile(optimizer='sgd',loss='mse')
model.fit(noisedSet,shapedSet,batch_size=32,epochs=15)

denoised = model.predict(noisedSet)

imgs = denoised.reshape(xRef.shape)

fig, axes = plt.subplots(3,10,figsize=(15,5))

for row in range(3):
    if row == 0:
        disp = xRef
        lbl = "Clean"
    elif row == 1:
        disp = noisedImgs
        lbl = "Noisy"
    else:
        disp = imgs
        lbl = "Denoised"
    axes[row,0].set_title(lbl)
    i=0
    digs = set()
    while len(digs) < 10:
        label = yRef[i]
        if label not in digs:
            ax = axes[row,label]
            ax.imshow(disp[i], cmap='gray')
            ax.axis('off')
            digs.add(label)
        i+=1
plt.savefig("visuals/basic_denoise_mnist.png", dpi=150, bbox_inches='tight')
print("Saved visuals/basic_denoise_mnist.png")
plt.show()


# xRef = noisedSet.reshape(xRef.shape[0],xRef.shape[1],xRef.shape[2])
# fig, axes = plt.subplots(1,10,figsize=(15,2))
# digs = set()

# i=0
# while len(digs) < 10:
#     label = yRef[i]
#     if label not in digs:
#         ax = axes[label]
#         ax.imshow(xRef[i], cmap='gray')
#         ax.set_title(str(label))
#         ax.axis('off')
#         digs.add(label)
#     i+=1
# plt.show()