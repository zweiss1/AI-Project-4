The files in this folder are the noised, clean, and predicted images for images outside of the training corpus
for the model trained for 5 epochs on 70 images I believe. The results are pretty blurry, so I
tried a few ways to upgrade the model; those results are stored in inferencetests. It turned out that 
training it for many more epochs on the full 138 images (leaving out 6 for inference) seemed to yield 
more clarity.