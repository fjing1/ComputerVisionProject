# Perform standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from IPython.display import display
# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Run this cell. If you see a picture of a cat you're all set!
with Image.open('../data/CATS_DOGS/test/CAT/10107.jpg') as im:
    im.show()

# Create a list of image filenames
path = '../data/CATS_DOGS'
img_names = []
for folder, subfolder, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '\\' +img)
# Transformations (data augmentation)
dog = Image.open('./data/CAT_DOGS/train/DOG/14.jpg')
print(dog.size)
dog.show()
print(dog.type)

r,g,b = dog.getpixel((0,0))
print(r,g,b)

# Resize
transform = transforms.Compose([
    transforms.ToTensor()#when we load the data, transform it to tensor
])
im = transform(dog)
print(im.size())

plt.imshow(np.transpose(im.numpy(),(1,2,0)))
plt.show()

# CenterCrop
transform = transform.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224)
])
im = transform(dog)
print(im.shape)
# Random Flip

# Random Rotate

# Put all transformations together
