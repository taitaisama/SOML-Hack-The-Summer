from google.colab import drive
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import cv2

def centre(image):
  r = 0
  l = 31
  u = 31
  d = 0
  for i in range(32):
    for j in range(32):
      if image[i][j] != 1:
        if i < u:
          u = i
        if i > d:
          d = i
        if j < l:
          l = j
        if j > r:
          r = j
  hor = 32 - (r - l + 1)
  ver = 32 - (d - u + 1)
  ld = int(hor/2)
  ud = int(ver/2)
  image = torch.roll(image, ld - l, 1)
  image = torch.roll(image, ud - u, 0)
  return image
                  
drive.mount('/content/gdrive')
img_path = '/content/gdrive/MyDrive/soml/testsoml/data/1.jpg'
img = cv2.imread(img_path, 0)
plt.imshow(img, cmap="gray")
plt.show()
reduced = cv2.resize(img, (96, 32))
tensor = ToTensor()(reduced)
imgs = torch.split(tensor, 32, 2)

img1 = centre(imgs[0][0])
img2 = centre(imgs[1][0])
img3 = centre(imgs[2][0])
