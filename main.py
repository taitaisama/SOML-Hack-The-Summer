from google.colab import drive
from __future__ import print_function
from zipfile import ZipFile
from numpy.random import shuffle
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

# parameters
batch_size = 96
test_size = 1200
learning_rate = 1.0
gamma = 0.7
iterations = 100

drive.mount('/content/gdrive')

zip_path = '/content/drive/MyDrive/soml/SoML-50.zip'
annotations_path = '/content/SoML-50/annotations.csv'
data_path = '/content/SoML-50/data/'

with ZipFile(zip_path, 'r') as zip:
  zip.extractall()

# order in which the input will be read
# the size is 40000 for training data, rest is for testing
training_order = [i for i in range(1, 40001)]
shuffle(training_order)
testing_order = [i for i in range(40001, 50001)]
shuffle(testing_order)
pos1 = 0
pos2 = 0

# this holds corrusopnding lables for a given input
# the input is given in order so the lables can be extrapolated from the number of the image
# there are 996 different images, so we want image_num % 996
# label_map[0] is first num, [1] is second, [2] is operator
# info about fix can just be found by image_num % 3
label_map = [[0]*996, [0]*996, [0]*996] 

# fills the lable map array
def makeLabels():
  for i in range(10):
    for j in range(10):
      flag = 0
      if j == 0 :
        flag = 1
      elif i == 0:
        flag = 0
      elif j > i:
        flag = 1
      elif i % j != 0 :
        flag = 1
      else:
        flag = 0
      if flag == 1 :
        for k in range(9):
          label_map[0][p+k] = i
          label_map[1][p+k] = j
          label_map[2][p+k] = math.floor(k/3)
        p += 9
      else:
        for k in range(12):
          label_map[0][p+k] = i
          label_map[1][p+k] = j
          label_map[2][p+k] = math.floor(k/3)
         p += 12

# method takes a 32*32 image and centres the number/element
# this is done by finding the bounds where the image starts and ends
# ie: where the first non-white pixel is from each side
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

# gets the image of the name num.jpg
def getImage(num):
  image_path = data_path + str(num) + ".jpg"
  img = cv2.imread(img_path, 0) # takes input in grayscale
  return img

# converts image to tensor along with other operations 
# returns a 3 x 1 x 32 x 32 tensor
def processImage(img):
  reduced = cv2.resize(img, (96, 32))
  tensor = ToTensor()(reduced)
  imgs = torch.split(tensor, 32, 2)
  image_array = torch.unsqueeze(torch.unsqueeze(centre(imgs[0][0]), 0), 0)
  image_array = concat(image_array, torch.unsqueeze(torch.unsqueeze(centre(imgs[1][0]), 0), 0), 0)
  image_array = concat(image_array, torch.unsqueeze(torch.unsqueeze(centre(imgs[2][0]), 0), 0), 0)
  return image_array

# gets the images for training
# returns a batch_size x 1 x 32 x 32 tensor
def getTrainBatch():
  if pos1 > 40000 - (batch_size/3):
    pos1 = 0
    shuffle(training_order)
  for i in range(int(batch_size/3)):
    image_num = training_order(pos1 + i)

  
