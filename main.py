from google.colab import drive
from __future__ import print_function
from zipfile import ZipFile
from numpy.random import shuffle
from numpy import asarray
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
import math
from PIL import Image
import numpy as np


# basic parameters
batch_size = 32 * 3 # make sure to keep batch and test size multiple of 3
test_size = 400 * 3 
learning_rate = 1.0
gamma = 0.7
seed = 2
iterations = 15

# neural network paramaters
hidden_layers = [196]
# format is out-channels, kernel size and stride
convolution_params = [[32, 3, 1], [64, 3, 1]]
# randomly turns numbers to 0, to reduce overfitting 
dropouts = [0.25, 0.5]
# does a maxpool of size n*n on the tensor 
max_pools = [2]


drive.mount('/content/gdrive')

zip_path = '/content/gdrive/MyDrive/soml/SoML-50.zip'
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
  p = 0
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

# takes an image and tells us the right, left, upper and lower boundries
def getBounds(img):
  # reduce in size for easier searching
  reduce = cv2.resize(img, (48, 16))
  imgarr = [reduce[0:16, 0:16], reduce[0:16, 16:32], reduce[0:16, 32:48]]
  # format of bounds is r l u d
  bounds = [[0, 15, 15, 0], [0, 15, 15, 0], [0, 15, 15, 0]]
  for k in range(3):
    for i in range(16):
      for j in range(16):
        if imgarr[k][i][j] != 255:
          if i < bounds[k][2]:
            bounds[k][2] = i
          if i > bounds[k][3]:
            bounds[k][3] = i
          if j < bounds[k][1]:
            bounds[k][1] = j
          if j > bounds[k][0]:
            bounds[k][0] = j

  for k in range(3):
    bounds[k][0] = min(15, bounds[k][0] + 1)
    bounds[k][3] = min(15, bounds[k][3] + 1)
    bounds[k][1] = max(0, bounds[k][1] - 1)
    bounds[k][2] = max(0, bounds[k][2] - 1)
  return bounds

# takes an image of size x*y and makes it a square of size x*x if x > y
def makeSquare (image, x, y):
  sqsize = max(x, y)
  img = Image.fromarray(image)
  new_image = Image.new('L', (sqsize, sqsize), (255))
  box = (int((sqsize - x) / 2), int((sqsize - y) / 2))
  new_image.paste(img, box)
  return asarray(new_image)

# takes an image and makes it into a tensor of (3, 1, 32, 32)
def cropAndProcessImage(img):
  bounds = getBounds(img)
  tensors = []
  for i in range(3):
    xs = i * 128
    b = bounds[i]
    l = (b[1] * 8)
    r = (b[0] * 8) + 7
    u = (b[2] * 8)
    d = (b[3] * 8) + 7
    cropped = img[u : d, xs + l: xs + r]
    squared = makeSquare(cropped, r - l + 1, d - u + 1)
    final = cv2.resize(squared, (32, 32))
    tensor = torch.unsqueeze(ToTensor()(final), 0)
    if i == 0:
      tensors = tensor
    else:
      tensors = torch.cat((tensors, tensor),0)

  return tensors

# gets the image of the name num.jpg
def getImage(num):
  image_path = data_path + str(num) + ".jpg"
  img = cv2.imread(image_path, 0) # takes input in grayscale
  return img

# gets the images and lables for training
# returns a batch_size x 1 x 32 x 32 tensor and an array of size batch_size
def getTrainBatch():

  global pos1
  global batch_size
  global training_order

  if pos1 > 40000 - int(batch_size/3):
    pos1 = 0
    shuffle(training_order)

  image_num = training_order[pos1]
  img = getImage(image_num)
  img_data = cropAndProcessImage(img)
  lable_data = translateLables(image_num)
  for i in range(1, int(batch_size/3)):
    image_num = training_order[pos1 + i]
    img = getImage(image_num)
    temptuple = (img_data, cropAndProcessImage(img))
    img_data = torch.cat(temptuple, 0)
    lable_data = lable_data + translateLables(image_num)

  # dimensions of data are now (batch_size, 1, 32, 32)
  pos1 += int(batch_size/3)
  return (img_data, lable_data)

def translateLables(num):
  num = num - 1
  num = num % 996
  data = []
  if num % 3 == 0: # prefix
    data.append(label_map[2][num] + 10)
    data.append(label_map[0][num])
    data.append(label_map[1][num])
  elif num % 3 == 1: # postfix
    data.append(label_map[0][num])
    data.append(label_map[1][num])
    data.append(label_map[2][num] + 10)
  else : # infix
    data.append(label_map[0][num])
    data.append(label_map[2][num] + 10)
    data.append(label_map[1][num])

  return data
    

def main():
  torch.manual_seed(seed)
  np.random.seed(seed)
  makeLabels()
  data = getTrainBatch()
  
main()
