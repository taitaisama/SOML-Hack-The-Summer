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
import math

# basic parameters
batch_size = 32 * 3 # make sure to keep batch and test size multiple of 3
test_size = 400 * 3 
learning_rate = 1.0
gamma = 0.7
seed = 1
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
# def getBounds(img):
  
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
  img = cv2.imread(image_path, 0) # takes input in grayscale
  return img

# converts image to tensor along with other operations 
# returns a 3 x 1 x 32 x 32 tensor
def processImage(img):
  # first we reduce the quality 16 times
  reduced = cv2.resize(img, (96, 32)) 
  # creates a tensor from reduced image
  tensor = ToTensor()(reduced)
  # tensor gets split into 3 parts, now images is of form [3, 1, 32, 32]
  imgs = torch.split(tensor, 32, 2)
  # image_aray is now [1, 1, 32, 32]
  # all three images get concatinated and we get [3, 1, 32, 32]
  # the one in the second dimension is used for convolutions
  image_array = torch.unsqueeze(torch.unsqueeze(centre(imgs[0][0]), 0), 0)
  temptuple = (image_array, torch.unsqueeze(torch.unsqueeze(centre(imgs[1][0]), 0), 0))
  image_array = torch.cat(temptuple, 0)
  temptuple = (image_array, torch.unsqueeze(torch.unsqueeze(centre(imgs[2][0]), 0), 0))
  image_array = torch.cat(temptuple, 0)
  return image_array

# gets the images for training
# returns a batch_size x 1 x 32 x 32 tensor
def getTrainBatch():

  global pos1
  global batch_size
  global training_order

  if pos1 > 40000 - int(batch_size/3):
    pos1 = 0
    shuffle(training_order)

  image_num = training_order[pos1]
  img = getImage(image_num)
  data = processImage(img)
  for i in range(1, int(batch_size/3)):
    image_num = training_order[pos1 + i]
    img = getImage(image_num)
    temptuple = (data, processImage(img))
    data = torch.cat(temptuple, 0)

  # dimensions of data are now (batch_size, 1, 32, 32)
  pos1 += int(batch_size/3)
  return data

def main():
  torch.manual_seed(seed)
  makeLabels()
  data = getTrainBatch()
  print(type(data), data.size())
  for i in range(10):
    img = data[i][0]
    img = img.detach()
    plt.imshow(img, cmap="gray")
    plt.show()

main()

