from google.colab import drive
from zipfile import ZipFile
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image, ImageOps
import numpy as np
import sys
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import random
from torch.optim.lr_scheduler import StepLR


drive.mount("/content/drive")
_indi_dir = "individualDatasets"
_proc_img_dir = "processedImages"
_zip_path = "/content/drive/MyDrive/soml/NEWSolML-50.zip"
_data_path = "/content/SoML-50/data/"
_annotation_path = "/content/SoML-50/annotations.csv"
_annotate_df = None
_annotate_dict = {}

#  parameters
_batch_size = 20*3
_test_size = 300*3
_learning_rate = 1.0
_gamma = 0.7
_seed = 1
_epochs = 15
_print_interval = 10
_first_iters =  5000

with ZipFile(_zip_path, 'r') as zip:
  zip.extractall()

_value_name_map = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "add", "sub", "multi", "div"]
_value_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class NeuralNetwork(nn.Module):

  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.convolution1 = nn.Conv2d(1, 32, 3, 1)
    self.convolution2 = nn.Conv2d(32, 64, 3, 1)
    self.linear1 = nn.Linear(9216, 128)
    self.linear2 = nn.Linear(128, 14)
    self.removeRandom1 = nn.Dropout(0.25)
    self.removeRandom2 = nn.Dropout(0.5)
  
  def forward(self, x):
    x = self.convolution1(x)
    x = F.relu(x)
    x = self.convolution2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.removeRandom1(x)
    x = torch.flatten(x, 1)
    x = self.linear1(x)
    x = F.relu(x)
    x = self.removeRandom2(x)
    x = self.linear2(x)
    output = F.log_softmax(x, dim=1)
    return output

def train(model, device, optimizer, number, possibleValues, iters):

  global _batch_size
  print("start|    training first ai    |end")
  print("      ", end="")
  model.train()
  loss = None
  for idx in range(iters):
    data, labels = getIndivBatch(number, possibleValues)
    data = torch.cat(data, 0)
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    if idx % int(iters/25) == 0:
      print("=", end="")
  print("\nFinal loss = ", loss.item())
  print("done")

# returns a _batch_size number of images from the _indi_dir
# it can be limited to only get some specific values
def getIndivBatch(number, possibleValues):

  global _batch_size, _value_nums, _indi_dir, _value_name_map

  images = []
  labels = []
  for i in range(_batch_size):

    rand = random.randint(0, number - 1)
    rand = possibleValues[rand]
    labels.append(rand)
    rand2 = random.randint(1, _value_nums[rand])
    rand = _value_name_map[rand]
    path = _indi_dir + "/" + rand + "_data/" 
    img = getImage(path, str(rand2) + ".jpg")
    images.append(torch.unsqueeze(ToTensor()(img), 0))
  
  return images, torch.tensor(labels)

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
  im_invert = ImageOps.invert(new_image)
  
  return asarray(im_invert)

# takes an image and makes it into an array of three images of size 28x28
def cropAndProcessImage(img):

  bounds = getBounds(img)
  images = []
  for i in range(3):
    xs = i * 128
    b = bounds[i]
    l = (b[1] * 8)
    r = (b[0] * 8) + 7
    u = (b[2] * 8)
    d = (b[3] * 8) + 7
    cropped = img[u : d, xs + l: xs + r]
    squared = makeSquare(cropped, r - l + 1, d - u + 1)
    final = cv2.resize(squared, (28, 28))
    images.append(final)

  return images

# gets the image 'name' in img_dir
def getImage(img_dir, name):

  image_path = img_dir + name
  img = cv2.imread(image_path, 0) # takes input in grayscale
  return img

# gets processed images from _proc_img_dir
def getProcessedImg(name):

  global _proc_img_dir
  data = []
  for i in range(1, 4):
    data.append(cv2.imread(_proc_img_dir + "/" + name + "_" + str(i)+ ".jpg"))

  return data

def processAllImgs():

  global _proc_img_dir, _data_path, _annotate_df
  print("start|    processing images    |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    name = _annotate_df['Image'][idx]
    data = cropAndProcessImage(getImage(_data_path, name))
    name = name[0: -4]
    for i in range(3):
      cv2.imwrite(_proc_img_dir + "/" + name + "_"  + str(i+1) + ".jpg", data[i])
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
  print("\ndone")

def initialProcessing():

  global _value_name_map, _indi_dir, _proc_img_dir, _annotate_df

  # first make all the directories
  os.system("mkdir " + _indi_dir)
  os.system("mkdir " + _proc_img_dir)
  for i in range(14):
    os.system("mkdir " + _indi_dir + "/" + _value_name_map[i] + "_data")

  _annotate_df = pd.read_csv(_annotation_path)
  
  processAllImgs()
  
def processAnnotations():

  global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict

  print("start|   making first batch    |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    result = _annotate_df['Value'][idx]
    fix = _annotate_df['Label'][idx]
    name = _annotate_df['Image'][idx]
    do_proc = True
    do_nums = True
    f, s, o = -1, -1, -1
    if result == 81: # 9 x 9 case
      f, s, o = 9, 9, 12
    elif result == 64: # 8 x 8 case
      f, s, o = 8, 8, 12
    elif result == 49: # 7 x 7 case
      f, s, o = 7, 7, 12
    elif result == 25: # 5 x 5 case
      f, s, o = 5, 5, 12
    elif result == -9: # 0 - 9 case
      f, s, o = 0, 9, 11
    elif result < 0: # opertor is -
      o = 11
      do_nums = False
    elif result == 13 or result == 17: # opertor is +
      o = 10
      do_nums = False
    else:
      do_proc = False
    if do_proc:
      images = getProcessedImg(name[0: -4])
      if fix == "infix":
        first, oper, second = images[0], images[1], images[2]
        if do_nums:
          _annotate_dict[name] = [f, o, s]
      elif fix == "prefix":
        oper, first, second = images[0], images[1], images[2]
        if do_nums:
          _annotate_dict[name] = [o, f, s]
      else:
        first, second, oper = images[0], images[1], images[2]
        if do_nums:
          _annotate_dict[name] = [f, s, o]
      if do_nums:
        _value_nums[f] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[f] + "_data/" + str(_value_nums[f]) + ".jpg", first)
        _value_nums[s] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[s] + "_data/" + str(_value_nums[s]) + ".jpg", second)
      _value_nums[o] += 1
      cv2.imwrite(_indi_dir + "/" + _value_name_map[o] + "_data/" + str(_value_nums[o]) + ".jpg", oper)
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
  print("\ndone")

def main():

  global _seed, _learning_rate, _gamma, _first_iters
  initialProcessing()
  processAnnotations()
  device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
  torch.manual_seed(_seed)
  np.random.seed(_seed)
  random.seed(_seed)
  model = NeuralNetwork().to(device)
  optimizer = optim.Adadelta(model.parameters(), lr=_learning_rate)
  scheduler = StepLR(optimizer, step_size=1, gamma=_gamma)
  train(model, device, optimizer, 8, [0, 5, 7, 8, 9, 10, 11, 12], _first_iters)

main()
