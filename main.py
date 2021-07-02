
from numpy import asarray
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
_data_path = ""
_annotation_path = ""
_annotate_df = None
_annotate_dict = {}
_made_csv = 'newAnotate.csv'
_annotate_arr = []


#  parameters
_batch_size = 20*3
_test_size = 300*3
_learning_rate = 1.0
_gamma = 0.7
_seed = 1
_epochs = 5
_print_interval = 10
_first_iters = 500


_value_name_map = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "add", "sub", "multi", "div"]
_value_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class NeuralNetwork(nn.Module):

  def __init__(self, hidden_length=128, middle_conv=32):
    super(NeuralNetwork, self).__init__()
    self.convolution1 = nn.Conv2d(1, middle_conv, 3, 1)
    self.convolution2 = nn.Conv2d(middle_conv, 64, 3, 1)
    self.linear1 = nn.Linear(9216, hidden_length)
    self.linear2 = nn.Linear(hidden_length, 14)
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

# def train_all(model, device, optimizer):
  
#   global _batch_size, _annotate_arr, _annotate_dict
#   model.train()
#   _annotate_arr = list(_annotate_dict.keys())
#   random.shuffle(_annotate_arr)
#   for idx in range(int(3 * len(_annotate_arr)/_batch_size)):
#     pos = int(idx * _batch_size / 3)
#     data, labels = getRandImages(pos)
#     data = torch.cat(data, 0)
#     data, labels = data.to(device), labels.to(device)
#     optimizer.zero_grad()
#     output = model(data)
#     loss = F.nll_loss(output, labels)
#     loss.backward()
#     optimizer.step()
#     if idx % int((3 * len(_annotate_arr)/_batch_size)/25) == 0:
#       print("=", end="")
#   print("\nFinal loss = ", loss.item(), "\n")

def train(model, device, optimizer, number, possibleValues, map_to, iters):

  global _batch_size
  model.train()
  loss = None
  for idx in range(iters):
    data, labels = getIndivBatch(number, possibleValues, map_to)
    data = torch.cat(data, 0)
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    if idx % int(iters/25) == 0:
      print("=", end="")
  print("\nFinal loss = ", loss.item(), "\n")

# returns _batch_size number of images from _annotate_arr
def getRandImages(pos):

  global _annotate_arr, _batch_size, _annotate_dict
  imgs = []
  labels = []
  if pos + (_batch_size/3) >= len(_annotate_arr):
    for i in range(pos, _annotate_arr):
      data = getProcessedImg(_annotate_arr[i][0: -4])
      nums = _annotate_dict[_annotate_arr[i]]
      for j in range(3):
        temp = data[j]
        imgs.append(torch.unsqueeze(ToTensor()(temp), 0))
        labels.append(nums[j])
    for i in range(0, int(_batch_size/3) + pos - _annotate_arr):
      data = getProcessedImg(_annotate_arr[i][0: -4])
      nums = _annotate_dict[_annotate_arr[i]]
      for j in range(3):
        temp = data[j]
        imgs.append(torch.unsqueeze(ToTensor()(temp), 0))
        labels.append(nums[j])
  else:
    for i in range(pos, pos + int(_batch_size/3)):
      data = getProcessedImg(_annotate_arr[i][0: -4])
      nums = _annotate_dict[_annotate_arr[i]]
      for j in range(3):
        temp = data[j]
        imgs.append(torch.unsqueeze(ToTensor()(temp), 0))
        labels.append(nums[j])
  return imgs, torch.tensor(labels)

# returns a _batch_size number of images from the _indi_dir
# it can be limited to only get some specific values
def getIndivBatch(number, possibleValues, map_to):

  global _batch_size, _value_nums, _indi_dir, _value_name_map

  images = []
  labels = []
  for i in range(_batch_size):

    rand = random.randint(0, number - 1)
    digit_to_send = map_to[rand]
    rand = possibleValues[rand]
    labels.append(digit_to_send)
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
    data.append(cv2.imread(_proc_img_dir + "/" + name + "_" + str(i)+ ".jpg", 0))

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
  print("\n")

def resetIndivDir():

  global _annotate_dict, _indi_dir, _value_name_map, _value_nums

  for i in range(14):
    os.system("cd " + _indi_dir + "/" + _value_name_map[i] + "_data/")
    os.system("rm *.jpg")
    os.system("cd -")
    _value_nums[i] = 0

  for name in _annotate_dict.keys():
    data = getProcessedImg(name[0: -4])
    for i in range(3):
      val = _annotate_dict[name][i]
      _value_nums[val] += 1
      cv2.imwrite(_indi_dir + "/" + _value_name_map[val] + "_data/" + str(_value_nums[val]) + ".jpg", data[i])


def saveAnnotations():

  global _annotate_dict, _made_csv
  names = []
  firsts = []
  seconds = []
  thirds = []
  for name in _annotate_dict.keys():
    names.append(name)
    firsts.append(_annotate_dict[name][0])
    seconds.append(_annotate_dict[name][1])
    thirds.append(_annotate_dict[name][2])
  df = pd.DataFrame({'Image': names, 'First': firsts, 'Second': seconds, 'Third': thirds})
  df.to_csv(_made_csv)

def initialProcessing():

  global _value_name_map, _indi_dir, _proc_img_dir, _annotate_df

  # first make all the directories
  os.system("mkdir " + _indi_dir)
  os.system("mkdir " + _proc_img_dir)
  for i in range(14):
    os.system("mkdir " + _indi_dir + "/" + _value_name_map[i] + "_data")

  _annotate_df = pd.read_csv(_annotation_path)
  
  processAllImgs()

def getResult(n1, o, n2):

  if o == 10:
    return n1 + n2
  elif o == 11:
    return n1 - n2
  elif o == 12:
    return n1 * n2
  elif o == 13:
    return n1 / n2

# def makeAnnotations2(model1, model2, model3, device):

#   global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict

#   count = 0
#   print("start|    making annotations   |end")
#   print("      ", end="")
#   for idx in _annotate_df.index:
#     if idx % int(len(_annotate_df.index) / 25) == 0:
#       print("=", end="")
#     first, oper, second = None, None, None
#     result = _annotate_df['Value'][idx]
#     fix = _annotate_df['Label'][idx]
#     name = _annotate_df['Image'][idx]
#     if name in _annotate_dict:
#       count += 1
#       continue
#     images = getProcessedImg(name[0: -4])
#     if fix == "infix":
#       first, oper, second = images[0], images[1], images[2]
#     elif fix == "prefix":
#       oper, first, second = images[0], images[1], images[2]
#     else:
#       first, second, oper = images[0], images[1], images[2]
#     firstTensor = torch.unsqueeze(ToTensor()(first), 1)
#     secondTensor = torch.unsqueeze(ToTensor()(second), 1)
#     operTensor = torch.unsqueeze(ToTensor()(oper), 1)
#     predict1_1 = runSingle(model1, firstTensor, device)
#     predict1_2 = runSingle(model2, firstTensor, device)
#     predict1_3 = runSingle(model3, firstTensor, device)
#     predict2_1 = runSingle(model1, secondTensor, device)
#     predict2_2 = runSingle(model2, secondTensor, device)
#     predict2_3 = runSingle(model3, secondTensor, device)
#     if predict1_1 == predict1_2 == predict1_3 and predict2_1 == predict2_2 == predict2_3:
#       predictOper1 = runSingle(model1, operTensor, device)
#       predictOper2 = runSingle(model2, operTensor, device)
#       predictOper3 = runSingle(model3, operTensor, device)
#       if predictOper1 == predictOper2 == predictOper3 and predictOper3 >= 10:
#         if result == getResult(predict1_1, predictOper1, predict2_1):
#           count += 1
#           if fix == "infix":
#             _annotate_dict[name] = [predict1_1.item(), predictOper1.item(), predict2_1.item()]
#           elif fix == "prefix":
#             _annotate_dict[name] = [predictOper1.item(), predict1_1.item(), predict2_1.item()]
#           else:
#             _annotate_dict[name] = [predict1_1.item(), predict2_1.item(), predictOper1.item()]
#   print("\nannotations done : ", count, "/", len(_annotate_df.index))

def getInvalidCondition(n1, o, n2, r):

  if r == 0 and (o == 12 or o == 13):
    return False
  if r == 4 and n1 == 2 and n2 == 2:
    return False
  if (o == 13 or o == 12) and n1 == r and n2 == 1:
    return False
  return True

# def makeAnnotations3(model1, model2, modelOper, device):

#   global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict

#   count = 0
#   print("start|    making annotations   |end")
#   print("      ", end="")
#   for idx in _annotate_df.index:
#     if idx % int(len(_annotate_df.index) / 25) == 0:
#       print("=", end="")
#     first, oper, second = None, None, None
#     result = _annotate_df['Value'][idx]
#     fix = _annotate_df['Label'][idx]
#     name = _annotate_df['Image'][idx]
#     if name in _annotate_dict:
#       count += 1
#       continue
#     images = getProcessedImg(name[0: -4])
#     if fix == "infix":
#       first, oper, second = images[0], images[1], images[2]
#     elif fix == "prefix":
#       oper, first, second = images[0], images[1], images[2]
#     else:
#       first, second, oper = images[0], images[1], images[2]
#     firstTensor = torch.unsqueeze(ToTensor()(first), 1)
#     secondTensor = torch.unsqueeze(ToTensor()(second), 1)
#     operTensor = torch.unsqueeze(ToTensor()(oper), 1)
#     predict1_1 = runSingle(model1, firstTensor, device)
#     predict1_2 = runSingle(model2, firstTensor, device)
#     predict2_1 = runSingle(model1, secondTensor, device)
#     predict2_2 = runSingle(model2, secondTensor, device)
#     if predict1_1 == predict1_2 and predict2_1 == predict2_2:
#       predictOper = runSingle(modelOper, operTensor, device)
#       if result == getResult(predict1_1, predictOper, predict2_1):
#         count += 1
#         if fix == "infix":
#           _annotate_dict[name] = [predict1_1.item(), predictOper.item(), predict2_1.item()]
#         elif fix == "prefix":
#           _annotate_dict[name] = [predictOper.item(), predict1_1.item(), predict2_1.item()]
#         else:
#           _annotate_dict[name] = [predict1_1.item(), predict2_1.item(), predictOper.item()]
#   print("\nannotations done : ", count, "/", len(_annotate_df.index))


def makeAnnotations(model1, model2, model3, modelOper, device):

  global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict

  count = 0
  print("start|    making annotations   |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
    first, oper, second = None, None, None
    result = _annotate_df['Value'][idx]
    fix = _annotate_df['Label'][idx]
    name = _annotate_df['Image'][idx]
    if name in _annotate_dict:
      count += 1
      continue
    images = getProcessedImg(name[0: -4])
    if fix == "infix":
      first, oper, second = images[0], images[1], images[2]
    elif fix == "prefix":
      oper, first, second = images[0], images[1], images[2]
    else:
      first, second, oper = images[0], images[1], images[2]
    firstTensor = torch.unsqueeze(ToTensor()(first), 1).to(device)
    secondTensor = torch.unsqueeze(ToTensor()(second), 1).to(device)
    operTensor = torch.unsqueeze(ToTensor()(oper), 1).to(device)
    predict1_1 = runSingle(model1, firstTensor, device)
    predict1_2 = runSingle(model2, firstTensor, device)
    predict1_3 = runSingle(model3, firstTensor, device)
    predict2_1 = runSingle(model1, secondTensor, device)
    predict2_2 = runSingle(model2, secondTensor, device)
    predict2_3 = runSingle(model3, secondTensor, device)
    if predict1_1 == predict1_2 == predict1_3 and predict2_1 == predict2_2 == predict2_3:
      predictOper = runSingle(modelOper, operTensor, device)
      if result == getResult(predict1_1, predictOper, predict2_1):
        if getInvalidCondition(predict1_1, predictOper, predict2_1, result):
          count += 1
          if fix == "infix":
            _annotate_dict[name] = [predict1_1.item(), predictOper.item(), predict2_1.item()]
          elif fix == "prefix":
            _annotate_dict[name] = [predictOper.item(), predict1_1.item(), predict2_1.item()]
          else:
            _annotate_dict[name] = [predict1_1.item(), predict2_1.item(), predictOper.item()]
  print("\nannotations done : ", count, "/", len(_annotate_df.index))

def processOnesAndTwos(modelNum, modelOper, device):

  global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict
  
  print("start|   making ones and twos  |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    first, oper, second = None, None, None
    result = _annotate_df['Value'][idx]
    fix = _annotate_df['Label'][idx]
    name = _annotate_df['Image'][idx]
    if result == 18: # possible case for 2 * 9
      images = getProcessedImg(name[0: -4])
      if fix == "infix":
        first, oper, second = images[0], images[1], images[2]
      elif fix == "prefix":
        oper, first, second = images[0], images[1], images[2]
      else:
        first, second, oper = images[0], images[1], images[2]
      firstTensor = torch.unsqueeze(ToTensor()(first), 1).to(device)
      secondTensor = torch.unsqueeze(ToTensor()(second), 1).to(device)
      operTensor = torch.unsqueeze(ToTensor()(oper), 1).to(device)
      predictFirst = runSingle(modelNum, firstTensor, device)
      predictSecond = runSingle(modelNum, secondTensor, device)
      predictOper = runSingle(modelOper, operTensor, device)
      if predictOper == 12:
        if predictFirst == 9 and predictSecond != 9:
          _value_nums[2] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[2] + "_data/" + str(_value_nums[2]) + ".jpg", second)
        elif predictFirst != 9 and predictSecond == 9:
          _value_nums[2] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[2] + "_data/" + str(_value_nums[2]) + ".jpg", first)

    elif result == 5 or result == 7: # we want 1 x 5 or 1 x 7
      images = getProcessedImg(name[0: -4])
      if fix == "infix":
        first, oper, second = images[0], images[1], images[2]
      elif fix == "prefix":
        oper, first, second = images[0], images[1], images[2]
      else:
        first, second, oper = images[0], images[1], images[2]
      firstTensor = torch.unsqueeze(ToTensor()(first), 1).to(device)
      secondTensor = torch.unsqueeze(ToTensor()(second), 1).to(device)
      operTensor = torch.unsqueeze(ToTensor()(oper), 1).to(device)
      predictFirst = runSingle(modelNum, firstTensor, device)
      predictSecond = runSingle(modelNum, secondTensor, device)
      predictOper = runSingle(modelOper, operTensor, device)
      if predictOper == 12:
        if predictFirst == result and predictSecond != result:
          _value_nums[1] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[1] + "_data/" + str(_value_nums[1]) + ".jpg", second)
        elif predictFirst != result and predictSecond == result:
          _value_nums[1] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[1] + "_data/" + str(_value_nums[1]) + ".jpg", first)
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
  print("\n")
   
def processAnnotations2(model, device):

  global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict
  
  print("start|   making second batch   |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    first, oper, second = None, None, None
    result = _annotate_df['Value'][idx]
    fix = _annotate_df['Label'][idx]
    name = _annotate_df['Image'][idx]
    do_proc = True
    store_n1 = True
    n1, n2, o = -1, -1, -1 # n1 is known number, n2 is other
    if result == 27: # 3 x 9 case
      n1, n2, o = 9, 3, 12
      store_n1 = False
    elif result == 28: # 4 x 7 case
      n1, n2, o = 7, 4, 12
    elif result == 42: # 6 x 7 case
      n1, n2, o = 7, 6, 12
    elif result == 20: # 4 x 5 case
      n1, n2, o = 5, 4, 12
    elif result == 32: # 4 x 8 case
      n1, n2, o = 8, 4, 12
      store_n1 = False
    elif result == 21: # 3 x 7 case
      n1, n2, o = 7, 3, 12
      store_n1 = False
    else: 
      do_proc = False
    if do_proc:
      images = getProcessedImg(name[0: -4])
      if fix == "infix":
        first, oper, second = images[0], images[1], images[2]
      elif fix == "prefix":
        oper, first, second = images[0], images[1], images[2]
      else:
        first, second, oper = images[0], images[1], images[2]
      firstTensor = torch.unsqueeze(ToTensor()(first), 1).to(device)
      secondTensor = torch.unsqueeze(ToTensor()(second), 1).to(device)
      operTensor = torch.unsqueeze(ToTensor()(oper), 1).to(device)
      predictFirst = runSingle(model, firstTensor, device)
      predictSecond = runSingle(model, secondTensor, device)
      predictOper = runSingle(model, operTensor, device)
      if predictOper != o:
        continue
      if predictFirst == n1 and predictFirst != n2:
        if store_n1:
          _value_nums[n1] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[n1] + "_data/" + str(_value_nums[n1]) + ".jpg", first)
        _value_nums[n2] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[n2] + "_data/" + str(_value_nums[n2]) + ".jpg", second)
        _value_nums[o] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[o] + "_data/" + str(_value_nums[o]) + ".jpg", oper)
      elif predictFirst == n2 and predictFirst != n1:
        if store_n1:
          _value_nums[n1] += 1
          cv2.imwrite(_indi_dir + "/" + _value_name_map[n1] + "_data/" + str(_value_nums[n1]) + ".jpg", second)
        _value_nums[n2] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[n2] + "_data/" + str(_value_nums[n2]) + ".jpg", first)
        _value_nums[o] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[o] + "_data/" + str(_value_nums[o]) + ".jpg", oper)
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
  print("\n")

def processDivision(modelOper, modelNum, device):

  global _value_name_map, _annotate_df, _value_nums, _indi_dir, _annotation_path, _annotate_dict
  
  print("start|  making division batch  |end")
  print("      ", end="")
  for idx in _annotate_df.index:
    first, oper, second = None, None, None
    result = _annotate_df['Value'][idx]
    fix = _annotate_df['Label'][idx]
    name = _annotate_df['Image'][idx]
    if result == 1: # possible division case for num/num
      images = getProcessedImg(name[0: -4])
      if fix == "infix":
        first, oper, second = images[0], images[1], images[2]
      elif fix == "prefix":
        oper, first, second = images[0], images[1], images[2]
      else:
        first, second, oper = images[0], images[1], images[2]
      firstTensor = torch.unsqueeze(ToTensor()(first), 1).to(device)
      secondTensor = torch.unsqueeze(ToTensor()(second), 1).to(device)
      operTensor = torch.unsqueeze(ToTensor()(oper), 1).to(device)
      predictFirst = runSingle(modelNum, firstTensor, device)
      predictSecond = runSingle(modelNum, secondTensor, device)
      predictOper = runSingle(modelOper, operTensor, device)
      if predictOper == 13 and predictFirst == predictSecond:
        _value_nums[13] += 1
        cv2.imwrite(_indi_dir + "/" + _value_name_map[13] + "_data/" + str(_value_nums[13]) + ".jpg", oper)
    if idx % int(len(_annotate_df.index) / 25) == 0:
      print("=", end="")
  print("\n")


def runSingle(model, tensor, device):

  model.eval()
  model = model.to(device)
  with torch.no_grad():
    tensor.to(device)
    output = model(tensor)
    prediction = output.argmax(dim=1, keepdim=True)
    return prediction[0]

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
  print("\n")

def loadStuff():

  global _annotate_dict, _annotate_df

  path1 = '/content/drive/MyDrive/soml/newAnotate.csv'
  path2 = '/content/drive/MyDrive/soml/processed.zip'
  os.system("mkdir " + _indi_dir)
  for i in range(14):
    os.system("mkdir " + _indi_dir + "/" + _value_name_map[i] + "_data")

  temp_df = pd.read_csv(path1)

  with ZipFile(path2, 'r') as zip:
    zip.extractall()

  for idx in temp_df.index:
    name = temp_df['Image'][idx]
    f = temp_df['First'][idx]
    s = temp_df['Second'][idx]
    t = temp_df['Third'][idx]
    _annotate_dict[name] = [f, s, t]

  _annotate_df = pd.read_csv(_annotation_path)

def main():

  global _seed, _learning_rate, _gamma, _first_iters, _epochs, _data_path, _annotation_path
  
  _data_path = sys.argv[1] + "/"
  _annotation_path = sys.argv[2]
  initialProcessing()
  processAnnotations()
  device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
  torch.manual_seed(_seed)
  np.random.seed(_seed)
  random.seed(_seed)
  model1 = NeuralNetwork().to(device)
  optimizer1 = optim.Adadelta(model1.parameters(), lr=_learning_rate)
  print("start|    training first ai    |end")
  print("      ", end="")
  train(model1, device, optimizer1, 8, [0, 5, 7, 8, 9, 10, 11, 12], [0, 5, 7, 8, 9, 10, 11, 12], _first_iters)
  processAnnotations2(model1, device)
  model1 = NeuralNetwork().to(device)
  optimizer1 = optim.Adadelta(model1.parameters(), lr=_learning_rate)
  print("start|  trainning operator ai  |end")
  print("      ", end="")
  # model1 learns to identify only +,-,x rest are made to 13
  train(model1, device, optimizer1, 11, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 13, 13, 13, 13, 13, 13, 13, 10, 11, 12], (_first_iters*2))
  model2 = NeuralNetwork().to(device)
  optimizer2 = optim.Adadelta(model2.parameters(), lr=_learning_rate)
  print("start|   trainning digit ai    |end")
  print("      ", end="")
  train(model2, device, optimizer2, 5, [5, 6, 7, 8, 9], [5, 6, 7, 8, 9], (_first_iters*2))
  processDivision(model1, model2, device)
  model1 = NeuralNetwork().to(device)
  optimizer1 = optim.Adadelta(model1.parameters(), lr=_learning_rate)
  print("start| trainning operator ai no2|end")
  print("      ", end="")
  train(model1, device, optimizer1, 4, [10, 11, 12, 13], [10, 11, 12, 13], (_first_iters*3))
  # model1 now is trained pretty well in operators 
  processOnesAndTwos(model2, model1, device)
  modelOper = model1
  model1 = NeuralNetwork().to(device)
  model2 = NeuralNetwork(200, 40).to(device)
  model3 = NeuralNetwork(100,20).to(device)
  optimizer1 = optim.Adadelta(model1.parameters(), lr=_learning_rate)
  optimizer2 = optim.Adadelta(model2.parameters(), lr=_learning_rate)
  optimizer3 = optim.Adadelta(model3.parameters(), lr=_learning_rate)
  print("start|  trainning digit ai no1  |end")
  print("      ", end="")
  train(model1, device, optimizer1, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (_first_iters*2))
  print("start|  trainning digit ai no2  |end")
  print("      ", end="")
  train(model2, device, optimizer2, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (_first_iters*2))
  print("start|  trainning digit ai no3  |end")
  print("      ", end="")
  train(model3, device, optimizer3, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (_first_iters*2))
  makeAnnotations(model1, model2, model3, modelOper, device)
  model1 = NeuralNetwork().to(device)
  optimizer1 = optim.Adadelta(model1.parameters(), lr=_learning_rate)
  scheduler = StepLR(optimizer1, step_size=1, gamma=_gamma)
  resetIndivDir()
  print("start |   trainning final ai    |end")
  for epoch in range(_epochs):
    print("epoch", epoch+1, end="")
    train(model1, device, optimizer1, 14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13], (_first_iters*5))
    scheduler.step()
  saveModel = "trained_model.pt"
  torch.save(model1.state_dict(), saveModel)
  print("Your Model Has been completely trained!")

main()
