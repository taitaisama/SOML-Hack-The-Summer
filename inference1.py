import time
from numpy import asarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image, ImageOps
import numpy as np
import sys
import pandas as pd
import os
from torchvision.transforms import ToTensor

# basic parameters
_image_names = None
_data_path = None

class NeuralNetwork(nn.Module):

  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.convolution1 = nn.Conv2d(1, 32, 3, 1)
    self.convolution2 = nn.Conv2d(32, 64, 3, 1)
    self.removeRandom1 = nn.Dropout(0.25)
    self.removeRandom2 = nn.Dropout(0.5)
    self.linear1 = nn.Linear(9216, 128)
    self.linear2 = nn.Linear(128, 14)
  
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

# takes an image and makes it into a tensor of (3, 1, 28, 28)
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
    final = cv2.resize(squared, (28, 28))
    tensors.append(ToTensor()(final))

  return tensors

# gets the image of
def getImage(path):

  img = cv2.imread(path, 0) # takes input in grayscale
  return img



def makeCsv(df):

 df.to_csv("IWantedATeammateButNoOneAskedAndImTooShy_1.csv", index=False)

def getImages(path):

  return [name for name in os.listdir(path)]

def makeDf(device, model):

  global _image_name 
  model = model.to(device)
  model.eval()
  data = []
  answers = []
  final = []
  for i in range(len(_image_name)):
    if i % 20 == 19:
      # print("data", data)
      concated_data = torch.unsqueeze(torch.cat(data, dim=0), 1).to(device)
      # print("concat", concated_data)
      output = model(concated_data)
      pred = output.argmax(dim=1, keepdim=True)
      pred = pred.tolist()
      for j in range(len(pred)):
        answers.append(pred[j][0])
      data = []
    image = getImage(_data_path + "/" + _image_name[i])
    image = cropAndProcessImage(image)
    data.append(image[0])
    data.append(image[1])
    data.append(image[2])
  concated_data = torch.unsqueeze(torch.cat(data, dim=0), 1)
  output = model(concated_data)
  pred = output.argmax(dim=1, keepdim=True)
  pred = pred.tolist()
  for j in range(len(pred)):
    answers.append(pred[j][0])
  data = []
  for i in range(int(len(answers)/3)):
    if answers[i*3] >= 10:
      final.append("prefix")
    elif answers[i*3 + 1] >= 10:
      final.append("infix")
    else:
      final.append("postfix")
  df = pd.DataFrame({'Image_Name': _image_name, "Label":final})
  return df

def getPrediction(model, tensor, device):
  
  model.eval()
  with torch.no_grad():
    tensor.to(device)
    output = model(tensor)
    prediction = output.argmax(dim=1, keepdim=True)
    return prediction


def main():
  global _image_name, _data_path
  torch.manual_seed(1)
  np.random.seed(1)
  model_name = 'modelFinal.pt'
  _data_path = sys.argv[1]
  _image_name = getImages(_data_path)
  device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load(model_name, map_location=device))
  model.eval()
  makeCsv(makeDf(device, model))

main()

