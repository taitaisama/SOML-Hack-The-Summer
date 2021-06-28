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
from PIL import Image, ImageOps
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
from torchvision.io import read_image
import csv 

# basic parameters
batch_size = 20
test_size = 300 
learning_rate = 1.0
gamma = 0.7
seed = 1
epochs = 15
print_interval = 10

drive.mount("/content/gdrive")

zip_path = '/content/gdrive/MyDrive/Database.zip'
annotations_path = '/content/SoML-50/annotations.csv'
train_data_path = '/content/TrainData/'
test_data_path = '/content/TestData/'
# copy_path = '/content/gdrive/MyDrive/soml/created_dataset/batch1data/'
csvfile_path = 'temporary.csv'

with ZipFile(zip_path, 'r') as zip:
  zip.extractall()

# this holds corrusopnding lables for a given input
# the input is given in order so the lables can be extrapolated from the number of the image
# there are 996 different images, so we want image_num % 996
# label_map[0] is first num, [1] is second, [2] is operator
# info about fix can just be found by image_num % 3
label_map = [[0]*996, [0]*996, [0]*996] 

class DataSet(Dataset):

  def __init__(self, img_dir, start, end, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.start = start
    self.end = end

  def __len__(self):
    return (self.end-self.start)

  def __getitem__(self, idx):
    image = getImage(self.img_dir, idx + self.start)
    label = idx + self.start
    image = cropAndProcessImage(image)
    label = translateLables(label)

    return image, label


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

def train(optimizer, model, loader, device):
  model.train()
  for batch_idx, (data, label) in enumerate(loader):
    optimizer.zero_grad()
    concated_data = torch.cat(data, dim=0)
    concat_labels = torch.cat(label, dim=0)
    concat_labels = concat_labels.to(device)
    concated_data = concated_data.to(device)
    output = model(concated_data)
    loss = F.nll_loss(output, concat_labels)
    loss.backward()
    optimizer.step()
    if batch_idx % print_interval == 0:
            print('LOGS [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 batch_idx * len(data) * 3, len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))


def test(model, device, loader):
  model.eval()
  totalLoss = 0
  correct = 0
  with torch.no_grad():
    for data, label in loader:
      concated_data = torch.cat(data, dim=0)
      concat_labels = torch.cat(label, dim=0)
      concat_labels = concat_labels.to(device)
      concated_data = concated_data.to(device)
      output = model(concated_data)
      totalLoss += F.nll_loss(output, concat_labels, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(concat_labels.view_as(pred)).sum().item()

  AverageLoss = totalLoss/(test_size*3)

  # print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
  #     test_loss, correct, len(test_loader.dataset),
  #     100. * correct / (len(test_loader.dataset)*3)))
  print('\nAverage Loss = ', AverageLoss, " Accuracy = ", correct, " out of ", len(loader.dataset)*3, " = ", correct / (len(loader.dataset)*3))


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
  im_invert = ImageOps.invert(new_image)
  return asarray(im_invert)

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
    final = cv2.resize(squared, (28, 28))
    tensors.append(ToTensor()(final))

  return tensors

# gets the image of the name num.jpg
def getImage(img_dir, num):
  image_path = img_dir + str(num) + ".jpg"
  img = cv2.imread(image_path, 0) # takes input in grayscale
  return img
  
# # gets the image of the name num.jpg
# def getTestImage(num):
#   num = num + 40000
#   image_path = test_data_path + str(num) + ".jpg"
#   img = cv2.imread(image_path, 0) # takes input in grayscale
#   return img

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
  device = torch.device("cuda")
  torch.manual_seed(seed)
  np.random.seed(seed)
  makeLabels()
  trainingDataset = DataSet(train_data_path, 1, 45000)
  testingDataset = DataSet(test_data_path, 45001, 50000)
  transform=transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])
  trainLoader = DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
  testLoader = DataLoader(testingDataset, batch_size=test_size, shuffle=True, num_workers=1, pin_memory=True)
  model = NeuralNetwork().to(device)
  optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
  scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
  for epoch in range(epochs):
    train(optimizer, model, trainLoader, device)
    test(model, device, testLoader)
    scheduler.step()
    saveModel = "model" + str(epoch+1) + ".pt"
    print("Epoch = ", epoch+1)
    torch.save(model.state_dict(), saveModel)

main()
