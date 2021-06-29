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


drive.mount("/content/drive")
indi_dir = "individualDatasets"
proc_img_dir = "processedImages"
zip_path = "/content/drive/MyDrive/soml/NEWSolML-50.zip"
data_path = "/content/SoML-50/data/"
annotation_path = "/content/SoML-50/annotations.csv"
annotate_df = None

with ZipFile(zip_path, 'r') as zip:
  zip.extractall()

value_name_map = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "add", "sub", "multi", "div"]
value_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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

# gets processed images from data_path
# if it has already been processed then it gets from proc_img_dir
def getProcessedImg(name):
  global proc_img_dir
  data = []
  for i in range(1, 4):
    data.append(cv2.imread(proc_img_dir + "/" + name + "_" + str(i)+ ".jpg"))

  return data

def processAllImgs():

  global proc_img_dir, data_path, annotate_df

  for idx in annotate_df.index:
    name = annotate_df['Image'][idx]
    data = cropAndProcessImage(getImage(data_path, name))
    name = name[0: -4]
    for i in range(3):
      cv2.imwrite(proc_img_dir + "/" + name + "_"  + str(i+1) + ".jpg", data[i])

def initialProcessing():

  global value_name_map, indi_dir, proc_img_dir, annotate_df

  # first make all the directories
  os.system("mkdir " + indi_dir)
  os.system("mkdir " + proc_img_dir)
  for i in range(14):
    os.system("mkdir " + indi_dir + "/" + value_name_map[i] + "_data")
  

  processAllImgs()
  
def processAnootations():

  global value_name_map, annotate_df, value_nums, indi_dir, annotation_path

  annotate_df = pd.read_csv(annotation_path)

  for idx in annotate_df.index:
    result = annotate_df['Value'][idx]
    fix = annotate_df['Label'][idx]
    name = annotate_df['Image'][idx]
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
        oper = images[1]
        first = images[0]
        second = images[2]
      elif fix == "prefix":
        oper = images[0]
        first = images[1]
        second = images[2]
      else:
        oper = images[2]
        first = images[0]
        second = images[1]
      if do_nums:
        value_nums[f] += 1
        cv2.imwrite(indi_dir + "/" + value_name_map[f] + "_data/" + str(value_nums[f]) + ".jpg", first)
        value_nums[s] += 1
        cv2.imwrite(indi_dir + "/" + value_name_map[s] + "_data/" + str(value_nums[s]) + ".jpg", second)
      value_nums[o] += 1
      cv2.imwrite(indi_dir + "/" + value_name_map[o] + "_data/" + str(value_nums[o]) + ".jpg", oper)


def main():

  initialProcessing()
  processAnootations()

main()
