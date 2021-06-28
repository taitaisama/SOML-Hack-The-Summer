from google.colab import drive
from zipfile import ZipFile
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image, ImageOps
import numpy as np
import sys
import csv 
import os


drive.mount("/content/drive")
zip_path = "/content/drive/MyDrive/soml/NEWSolML-50.zip"
data_path = "/content/SoML-50/data/"
annotation_path = "/content/SoML-50/annotations.csv"

with ZipFile(zip_path, 'r') as zip:
  zip.extractall()


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
    tensors.append((final))

  return tensors

# gets the image of the name num.jpg
def getImage(img_dir, num):
  image_path = img_dir + num
  img = cv2.imread(image_path, 0) # takes input in grayscale
  return img

!mkdir zero_data
!mkdir multi_data
!mkdir add_data
!mkdir sub_data
!mkdir nine_data
!mkdir five_data
!mkdir seven_data
!mkdir eight_data


def processAnootations():

  multiply_dir = "/content/multi_data/"
  addition_dir = "/content/add_data/"
  subtract_dir = "/content/sub_data/"
  zeros_dir = "/content/zero_data/"
  nines_dir = "/content/nine_data/"
  fives_dir = "/content/five_data/"
  sevens_dir = "/content/seven_data/"
  eights_dir = "/content/eight_data/"

  zero_num = 0
  multi_num = 0
  add_num = 0
  sub_num = 0
  nine_num = 0
  five_num = 0
  seven_num = 0
  eight_num = 0

  with open(annotation_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      result = row[2]
      fix = row[1]
      name = row[0]
      if name == "Image":
        continue
      print(name, fix, result)
      images = cropAndProcessImage(getImage(data_path, name))
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
      if result == "81": # 9 x 9 case
        nine_num += 1
        cv2.imwrite(os.path.join(nines_dir , str(nine_num) + ".jpg"), first)
        nine_num += 1
        cv2.imwrite(os.path.join(nines_dir , str(nine_num) + ".jpg"), second)
        multi_num += 1
        cv2.imwrite(os.path.join(multiply_dir, str(multi_num) + ".jpg"), oper)
      elif result == "64": # 8 x 8 case
        eight_num += 1
        cv2.imwrite(os.path.join(eights_dir , str(eight_num) + ".jpg"), first)
        eight_num += 1
        cv2.imwrite(os.path.join(eights_dir , str(eight_num) + ".jpg"), second)
        multi_num += 1
        cv2.imwrite(os.path.join(multiply_dir, str(multi_num) + ".jpg"), oper)
      elif result == "49": # 7 x 7 case
        seven_num += 1
        cv2.imwrite(os.path.join(sevens_dir , str(seven_num) + ".jpg"), first)
        seven_num += 1
        cv2.imwrite(os.path.join(sevens_dir , str(seven_num) + ".jpg"), second)
        multi_num += 1
        cv2.imwrite(os.path.join(multiply_dir, str(multi_num) + ".jpg"), oper)
      elif result == "25": # 5 x 5 case
        five_num += 1
        cv2.imwrite(os.path.join(fives_dir , str(five_num) + ".jpg"), first)
        five_num += 1
        cv2.imwrite(os.path.join(fives_dir , str(five_num) + ".jpg"), second)
        multi_num += 1
        cv2.imwrite(os.path.join(multiply_dir, str(multi_num) + ".jpg"), oper)
      elif result == "-9":
        zero_num += 1
        cv2.imwrite(os.path.join(zeros_dir , str(zero_num) + ".jpg"), first)
        nine_num += 1
        cv2.imwrite(os.path.join(nines_dir , str(nine_num) + ".jpg"), second)
        sub_num += 1
        cv2.imwrite(os.path.join(subtract_dir, str(sub_num) + ".jpg"), oper)
      elif int(result) < 0:
        sub_num += 1
        cv2.imwrite(os.path.join(subtract_dir, str(sub_num) + ".jpg"), oper)
      elif result == "13" or result == "17":
        add_num += 1
        cv2.imwrite(os.path.join(addition_dir, str(add_num) + ".jpg"), oper)


def main():
  processAnootations()

main()

!zip -r data.zip zero_data add_data eight_data five_data multi_data nine_data seven_data sub_data
