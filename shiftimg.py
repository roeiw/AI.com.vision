from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage.transform import warp, AffineTransform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from operator import itemgetter

name=39915
image1=plt.imread('Input/training/' + str(name) +'.jpg')
print(image1.shape)
# plt.imshow(image1)
df=pd.read_csv('Input/training/train_set.csv')
df_new=df[df.name_id.eq(name)]
table = df_new.drop(columns=['name_id']).to_numpy()
# list=list(df_new)
# print(list)
for row in table:
  print(row)
  x1_vals = [row[0],row[2]]
  print(x1_vals)
  y1_vals = [row[1],row[3]]
  x2_vals = [row[2],row[4]]
  y2_vals = [row[3],row[5]]
  x3_vals = [row[4],row[6]]
  y3_vals = [row[5],row[7]]
  x4_vals = [row[6],row[0]]
  y4_vals = [row[7],row[1]]
  middlex = row[8]
  middley = row[9]
  dx=middlex-34.5
  dy=middley-55
  print(middlex)
  print(middley)
  print(dx)
  print(dy)
  transform=AffineTransform(translation=(dx,dy))
  shift_img=warp(image1,transform,mode="wrap")
  plt.imshow(shift_img)
  plt.show()