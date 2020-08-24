from typing import Any, Union

import matplotlib.image as mpimg
from math import floor
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import numpy as np
import random
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame, Series
from pandas.core.arrays import ExtensionArray
import statistics

name=12886
image1=plt.imread('C:/Users/roei.w/Desktop/machine/new_data/tr/' + str(name) +'.jpg')
plt.figure(0)
plt.imshow(image1)
df1=pd.read_csv('C:/Users/roei.w/Desktop/machine/train.csv')
df2=pd.read_csv('C:/Users/roei.w/Desktop/machine/new_data/tr/out.csv')
df1=df1[df1.image_id.eq(int(name))]
# df1=df1.drop(columns=image_id)
# print(df1)
tags=list(df1['tag_id'])
df2=df2[df2['name_id'].isin(tags)]
print(df2)
# print(tags)
# table = df_new.drop(columns=['name_id']).to_numpy()
# print(table)
# list=list(df_new)
# print(list)
i=0
images=[]
for row in df2.iterrows():
  row=list(row)
  # print(row)
  # print(item)
  # print(type(item))
  # row=df2.loc[df2['name_id']==int(item)]
  # row=list(row.drop(columns='name_id'))

  # row=row.to_numpy()
  # print(row)
  # print(row)
  # print(row[0])
  # print(row[1])
  print(row[1][0])
  row1=row[1]

  x1_vals = [row1[1],row1[3]]
  y1_vals = [row1[2],row1[4]]
  x2_vals = [row1[3],row1[5]]
  y2_vals = [row1[4],row1[6]]
  x3_vals = [row1[5],row1[7]]
  y3_vals = [row1[6],row1[8]]
  x4_vals = [row1[7],row1[1]]
  y4_vals = [row1[8],row1[2]]
  middlex = row1[9]
  middley = row1[10]

  image = plt.imread('new_data/tr/' + str(row1[0]) + '.jpg')
  plt.figure()
  plt.plot(x1_vals, y1_vals,x2_vals, y2_vals,x3_vals, y3_vals,x4_vals, y4_vals,color='b')
  plt.plot(middlex,middley,'r.')
  plt.imshow(image)
  # print(middlex)
  # print(middley)
  i=i+1
  plt.show()

# plt.show()

# print(statistics.median(lenx))
# print(statistics.median(leny))
# print(median(lenx))
# print(max(leny))










