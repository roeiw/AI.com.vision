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

name=31153
image1=plt.imread('Input/temp/' + str(name) +'.jpg')
plt.imshow(image1)
df=pd.read_csv('Input/temp/train_set.csv')
df_new=df[df.name_id.eq(name)]
table = df_new.drop(columns=['name_id']).to_numpy()
print(table)
# list=list(df_new)
# print(list)
i=0
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
  plt.plot(x1_vals, y1_vals,x2_vals, y2_vals,x3_vals, y3_vals,x4_vals, y4_vals,color='b')
  plt.plot(middlex,middley,'r.')
  print(middlex)
  print(middley)
  i=i+1
  plt.show()

plt.show()

# print(statistics.median(lenx))
# print(statistics.median(leny))
# print(median(lenx))
# print(max(leny))










