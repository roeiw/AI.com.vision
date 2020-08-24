import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

#
#
# path = '/machine/Input/train/'
# name ='10192'

num_list=['15807']
result_list=[]
data_path = 'C:/Users/roei.w/Desktop/machine/Input/FB1/'
old_df=pd.read_csv(data_path + 'train_set.csv')
# image_number = '10118'
for num in num_list:
    image_number = num
    image_path = data_path + image_number + '.jpg'
    image=Image.open(image_path)
    plt.imshow(image)
    x=plt.ginput(1)
    res=(image_number,x)
    # print("x is ",res)
    result_list.append(res)
    print(res)
    new_df=old_df.loc[old_df['name_id'].isin([image_number])]['roof_x_center']=x[0]
    print(new_df)
