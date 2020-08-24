from typing import Any, Union

import matplotlib.image as mpimg
from math import floor
import matplotlib.pyplot as plt
import pandas as pd
import csv
import PIL
from PIL import Image
import numpy as np
import os
from skimage.transform import warp, AffineTransform
import runimage
import random
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame, Series
from pandas.core.arrays import ExtensionArray
import statistics

with open('C:/Users/roei.w/Desktop/machine/train.csv', 'r') as csvin, open('C:/Users/roei.w/Desktop/machine/Input/temp11/train_set.csv','w') as csvout:
    writer = csv.writer(csvout, lineterminator='\n')
    reader = csv.reader(csvin)
    headers=['name_id','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y','p4_x','p4_y','center_x','center_y','general_class','sub_class']
    flist = runimage.get_img_list()
    print(flist)
    # print(flist)
    img_size=90
    writer.writerow(headers)
    i=0
    for image in os.listdir('fol/valid'):
        name = image.split('.')[0]
        img = PIL.Image.open('fol/valid/'+image)
        W,H=img.size
        df = pd.read_csv('train.csv')
        df_new = df[df.image_id.eq(int(name))]
        # print(name)
        df_new= df_new[df_new.general_class =='small vehicle']
        df_new = df_new[~df_new['tag_id'].isin(flist)]
        # print(df_new)
        # df_new= df_new[df_new.subclass !='small vehicle']
        lenx = []
        leny = []
        tags = df_new.tag_id.to_numpy()
        table = df_new.drop(columns=['tag_id', 'image_id']).to_numpy()
        count=0
        for row in table:
            paste_bottom = 0
            paste_top = 1
            paste_right = 0
            paste_left = 1
            listx = [row[0], row[2], row[4], row[6]]
            listy = [row[1], row[3], row[5], row[7]]
            middlex = round((listx[0] + listx[2]) / 2,4)
            middley = round((listy[0] + listy[2]) / 2,4)
            listz = [row[8], row[9]]
            numminx = min(listx)
            nummaxx = max(listx)
            val1 = nummaxx - numminx
            numminy = min(listy)
            nummaxy = max(listy)
            val2 = nummaxy - numminy
            # middlex=middlex.__format__(".3f")
            # middley=middley.__format__(".3f")
            # middley="%.4f"%middley
            # print(type(middlex))
            top = max(0, floor(middley - (img_size/2)))
            left = max(0, floor(middlex - (img_size/2)))
            right = min(W, floor(middlex + (img_size/2)))
            bottom = min(floor(middley + (img_size/2)), H)
            listx = [x - left for x in listx]
            listy = [y - top for y in listy]
            middlex = middlex - left
            middley = middley - top
            need_adj_x=0
            need_adj_y=0
            if top == 0:
                paste_bottom = img_size
                paste_top = img_size-(bottom - top)
                need_adj_y=1
               # bottom=nummaxy+numminy #calc the bottom if the top is on the edge
            elif bottom == H:
                paste_top = 0
                paste_bottom = bottom-top
                #top = numminy-(H-nummaxy) # calc the top if the botoon is in the edge
            if left == 0:
                paste_right = img_size
                paste_left = img_size-(right-left)
                need_adj_x=1
                # right = nummaxx+numminx # calc right if left is on edge
            elif right == W:
                paste_left = 0
                paste_right = right-left
                # left = numminx -(W-nummaxx) # calc...
            if right-left==img_size and bottom-top==img_size:
                paste_box=(0,0,img_size,img_size)
            elif right-left==img_size and bottom-top!=img_size:
                paste_box=(0,paste_top,img_size,paste_bottom)
                listy=[y + need_adj_y*(img_size-(bottom-top)) for y in listy]
                middley=middley + need_adj_y*(img_size-(bottom-top))
            elif right-left!=img_size and bottom-top == img_size:
                paste_box = (paste_left, 0, paste_right, img_size)
                listx=[x + need_adj_x*(img_size-(right-left)) for x in listx]
                middlex=middlex + need_adj_x*(img_size-(right-left))
            elif right - left != img_size and bottom - top != img_size:
                paste_box = (paste_left, paste_top, paste_right, paste_bottom)
                listy=[y + need_adj_y*(img_size-(bottom-top)) for y in listy]
                listx=[x + need_adj_x*(img_size-(right-left)) for x in listx]
                middley=middley + need_adj_y*(img_size-(bottom-top))
                middlex=middlex + need_adj_x*(img_size-(right-left))
            #paste_box = (left*paste_left, top*paste_top, img_size*paste_right, img_size*paste_bottom)
            # if top==0: bottom = img_size
            # elif bottom==H: top=H-img_size
            # if left==0: right = img_size
            # elif right==W: left=W-img_size
            # print(middlex)
            # print(middley)
            # dx = middlex - (img_size/2)
            # dy = middley - (img_size/2)
            # middlex=(img_size/2)
            # middley=(img_size/2)
            # listx = [x + dx for x in listx]
            # listy = [y - dy for y in listy]
            x1_vals = [listx[0], listx[1]]
            y1_vals = [listy[0], listy[1]]
            x2_vals = [listx[1], listx[2]]
            y2_vals = [listy[1], listy[2]]
            x3_vals = [listx[2], listx[3]]
            y3_vals = [listy[2], listy[3]]
            x4_vals = [listx[3], listx[0]]
            y4_vals = [listy[3], listy[0]]
            img1 = img.crop((left, top, right, bottom))
            #img11=np.array(img1)
            #transform = AffineTransform(translation=(dx, dy))
            # shift_img = warp(img11, transform, mode="wrap")
            # if tags[count]==11642:
            #     print(dx)
            #     print(dy)
            #     plt.imshow(shift_img)
            #     plt.plot(x1_vals, y1_vals, x2_vals, y2_vals, x3_vals, y3_vals, x4_vals, y4_vals, color='b')
            #     # plt.show()
            # plt.show()
            new_img=Image.new('RGB',(img_size,img_size))
            print(img1.size)
            print(paste_box)
            print(str(tags[count]))
            img1=img1.convert('RGB')
            new_img.paste(img1,paste_box)
            new_img.save('Input/temp11/' + str(tags[count]) + '.jpg')
            valss = []
            valss.append(int(tags[count]))
            for j in range(4):
                valss.append(float(listx[j]))
                valss.append(float(listy[j]))
            valss.append(float(middlex))
            valss.append(float(middley))
            valss.append(listz[0])
            valss.append(listz[1])
            writer.writerow(valss)
            count = count +1
        i = i + 1
        # if i==5: break




