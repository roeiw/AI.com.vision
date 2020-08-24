import pandas as pd
import glob
import csv
from PIL import Image
import numpy as np
import os
import math

def get_img_list():
    with open('C:/Users/roei.w/Desktop/machine/train.csv', 'r') as csvin:
        reader = csv.reader(csvin)
        #headers = ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y','general_class']
        # is_small = reader['general_class'] == 'small_vehicle'
        headers=next(reader)
        count=0
        max_x_id=0
        max_y_id=0
        max_width = 0
        max_height = 0
        pickupc=0
        area_sum=0
        avg_bbox_area=1506
        flist=[]
        hlist=[]
        bbox_count=0
        # print(headers[1])
        last_img='0'
        for row in reader:
            near_b_x = False
            near_b_y = False
            if row[10] == 'small vehicle' and row[0]!='12886' and row[0]!='21981' :
                if last_img !=row[1]:
                    img_path=glob.glob('fol/tr/' + row[1] + '.*')
                    if not img_path:
                        img_path=glob.glob('fol/valid/' +row[1] +'.*')
                    img = Image.open(img_path[0])
                # print(img_path)
                # print(row[0])
                W, H = img.size
                # print("size of picture is: %d %d" % (W,H))
                W1=float(W*0.001)
                W9=float(W-W1)
                H1=float(H*0.001)
                H9=float(H-H1)
                max_x=max(float(row[2]),float(row[4]),float(row[6]),float(row[8]))
                min_x=min(float(row[2]),float(row[4]),float(row[6]),float(row[8]))
                max_y=max(float(row[3]),float(row[5]),float(row[7]),float(row[9]))
                min_y=min(float(row[3]),float(row[5]),float(row[7]),float(row[9]))
                # print("max x is %d" % max_x)
                # print("min x is %d" % min_x)
                # print("min y is %d" % min_y)
                # print("max y is %d" % max_y)
                # print("W1 is %d" % W1)
                # print("H1 is %d" % H1)
                if(max_x>W9 or min_x<W1):
                    # print("in near x")
                    near_b_x=True
                elif(max_y>H9 or min_y<H1):
                    # print("in near y")
                    near_b_y=True
                size_x= max(abs(float(row[4])-float(row[8])),abs(float(row[6])-float(row[2])))
                val_y1=pow(float(row[3])-float(row[5]),2)
                val_x1=pow(float(row[2])-float(row[4]),2)
                val_x2=pow(float(row[2])-float(row[8]),2)
                val_y2=pow(float(row[3])-float(row[9]),2)
                distance_a=math.sqrt(val_x1+val_y1)
                distance_b=math.sqrt(val_x2+val_y2)
                area=distance_a*distance_b
                # area_sum=area_sum+area
                lenx=abs(float(row[4])-float(row[2]))
                leny=abs(float(row[5])-float(row[7]))
                size_y= max(abs(float(row[5])-float(row[9])),abs(float(row[7])-float(row[3])))
                if  (near_b_y==True or near_b_x==True):
                    bbox_count+=1
                    flist.append(row[0])
                    count = count + 1
                    continue
                if size_x>91 or size_y>91:
                    pickupc=pickupc+1
                    flist.append(row[0])
                    count = count + 1
                    continue
                if size_x>max_width:
                    max_x_id=row[0]
                    max_width = max(max_width, size_x)
                if size_y > max_height:
                    max_y_id = row[0]
                    max_height = max(max_height, size_y)
                last_img = row[1]
                count=count+1

        # average_area=area_sum/count
        # print ("max x width is: %d" % max_width)
        # print ("max y height is: %d" % max_height)
        # print("average area is: %d" %average_area)
        # print(count)
        # print(bbox_count)
        # print(pickupc)
        # print ("max x id is: %s" % max_x_id)
        # print("max y id is: %s" % max_y_id)
        # print(flist)

        return flist


def main():
        get_img_list()
if __name__=="__main__":
            main()






