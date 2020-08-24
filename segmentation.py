import skimage.segmentation as seg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from PIL import Image, ImageStat
import math
from math import ceil, floor

data_path = 'C:/Users/Adam/Desktop/adam/project/data/training/temp/'
image_number = '27517'
image_path = data_path + image_number + '.jpg'
csv_path = data_path + 'train_set.csv'

is_slope_1_bigger = False
is_slope_2_bigger = False
steps_delta = 0.05
bright_pixels_delta = 0.25
dark_pixels_delta = 0.18
pixels_delta = 0
img_size = 90
extant_val = 1
distance_ratio = 0.05
image_mean = 0
last_slope_1_sign = 2
last_slope_2_sign = 2
print_step = False
mean_radius = 4

def pixels_mean (pixels, x_center, y_center, slope):
    sum = 0
    counter = 0
    global mean_radius
    dx = 0.01
    dy = slope*dx
    if abs(dy) > 0.5:
        dy = np.sign(dy)*0.5
        dx = dy / slope
    last_x = x_center
    last_y = y_center
    while(math.sqrt((last_x - x_center)**2 + (last_y - y_center)**2) < mean_radius):
        sum += pixels[last_x, last_y]
        counter += 1
        last_y += dy
        last_x += dx
    dx = -0.01
    dy = slope*dx
    if abs(dy) > 0.5:
        dy = np.sign(dy)*0.5
        dx = dy / slope
    last_x = x_center
    last_y = y_center
    while (math.sqrt((last_x - x_center) ** 2 + (last_y - y_center) ** 2) < mean_radius):
        sum += pixels[last_x, last_y]
        counter += 1
        last_y += dy
        last_x += dx
    return (sum/(counter))

# finds out if we passed the bounding box, if so we can stop
def pass_the_bbox(x_cord, y_cord, x_ref_1, y_ref_1, x_ref_2, y_ref_2, bbox_slope, first_access): #get the x, y cordinates of current pixel and the last slope between those cordinates an a point on the bounding box
    dx_1 = x_cord - x_ref_1
    dy_1 = y_cord - y_ref_1
    dx_2 = x_cord - x_ref_2
    dy_2 = y_cord - y_ref_2

    try:
        slope_1 = dy_1 / dx_1
        slope_2 = dy_2 / dx_2
    except:
        return 1
    global is_slope_1_bigger
    global is_slope_2_bigger
    global last_slope_1_sign
    global last_slope_2_sign
    global print_step
    global img_size

    if first_access:
        # print("the bbox slope is: ", bbox_slope)
        # print("the ref point is: ", (x_ref_2, y_ref_2))
        # print("cordinates are: ", (x_cord, y_cord), " dx and dy are: ", (dx_2,dy_2))
        # if x_cord == 38.581938334354994 and y_cord == 44.550000000000026:
        #     print_step = True
        # else: print_step = False

        if slope_1 < bbox_slope:
            #print("the slope is smaller --------------------- ", slope_1)
            is_slope_1_bigger = False
        else:

            is_slope_1_bigger = True
        if slope_2 < bbox_slope:
            is_slope_2_bigger = False
            # print("the slope is smaller --------------------- ", slope_2)
        else:
            # print("the slope is bigger ---------------------- ", slope_2)
            is_slope_2_bigger = True
    slope_1_sign = np.sign(slope_1)
    slope_2_sign = np.sign(slope_2)
    # if (print_step):
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print("the slope 2 is: ", slope_2, "cordinates are: ", (x_cord, y_cord))
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # if abs (bbox_slope) > 50:
        # print("the slope is big and he is: ", slope_1, "cordinates are: ", (x_cord, y_cord))
    if (slope_1 > bbox_slope and is_slope_1_bigger== False) or x_cord > img_size or x_cord < 0 or y_cord <0 or y_cord > img_size :
        # print("the final slope 1 is: ", slope_1)
        return 1
    elif slope_1 < bbox_slope and is_slope_1_bigger == True:
        # print("the final slope 1 is: ", slope_1)
        return 1
    elif slope_2 > bbox_slope  and is_slope_2_bigger == False:
        # print ("")
        return 1
    elif slope_2 < bbox_slope and is_slope_2_bigger == True:
        # print("2222222222222222222222222222222222222222222222")
        return 1

    if slope_1_sign != last_slope_1_sign and (abs (bbox_slope) > 75) and last_slope_1_sign != 2:
        # print ("failed on slope 1")
        return 1
    elif slope_2_sign != last_slope_2_sign and (abs(bbox_slope) > 75) and last_slope_2_sign != 2:
        # print(" slope 2 sign is ", slope_2_sign, " last sign is ", last_slope_2_sign)
        # print("slope is ", slope_2)
        # print("failed on slope 2")
        return 1
    else:
        last_slope_1_sign = np.sign(slope_1)
        last_slope_2_sign = np.sign(slope_2)
        return 0

def find_roof_edges(start_x, start_y, ref_pixel, slope, bbox_slope, pixels, row):
    # print ("the ref pixel is: " , ref_pixel, " the starting point is: ", (start_x, start_y))
    global steps_delta
    global pixels_delta
    global extant_val
    global distance_ratio
    global image_mean
    global last_slope_1_sign
    global last_slope_2_sign

    last_pixels = []
    pass_bbox = []
    # print("the mean of image pixels is: ", image_mean[0])
    for n in range(2):
        if (abs(ref_pixel-image_mean[0]) < 12 or ref_pixel < 90):
            pixels_delta = dark_pixels_delta
            # print("the pixel delta is: ", pixels_delta)
        else:
            pixels_delta = bright_pixels_delta
            # print("the pixel delta is: ", pixels_delta)
        # if abs(ref_pixel-image_mean[0]) > 30:
        #     pixels_delta = 0.23

       # print("the pixel delta is: ", pixels_delta)
        first_iter = True
        dy = (-1) ** n * steps_delta
        try:
            dx = dy / slope
        except:
            dx = np.sign(dy)*3.5
        if abs(dx) > 3:
            dx = np.sign(dx)*3
            dy = np.sign(dy)*3*slope
        # print("the step in y direction: ", dy ," the step in x direction is: ", dx)
        current_pixel = ref_pixel
        last_x = start_x
        last_y = start_y
        distance = 0
        last_slope_1_sign = 2
        last_slope_2_sign = 2
        while abs(current_pixel - ref_pixel) < pixels_delta * ref_pixel:
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ pixel: ', current_pixel)
            if first_iter:
                res = pass_the_bbox(last_x, last_y, row[0], row[1], row[4], row[5], bbox_slope, first_iter)
            else:
                res = pass_the_bbox(last_x+ dx, last_y+ dy, row[0], row[1], row[4], row[5], bbox_slope, first_iter)
            if (res):
                # print("passed the bounding box in cordinates: ", (last_x, last_y))
                last_x += extant_val * dx
                last_y += extant_val * dy

                break
            last_y += dy
            last_x += dx
            # distance = math.sqrt((last_x - middlex) ** 2 + (last_y - middley) ** 2)
            # print('4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444')
            # print(last_x, last_y, current_pixel)
            # print(
            #     '4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444')
            current_pixel = pixels_mean(pixels, last_x, last_y, bbox_slope)
            first_iter = False
        pass_bbox.append(res)
        # print ("-------------------------------------------------------------------------------------------------------")
        # print("the result is: ", res, " data is: ", (last_x, last_y, current_pixel))
        # print("-------------------------------------------------------------------------------------------------------")
        last_pixels.append((last_x, last_y, current_pixel))
        # plt.plot(last_x, last_y, 'm.')
    return last_pixels, pass_bbox


    # if np.sign(slope_1) != np.sign(last_slope_1) or np.sign(slope_2) != np.sign(last_slope_2):
    #     return 1000, -1000 # return 1000, -1000 if we passed the bounding box
    # return last_slope_1, last_slope_2




def run_first_length(img, row):
    global is_slope_1_bigger
    global is_slope_2_bigger
    global steps_delta
    global bright_pixels_delta

    global dark_pixels_delta
    global pixels_delta
    global img_size
    global extant_val
    global distance_ratio
    global image_mean
    # image1=io.imread(image_path)
    # plt.imshow(image1)
    # df=pd.read_csv(csv_path)
    # df_new=df[df.name_id.eq(int(image_number))]
    # table = df_new.drop(columns=['name_id']).to_numpy()
    # #print(table)
    # # list=list(df_new)
    # # print(list)
    # i=0
    # for row in table:
      #print(row)
    x1_vals = [row[0],row[2]]
    #print(x1_vals)
    y1_vals = [row[1],row[3]]
    x2_vals = [row[2],row[4]]
    y2_vals = [row[3],row[5]]
    x3_vals = [row[4],row[6]]
    y3_vals = [row[5],row[7]]
    x4_vals = [row[6],row[0]]
    y4_vals = [row[7],row[1]]

    # plt.plot(row[0], row[1], 'r.')
    # plt.plot(row[2], row[3], 'b.')
    # plt.plot(row[6], row[7], 'g.')
    # plt.plot(row[4], row[5], 'c.')
    dy1 = row[7]-row[1]
    dx1 = row[6]-row[0]
    dy2 = row[3]-row[1]
    dx2 = row[2]-row[0]
    distance1 = math.sqrt(math.pow(dx1, 2) + math.pow(dy1, 2))
    try: angle1 = dy1/dx1
    except: return (45,45)
    distance2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))
    try: angle2 = dy2 / dx2
    except: return(45,45)
    if distance1 > distance2:
        car_len_slope = angle1
        car_width_slope = angle2
        car_len = distance1
        car_width = distance2
    else:
        car_len_slope = angle2
        car_width_slope = angle1
        car_len = distance2
        car_width = distance1

    middlex = row[8]
    middley = row[9]
     # plt.plot(middlex, middley , 'g.')
    #image = Image.open(image_path)
    image = img.convert('L')
    pixels = image.load()
    # print("pixels matrix is: ", pixels)
    reference_pixel = pixels_mean(pixels, middlex, middley, car_width_slope)

    stats = ImageStat.Stat(image).mean
    stats2 = ImageStat.Stat(image).median
    image_mean = stats2
    #
    # print("function mean return value: ", mean(pixels, middlex, middley, 10))
    # # img = np.array(image)
    # # img =  img_as_float(img)
    # print("the average value of all pixels is: ", stats)
    # print("the median value of all pixels is: ", stats2)
    #print("the width slop is: ", car_width_slope, " the len slope is: ", car_len_slope)
    # print("the middle pixel value is: ", reference_pixel)

    length_edges, pass_bbox_l = find_roof_edges(middlex, middley, reference_pixel ,car_len_slope, car_width_slope, pixels, row )

    # print("the values after going through the length is: ", length_edges)

    tmp_y = (length_edges[0][1] + length_edges[1][1]) / 2
    tmp_x = (length_edges[0][0] + length_edges[1][0]) / 2
    # plt.plot(tmp_x, tmp_y, 'r.')

    if (length_edges[0][2] > reference_pixel and length_edges[1][2]<reference_pixel):
        if not pass_bbox_l[0]: repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope, car_width_slope, pixels, row)
        else : repeat1 = length_edges
        if not pass_bbox_l[1] : repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2]+13, car_len_slope, car_width_slope, pixels, row)
        else: repeat2 = length_edges

        if repeat1[0][2] > reference_pixel + 45:
            repeat3, pass_bbox = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_len_slope, car_width_slope, pixels, row)

            tmp_y = (repeat3[0][1] + repeat2[1][1]) / 2
            tmp_x = (repeat3[0][0] + repeat2[1][0]) / 2
            # print("values after fixing the length repeat3: ", repeat3)
        else:
            tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
            tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2

    elif length_edges[0][2] < reference_pixel and length_edges[1][2] > reference_pixel:
        if not pass_bbox_l[0]: repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope, car_width_slope, pixels, row)
        else: repeat1 = length_edges
        if not pass_bbox_l[1]: repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2], car_len_slope, car_width_slope, pixels, row)
        else: repeat2 = length_edges

        if repeat2[1][2] > reference_pixel + 45:
            repeat3,_ = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_len_slope, car_width_slope, pixels, row)
            # print("12222312312312312321321312", repeat3[1][1])
            tmp_y = (repeat3[1][1] + repeat1[0][1]) / 2
            tmp_x = (repeat3[1][0] + repeat1[0][0]) / 2
            # print("values after fixing the length repeat3: ", repeat3)
            # print("the tmp points are ", (tmp_x, tmp_y))
        else:
            tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
            tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
        # print("values after fixing the length repeat1: ", repeat1)
        # print("values after fixing the length repeat2: ", repeat2)

#===========================================================================================
    # elif length_edges[0][2] > reference_pixel and length_edges[1][2] > reference_pixel:
    #     if not pass_bbox_l[0]: repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope, car_width_slope)
    #     else: repeat1 = length_edges
    #     if not pass_bbox_l[1]: repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2], car_len_slope, car_width_slope)
    #     else: repeat2 = length_edges
    #
    #     tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
    #     tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
    #
    #     print("values after fixing the length repeat1: ", repeat1)
    #     print("values after fixing the length repeat2: ", repeat2)

 #==============================================================
    # print ("the points now are ", (tmp_x, tmp_y))
    new_ref_pixel = pixels_mean(pixels, tmp_x, tmp_y, car_len_slope)
    width_edges, pass_bbox_w = find_roof_edges(tmp_x, tmp_y, new_ref_pixel , car_width_slope, car_len_slope, pixels, row)
    # print(
    #     "======================================================================================="
    # )
    # print("values after going on the width of the car: ", width_edges, pass_bbox_w)
    # print("======================================================================================")

    # final_slope = (width_edges[1][1]-row[1])/(width_edges[1][0]-row[0])
    # print("the final slope is: ", final_slope)
    # print(row[0],row[1])
    # print(row[2],row[3])
    # print(row[6],row[7])
    if width_edges[0][2] > new_ref_pixel and width_edges[1][2] < new_ref_pixel:
        if pass_bbox_w[0] == 0: repeat1, pass_bbox = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2], car_width_slope, car_len_slope, pixels, row)
        else: repeat1 = width_edges
        if pass_bbox_w[1] == 0: repeat2, pass_bbox = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2], car_width_slope, car_len_slope, pixels, row)
        else: repeat2 = width_edges

        desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
        desired_x = (repeat1[0][0] + repeat2[1][0]) / 2

        # print("after fixing the width: ", repeat1, pass_bbox_w)
        # print("after fixing the width: ", repeat2, pass_bbox_w)
        if repeat1[0][2] > new_ref_pixel + 45:
            repeat3, pass_bbox = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_width_slope, car_len_slope, pixels, row)

            desired_y = (repeat3[0][1] + repeat2[1][1]) / 2
            desired_x = (repeat3[0][0] + repeat2[1][0]) / 2
            # print("values after fixing the width repeat3: ", repeat3)
        else:
            desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
            desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
        # print("values after fixing the width repeat1: ", repeat1)
        # print("values after fixing the width repeat2: ", repeat2)

    elif  (width_edges[0][2] < new_ref_pixel and width_edges[1][2] > new_ref_pixel):
        if pass_bbox_w[0] == 0: repeat1, pass_bbox = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2], car_width_slope, car_len_slope, pixels, row)
        else: repeat1 = width_edges
        if pass_bbox_w[1] == 0: repeat2, pass_bbox = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2], car_width_slope, car_len_slope, pixels, row)
        else: repeat2 = width_edges

        if repeat2[1][2] > new_ref_pixel + 45:
            repeat3, pass_bbox = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_width_slope, car_len_slope, pixels, row)

            desired_y = (repeat3[1][1] + repeat1[0][1]) / 2
            desired_x = (repeat3[1][0] + repeat1[0][0]) / 2
            # print("values after fixing the width repeat3: ", repeat3)
        else:
            desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
            desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
        # print("values after fixing the width repeat1: ", repeat1)
        # print("values after fixing the width repeat2: ", repeat2)
    else:
        desired_x = (width_edges[0][0] + width_edges[1][0]) / 2
        desired_y = (width_edges[0][1] + width_edges[1][1]) / 2

    # print(desired_x, desired_y)
    # plt.plot(desired_x, desired_y, 'y.')
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return desired_x, desired_y


if __name__ == '__main__':
        # global is_slope_1_bigger
        # global is_slope_2_bigger
        # global steps_delta
        # global bright_pixels_delta
        #
        # global dark_pixels_delta
        # global pixels_delta
        # global img_size
        # global extant_val
        # global distance_ratio
        # global image_mean
        image1=io.imread(image_path)
        plt.imshow(image1)
        df=pd.read_csv(csv_path)
        df_new=df[df.name_id.eq(int(image_number))]
        table = df_new.drop(columns=['name_id']).to_numpy()
        #print(table)
        # list=list(df_new)
        # print(list)
        i=0
        for row in table:
            # print(row)
            x1_vals = [row[0], row[2]]
            # print(x1_vals)
            y1_vals = [row[1], row[3]]
            x2_vals = [row[2], row[4]]
            y2_vals = [row[3], row[5]]
            x3_vals = [row[4], row[6]]
            y3_vals = [row[5], row[7]]
            x4_vals = [row[6], row[0]]
            y4_vals = [row[7], row[1]]

            # plt.plot(row[0], row[1], 'r.')
            # plt.plot(row[2], row[3], 'b.')
            # plt.plot(row[6], row[7], 'g.')
            # plt.plot(row[4], row[5], 'c.')
            dy1 = row[7] - row[1]
            dx1 = row[6] - row[0]
            dy2 = row[3] - row[1]
            dx2 = row[2] - row[0]
            distance1 = math.sqrt(math.pow(dx1, 2) + math.pow(dy1, 2))
            angle1 = dy1 / dx1
            distance2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))
            angle2 = dy2 / dx2
            if distance1 > distance2:
                car_len_slope = angle1
                car_width_slope = angle2
                car_len = distance1
                car_width = distance2
            else:
                car_len_slope = angle2
                car_width_slope = angle1
                car_len = distance2
                car_width = distance1

            middlex = row[8]
            middley = row[9]
        plt.plot(middlex, middley , 'g.')
        image = Image.open(image_path)
        image = image.convert('L')
        pixels = image.load()
        reference_pixel = pixels_mean(pixels, middlex, middley, car_width_slope)

        # stats = ImageStat.Stat(image).mean
        # stats2 = ImageStat.Stat(image).median
        # # image_mean = stats2
        #
        # print("function mean return value: ", mean(pixels, middlex, middley, 10))
        # # img = np.array(image)
        # # img =  img_as_float(img)
        # print("the average value of all pixels is: ", stats)
        # print("the median value of all pixels is: ", stats2)
        # print("the width slop is: ", car_width_slope, " the len slope is: ", car_len_slope)
        print("the middle pixel value is: ", reference_pixel)

        length_edges, pass_bbox_l = find_roof_edges(middlex, middley, reference_pixel, car_len_slope, car_width_slope, pixels, row)

        print("the values after going through the length is: ", length_edges)

        tmp_y = (length_edges[0][1] + length_edges[1][1]) / 2
        tmp_x = (length_edges[0][0] + length_edges[1][0]) / 2
        plt.plot(tmp_x, tmp_y, 'r.')

        if (length_edges[0][2] > reference_pixel and length_edges[1][2] < reference_pixel):
            if not pass_bbox_l[0]:
                repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2],
                                                     car_len_slope, car_width_slope, pixels, row)
            else:
                repeat1 = length_edges
            if not pass_bbox_l[1]:
                repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2] + 13,
                                                     car_len_slope, car_width_slope, pixels, row)
            else:
                repeat2 = length_edges

            if repeat1[0][2] > reference_pixel + 45:
                repeat3, pass_bbox = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_len_slope,
                                                     car_width_slope, pixels, row)

                tmp_y = (repeat3[0][1] + repeat2[1][1]) / 2
                tmp_x = (repeat3[0][0] + repeat2[1][0]) / 2
                print("values after fixing the length repeat3: ", repeat3)
            else:
                tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
                tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2

        elif length_edges[0][2] < reference_pixel and length_edges[1][2] > reference_pixel:
            if not pass_bbox_l[0]:
                repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2],
                                                     car_len_slope, car_width_slope, pixels, row)
            else:
                repeat1 = length_edges
            if not pass_bbox_l[1]:
                repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2],
                                                     car_len_slope, car_width_slope, pixels, row)
            else:
                repeat2 = length_edges

            if repeat2[1][2] > reference_pixel + 45:
                repeat3, _ = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_len_slope,
                                             car_width_slope, pixels, row)
                print("12222312312312312321321312", repeat3[1][1])
                tmp_y = (repeat3[1][1] + repeat1[0][1]) / 2
                tmp_x = (repeat3[1][0] + repeat1[0][0]) / 2
                print("values after fixing the length repeat3: ", repeat3)
                print("the tmp points are ", (tmp_x, tmp_y))
            else:
                tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
                tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
            print("values after fixing the length repeat1: ", repeat1)
            print("values after fixing the length repeat2: ", repeat2)

        # ===========================================================================================
        # elif length_edges[0][2] > reference_pixel and length_edges[1][2] > reference_pixel:
        #     if not pass_bbox_l[0]: repeat1, pass_bbox = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope, car_width_slope)
        #     else: repeat1 = length_edges
        #     if not pass_bbox_l[1]: repeat2, pass_bbox = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2], car_len_slope, car_width_slope)
        #     else: repeat2 = length_edges
        #
        #     tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
        #     tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
        #
        #     print("values after fixing the length repeat1: ", repeat1)
        #     print("values after fixing the length repeat2: ", repeat2)

        # ==============================================================
        print("the points now are ", (tmp_x, tmp_y))
        new_ref_pixel = pixels_mean(pixels, tmp_x, tmp_y, car_len_slope)
        width_edges, pass_bbox_w = find_roof_edges(tmp_x, tmp_y, new_ref_pixel, car_width_slope, car_len_slope, pixels, row)
        print(
            "======================================================================================="
        )
        print("values after going on the width of the car: ", width_edges, pass_bbox_w)
        print("======================================================================================")

        final_slope = (width_edges[1][1] - row[1]) / (width_edges[1][0] - row[0])
        # print("the final slope is: ", final_slope)
        # print(row[0],row[1])
        # print(row[2],row[3])
        # print(row[6],row[7])
        if width_edges[0][2] > new_ref_pixel and width_edges[1][2] < new_ref_pixel:
            if pass_bbox_w[0] == 0:
                repeat1, pass_bbox = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2],
                                                     car_width_slope, car_len_slope, pixels, row)
            else:
                repeat1 = width_edges
            if pass_bbox_w[1] == 0:
                repeat2, pass_bbox = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2],
                                                     car_width_slope, car_len_slope, pixels, row)
            else:
                repeat2 = width_edges

            desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
            desired_x = (repeat1[0][0] + repeat2[1][0]) / 2

            print("after fixing the width: ", repeat1, pass_bbox_w)
            print("after fixing the width: ", repeat2, pass_bbox_w)
            if repeat1[0][2] > new_ref_pixel + 45:
                repeat3, pass_bbox = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_width_slope,
                                                     car_len_slope, pixels, row)

                desired_y = (repeat3[0][1] + repeat2[1][1]) / 2
                desired_x = (repeat3[0][0] + repeat2[1][0]) / 2
                print("values after fixing the width repeat3: ", repeat3)
            else:
                desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
                desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
            print("values after fixing the width repeat1: ", repeat1)
            print("values after fixing the width repeat2: ", repeat2)

        elif (width_edges[0][2] < new_ref_pixel and width_edges[1][2] > new_ref_pixel):
            if pass_bbox_w[0] == 0:
                repeat1, pass_bbox = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2],
                                                     car_width_slope, car_len_slope,pixels, row)
            else:
                repeat1 = width_edges
            if pass_bbox_w[1] == 0:
                repeat2, pass_bbox = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2],
                                                     car_width_slope, car_len_slope, pixels, row)
            else:
                repeat2 = width_edges

            if repeat2[1][2] > new_ref_pixel + 45:
                repeat3, pass_bbox = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_width_slope,
                                                     car_len_slope, pixels, row)

                desired_y = (repeat3[1][1] + repeat1[0][1]) / 2
                desired_x = (repeat3[1][0] + repeat1[0][0]) / 2
                print("values after fixing the width repeat3: ", repeat3)
            else:
                desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
                desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
            print("values after fixing the width repeat1: ", repeat1)
            print("values after fixing the width repeat2: ", repeat2)
        else:
            desired_x = (width_edges[0][0] + width_edges[1][0]) / 2
            desired_y = (width_edges[0][1] + width_edges[1][1]) / 2

        print(desired_x, desired_y)
        plt.plot(desired_x, desired_y, 'y.')
        plt.imshow(image, cmap='gray')
        plt.show()



# def run_first_width():
#     global is_slope_1_bigger
#     global is_slope_2_bigger
#     global steps_delta
#     global bright_pixels_delta
#
#     global dark_pixels_delta
#     global pixels_delta
#     global img_size
#     global extant_val
#     global distance_ratio
#     global image_mean
#
#
#     print("function mean return value: ", mean(pixels, middlex, middley, 10))
#     # img = np.array(image)
#     # img =  img_as_float(img)
#     print("the average value of all pixels is: ", stats)
#     print("the median value of all pixels is: ", stats2)
#     # print("the width slop is: ", car_width_slope, " the len slope is: ", car_len_slope)
#     print("the middle pixel value is: ", reference_pixel)
#
#     width_edges = find_roof_edges(middlex, middley, reference_pixel, car_width_slope, car_len_slope)
#
#     print("values after going on the width of the car: ", width_edges)
#
#     final_slope = (width_edges[1][1] - row[1]) / (width_edges[1][0] - row[0])
#     # print("the final slope is: ", final_slope)
#     # print(row[0],row[1])
#     # print(row[2],row[3])
#     # print(row[6],row[7])
#     if width_edges[0][2] > reference_pixel and width_edges[1][2] < reference_pixel:
#         repeat1 = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2], car_width_slope,
#                                   car_len_slope)
#         repeat2 = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2], car_width_slope,
#                                   car_len_slope)
#
#         print("after fixing the width: ", repeat1)
#         print("after fixing the width: ", repeat2)
#         if repeat1[0][2] > reference_pixel + 45:
#             repeat3 = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_width_slope, car_len_slope)
#
#             desired_y = (repeat3[0][1] + repeat2[1][1]) / 2
#             desired_x = (repeat3[0][0] + repeat2[1][0]) / 2
#             print("values after fixing the width repeat3: ", repeat3)
#         else:
#             desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
#             desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
#         print("values after fixing the width repeat1: ", repeat1)
#         print("values after fixing the width repeat2: ", repeat2)
#
#     elif (width_edges[0][2] < reference_pixel and width_edges[1][2] > reference_pixel):
#         repeat1 = find_roof_edges(width_edges[0][0], width_edges[0][1], width_edges[0][2], car_width_slope,
#                                   car_len_slope)
#         repeat2 = find_roof_edges(width_edges[1][0], width_edges[1][1], width_edges[1][2], car_width_slope,
#                                   car_len_slope)
#
#         if repeat2[1][2] > reference_pixel + 45:
#             repeat3 = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_width_slope, car_len_slope)
#
#             desired_y = (repeat3[1][1] + repeat1[0][1]) / 2
#             desired_x = (repeat3[1][0] + repeat1[0][0]) / 2
#             print("values after fixing the width repeat3: ", repeat3)
#         else:
#             desired_y = (repeat1[0][1] + repeat2[1][1]) / 2
#             desired_x = (repeat1[0][0] + repeat2[1][0]) / 2
#         print("values after fixing the width repeat1: ", repeat1)
#         print("values after fixing the width repeat2: ", repeat2)
#     else:
#         desired_x = (width_edges[0][0] + width_edges[1][0]) / 2
#         desired_y = (width_edges[0][1] + width_edges[1][1]) / 2
#
#     new_ref_pixel = pixels[desired_x, desired_y]
#     length_edges = find_roof_edges(desired_x, desired_y, new_ref_pixel, car_len_slope, car_width_slope)
#
#     print("the values after going through the length is: ", length_edges)
#
#     tmp_y = (length_edges[0][1] + length_edges[1][1]) / 2
#     tmp_x = (length_edges[0][0] + length_edges[1][0]) / 2
#     plt.plot(tmp_x, tmp_y, 'r.')
#
#     if length_edges[0][2] > new_ref_pixel and length_edges[1][2] < new_ref_pixel:
#         repeat1 = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope,
#                                   car_width_slope)
#         repeat2 = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2], car_len_slope,
#                                   car_width_slope)
#
#         if repeat1[0][2] > new_ref_pixel + 45:
#             repeat3 = find_roof_edges(repeat1[0][0], repeat1[0][1], repeat1[0][2], car_len_slope, car_width_slope)
#
#             tmp_y = (repeat3[0][1] + repeat2[1][1]) / 2
#             tmp_x = (repeat3[0][0] + repeat2[1][0]) / 2
#             print("values after fixing the length repeat3: ", repeat3)
#         else:
#             tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
#             tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
#         print("values after fixing the length repeat1: ", repeat1)
#         print("values after fixing the length repeat2: ", repeat2)
#
#     elif length_edges[0][2] < new_ref_pixel and length_edges[1][2] > new_ref_pixel:
#         repeat1 = find_roof_edges(length_edges[0][0], length_edges[0][1], length_edges[0][2], car_len_slope,
#                                   car_width_slope)
#         repeat2 = find_roof_edges(length_edges[1][0], length_edges[1][1], length_edges[1][2], car_len_slope,
#                                   car_width_slope)
#
#         if repeat2[1][2] > new_ref_pixel + 45:
#             repeat3 = find_roof_edges(repeat2[1][0], repeat2[1][1], repeat2[1][2], car_len_slope, car_width_slope)
#
#             tmp_y = (repeat3[1][1] + repeat1[0][1]) / 2
#             tmp_x = (repeat3[1][0] + repeat1[0][0]) / 2
#             print("values after fixing the length repeat3: ", repeat3)
#         else:
#             tmp_y = (repeat1[0][1] + repeat2[1][1]) / 2
#             tmp_x = (repeat1[0][0] + repeat2[1][0]) / 2
#         print("values after fixing the length repeat1: ", repeat1)
#         print("values after fixing the length repeat2: ", repeat2)
#
#
#
#     print(tmp_x, tmp_y)
#     plt.plot(tmp_x, tmp_y, 'y.')
#     plt.imshow(image, cmap='gray')
#     plt.show()
#
# image1 = io.imread(image_path)
# plt.imshow(image1)
# df = pd.read_csv(csv_path)
# df_new = df[df.name_id.eq(int(image_number))]
# table = df_new.drop(columns=['name_id']).to_numpy()
# # print(table)
# # list=list(df_new)
# # print(list)
# i = 0
# for row in table:
#     # print(row)
#     x1_vals = [row[0], row[2]]
#     # print(x1_vals)
#     y1_vals = [row[1], row[3]]
#     x2_vals = [row[2], row[4]]
#     y2_vals = [row[3], row[5]]
#     x3_vals = [row[4], row[6]]
#     y3_vals = [row[5], row[7]]
#     x4_vals = [row[6], row[0]]
#     y4_vals = [row[7], row[1]]
#
#     plt.plot(row[0], row[1], 'r.')
#     plt.plot(row[2], row[3], 'b.')
#     plt.plot(row[6], row[7], 'g.')
#     plt.plot(row[4], row[5], 'c.')
#     dy1 = row[7] - row[1]
#     dx1 = row[6] - row[0]
#     dy2 = row[3] - row[1]
#     dx2 = row[2] - row[0]
#     distance1 = math.sqrt(math.pow(dx1, 2) + math.pow(dy1, 2))
#     angle1 = dy1 / dx1
#     distance2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))
#     angle2 = dy2 / dx2
#     if distance1 > distance2:
#         car_len_slope = angle1
#         car_width_slope = angle2
#         car_len = distance1
#         car_width = distance2
#     else:
#         car_len_slope = angle2
#         car_width_slope = angle1
#         car_len = distance2
#         car_width = distance1
#
#     middlex = row[8]
#     middley = row[9]
#     plt.plot(middlex, middley, 'g.')
# image = Image.open(image_path)
# image = image.convert('L')
# pixels = image.load()
# reference_pixel = pixels[middlex, middley]
#
# stats = ImageStat.Stat(image).mean
# stats2 = ImageStat.Stat(image).median
# image_mean = stats2
