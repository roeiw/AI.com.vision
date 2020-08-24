from pathlib import Path
import segmentation
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt

path = 'C:/Users/roei.w/Desktop/machine/Input/'
with open(path + 'validation/train_set.csv', 'r') as csvin, open(path + 'ValidPoint/tmp.csv', 'a') as csvout:
    reader = csv.reader(csvin)
    writer = csv.writer(csvout, lineterminator = '\n')
    headers = next(reader)

    headers.append('roof_x_center')
    headers.append('roof_y_center')

    #writer.writerow(headers)
    img_not_found =[]
    stop = True
    cnt = 0
    for row in reader:
        print("got into for")
        # if stop:
        #     if row[0] == '15309':
        #         stop = False
        #     continue
        tmp_list = [float(i) for i in row[1:11]]
        image_path = path + 'validation/' + row[0] + '.jpg'
        print(image_path)
        # if not stop:
        #     if Path(path + 'checkRoof/' + row[0]).exists():
        #         cnt += 1
        #         continue
        #     else:
        #         print(cnt)
        #         stop = True
        try:
            image = Image.open(image_path)
        except:
            img_not_found.append(row[0])
            continue
        image.convert('L')
        try:
            roof_x, roof_y = segmentation.run_first_length(image, tmp_list)
        except:
            roof_x = 45
            roof_y = 45
        tmp_list = row
        tmp_list.append(roof_x)
        tmp_list.append(roof_y)
        writer.writerow((tmp_list))
        plt.title(row[0])
        plt.imshow(image)
        plt.plot(roof_x, roof_y, 'c.')
        print("got to saving")
        plt.savefig(path + 'ValidPoint/' + row[0])
        plt.close()


    print(img_not_found)
    print(len(img_not_found))
