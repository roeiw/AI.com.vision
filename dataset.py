from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from operator import itemgetter
import math
from shapely.geometry import Polygon,Point
from skimage.transform import warp, AffineTransform, rotate


def get_classes(csv_file):
    file=pd.read_csv(csv_file)
    classes=file['sub_class'].unique()

class carsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self.landmarks_frame['sub_class'].unique()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,str(self.landmarks_frame.iloc[idx, 0]))
        img_name = img_name+'.jpg'
        # print(img_name)
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:11]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('double').reshape(-1,2)
        # print(type(landmarks))
        labels=self.landmarks_frame.iloc[idx,11:13]
        # print(landmarks)
        labels=labels.astype('str')
        roof_center=self.landmarks_frame.iloc[idx,13:15]
        roof_center=np.array(roof_center)
        roof_center=roof_center.astype('double')
        # print(torch.from_numpy(landmarks))
        # print(type(image),type(labels))
        sample = {'image': image, 'landmarks': landmarks, 'labels': labels, 'roof_center': roof_center}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, labels, roof_center = sample['image'], sample['landmarks'], sample['labels'], sample['roof_center']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks[0:5] = landmarks[0:5] * [new_w / w, new_h / h]
        roof_center=roof_center*[new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks, 'labels':labels, 'roof_center':roof_center}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks, labels, roof_center = sample['image'], sample['landmarks'], sample['labels'], sample['roof_center']
        # print(type(image))
        # print(type(landmarks))
        # print(type(labels))
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # print(labels)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'labels': list(labels),
                'roof_center':torch.from_numpy(roof_center)}

def show_landmarks(image, landmarks, labels,roof_center):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.plot(roof_center[0], roof_center[1],'b.')
    plt.pause(0.001)  # pause a bit so that plots are updated

class random_affine_transform(object):
    def __init__(self,image_size):
        self.height=image_size[1]
        self.width=image_size[0]
    def __call__(self, sample):
        # print("inside call, affine")
        image, landmarks, labels, roof_center = sample['image'], sample['landmarks'], sample['labels'], sample['roof_center']
        # for x in landmarks: print(tuple(x))
        polygon=Polygon([tuple(landmarks[0]),tuple(landmarks[1]),tuple(landmarks[2]),tuple(landmarks[3])])
        # print("inside call, polygon")
        max_x=max(landmarks, key=itemgetter(0))[0]
        max_y=max(landmarks, key=itemgetter(1))[1]
        min_x=min(landmarks, key=itemgetter(0))[0]
        min_y=min(landmarks, key=itemgetter(1))[1]
        center=landmarks[4]
        x_center=center[0]
        y_center=center[1]
        contain=False
        max_x_shift=math.floor(min(abs(self.width-max_x),min_x,abs(max_x-x_center)))
        max_y_shift=math.floor(min(abs(self.height-max_y),min_y,abs(max_y-y_center)))
        # print(polygon.bounds)
        # print(polygon.area)
        print("max x shift is: ",max_x_shift)
        print("max y shift is: ",max_y_shift)
        i=0
        while not contain:
            random_x=np.random.randint(-abs(max_x_shift),abs(max_x_shift))
            random_y=np.random.randint(-abs(max_y_shift),abs(max_y_shift))
            # print(random_x)
            # print(random_y)
            point=Point(random_x+45,random_y+45)
            i+=1

            if polygon.contains(point):
                # print("contains")
                contain=True
        transform = AffineTransform(translation=(random_x, random_y))
        shift_img = warp(image, transform, mode="wrap")
        landmarks=landmarks[0:5]-[random_x,random_y]
        roof_center=roof_center-[random_x,random_y]
        # print(landmarks)
        return {'image': shift_img, 'landmarks': landmarks, 'labels': labels, 'roof_center': roof_center}

class image_rotation(object):
    def __init__(self, image_size):
        self.height = image_size[1]
        self.width = image_size[0]

    def __call__(self, sample):
        image, landmarks, labels, roof_center = sample['image'], sample['landmarks'], sample['labels'], sample['roof_center']
        degree=np.random.randint(180)
        # print(degree)
        rotated_image=rotate(image,degree,mode="wrap")
        # print(landmarks)
        cordinates=[]
        for point in landmarks:
            cordinates.append(rotate_point(point,-math.radians(degree),(self.width/2,self.height/2)))
        cordinates = np.array([cordinates])
        cordinates = cordinates.astype('double').reshape(-1, 2)
        new_roof_center=rotate_point(roof_center,-math.radians(degree),(self.width/2,self.height/2))
        new_roof_center = np.array(new_roof_center)
        new_roof_center = new_roof_center.astype('double')
        # print(type(cordinates))
        return {'image': rotated_image, 'landmarks': cordinates, 'labels': labels, 'roof_center': new_roof_center }

def rotate_point(origin,degrees, center):
    cx, cy= center
    # print(center)
    print(origin)
    ox, oy= origin
    # print(ox,oy)
    # print(origin)
    qx=cx+math.cos(degrees)*(ox-cx)-math.sin(degrees)*(oy-cy)
    qy=cy+math.sin(degrees)*(ox-cx)+math.cos(degrees)*(oy-cy)
    return [qx,qy]




    # for x in landmarks:
        #     local_x=x[0]
        #     local_y=x[1]
        #     if local_x > max_x: max_x=local_x
        #     if local_y > max_y : max_y=local_y
        #     if local_x < min_x: min_x=local_x
        #     if local_y < min_y : min_y=local_y
        #




#
# carsdata=carsDataset('./Input/final_tag.csv','./Input/train')
# # print(carsdata.classes)
# affine = random_affine_transform((90,90))
# rotateed = image_rotation((90,90))
# composed = transforms.Compose([Rescale((110,110)),image_rotation((110,110)),random_affine_transform((110,110))])
# # Apply each of the above transforms on sample.
# fig = plt.figure()
# sample = carsdata[5672]
# # print(sample.image_name())
# sample['landmarks'] = sample['landmarks'][0:5]
# # print(sample)
# transformed_sample = rotateed(sample)
#     # ax = plt.subplot(1, 3, i + 1)
#     # plt.tight_layout()
#     # ax.set_title(type(tsfrm).__name__)
# show_landmarks(**transformed_sample)
# plt.show()