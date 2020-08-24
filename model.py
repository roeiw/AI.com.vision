from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import dataset

transform=transforms.Compose([
    dataset.Rescale((110,110)),
    dataset.image_rotation((110,110)),
    dataset.random_affine_transform((110, 110)),
    dataset.ToTensor()
])

trainrootdir='./Input/train'
traincsvfile='./Input/final_tag.csv'


train_datasets = dataset.carsDataset(traincsvfile,trainrootdir,transform)
train_dataloders = torch.utils.data.DataLoader(train_datasets, batch_size=4,shuffle=True, num_workers=0)

train_dataset_size =len(train_datasets)
class_names = train_datasets.classes

validrootdir='./Input/validation'
validcsvfile='./Input/validation/train_set.csv'

valid_datasets = dataset.carsDataset(validcsvfile,validrootdir,transform)
valid_dataloders = torch.utils.data.DataLoader(valid_datasets, batch_size=4,shuffle=True, num_workers=0)

valid_dataset_size =len(valid_datasets)
class_names = train_datasets.classes

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()
#
#     # Get a batch of training data
data =next(iter(train_dataloders))
images=data['image']
img_size=images.size(2)
grid_border_size=2
cordinates=data['landmarks']
labels=data['labels']
roof_center=data['roof_center']
listt=[i[4][1] for i in cordinates]
# print(listt)
# print(cordinates)
# for x in range(4):
#     print(listt)

# print(labels[0])
plt.figure()

# Make a grid from batch
out = torchvision.utils.make_grid(images)
# inp = images[1].numpy().transpose((1, 2, 0))
plt.imshow(out.numpy().transpose((1,2,0)))
# print("hello landmarks")
imshow(out, title=[labels[1][x] for x in range(4)])
print(roof_center)
for i in range(4):
    plt.scatter(cordinates[i,:, 0].numpy()+i*img_size+(i+1)*grid_border_size,cordinates[i,:, 1].numpy()+grid_border_size,s=10, marker='.', c='r')
    plt.scatter(55+i*img_size+(i+1)*grid_border_size,55+grid_border_size,s=2, marker='.', c='c')
    plt.scatter(roof_center[i,0].numpy()+i*img_size+(i+1)*grid_border_size,roof_center[i,1].numpy()+grid_border_size,s=2, marker='.', c='g')
plt.axis("off")
plt.ioff()
plt.show()

def train_model(model,criterion,optimizer, scheduler, num_of_epoches):
    start=time.time()
    for epoch in range(num_of_epoches):
        model.train()
        running_loss=0.0
        correct=0.0

        for i,data in enumerate(train_dataloders):
            images=data['image']
            cordinates=data['landmarks']
            # labels = data['labels']
            images=images.to(device)
            cordinates=cordinates.to(device)
            # labels=labels
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                prediction= model(images.double())
                x=[i[4] for i in cordinates]
                result=torch.stack(x,0)
                print("result is: ",result," and type is: ", type(result))
                print("prediction is: ",prediction," and type is: ", type(prediction))
                loss = criterion(prediction, result)
                loss.backward()
                optimizer.step()
            running_loss+=loss.item()*images.size(0)
            correct+=torch.sum(prediction==result)
            scheduler.step()
            if i == 1:
                print("got to i=1")
                break
        epoch_loss=running_loss/train_dataset_size
        epoch_acc=correct/train_dataset_size
        print("loss: {:.3f}".format(epoch_loss))
        print("accuracy: {:.3f}".format(epoch_acc))
    time_elapsed=time.time()-start
    print('triaing complete in {:.0f} mins and {:.0f} secs'.format(time_elapsed//60, time_elapsed%60))

# model=torchvision.models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad=False
# num_of_features=model.fc.in_features
# model.fc=nn.Linear(num_of_features,2)
# model=model.double()
# model=model.to(device)
# critirion=nn.MSELoss()
# optimizer=optim.Adam(model.fc.parameters(),lr=0.01)
# scheduler=lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)
# model=train_model(model,critirion,optimizer,scheduler,num_of_epoches=2)










