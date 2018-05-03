from __future__ import print_function, division

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   #


# BY: NAQVI, SYED ABID ABBAS
# PHD STUDENT: at SHANGHAI JIAO-TONG UNIVERSITY from Pakistan

########### Note: You must have CUDA device enabled to run this tutorial

# Bugs fixed in TRANSFER-LEARNING TUTORIAL https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# I found out that, I was not able to compile this tutorial code on my machine.
# So, I have fixed it and it is Now running perfectly on my Machine
# Operating System: Ubuntu 16.04 LTS
# NVIDIA: 1050Ti
# IDE: Pycharm
# Python 3.6.4 |Anaconda custom (64-bit)|


###  Data loading and shuffling/augmentation/normalization : all handled by torch automatically.
# This is a little hard to understand initially, so I'll explain in detail here!
# For training, the data gets transformed by undergoing augmentation and normalization.
# The RandomSizedCrop basically takes a crop of an image at various scales between 0.01 to 0.8
# times the size of the image and resizes it to given number
# Horizontal flip is a common technique in computer vision to augment the size of your data set.
#  Firstly, it increases the number of times the network gets
# to see the same thing, and secondly it adds rotational invariance to your networks learning.


# Just normalization for validation, no augmentation.

# You might be curious where these numbers came from? For the most part, they were used in popular architectures like the AlexNet paper.
# It is important to normalize your dataset by calculating the mean and standard deviation of your dataset images and making your data unit normed.
#  However,it takes a lot of computation to do so, and some papers have shown that it doesn't matter too much if they are slightly off.
#  So, people just use imagenet dataset's mean and standard deviation to normalize their dataset approximately.
#  These numbers are imagenet mean and standard deviation!

# If you want to read more, transforms is a function from torchvision, and you can go read more here -
# http://pytorch.org/docs/master/torchvision/transforms.html



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '/home/thaqafi/PycharmProjects/transfer_learning/hymenoptera'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
print("--> dsets",dsets)
print("\n")
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=10,
                                               shuffle=True, num_workers=25)
                for x in ['train', 'val']}
print("data laoder ", dset_loaders)
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes



print("Data Set Classes Are:", dset_classes)
print("Data Set Class Object type is: ")# It is a List
print(help(dset_classes))
print("##############################")
print("##############################")
print("##############################")
print("Data set loader Object type is: ") # It is Dictionary
print( help(dset_loaders))
print("##############################")
print("##############################")
print("##############################")
print("Data set Object type is: ")
print(help(dsets))
print("##############################")
print("##############################")
# You can comment this for loop
for i, (z, yu) in enumerate(dset_loaders['val']):
    print("Frits training value: --> ", z)
    print("yu value-->" , yu)






def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated




### Writing the functions that do training and validation phase.

# These functions basically do forward propogation, back propogation, loss calculation, update weights of model, and save best model!


## The below function will train the model. Here's a short basic outline -

# For the number of specified epoch's, the function goes through a train and a validation phase. Hence the nested for loop.

# In both train and validation phase, the loaded data is forward propogated through the model (architecture defined ahead).
# In PyTorch, the data loader is basically an iterator. so basically there's a get_element function which gets called everytime
# the program iterates over data loader. So, basically, get_item on dset_loader below gives data, which contains 2 tensors - input and target.
# target is the class number. Class numbers are assigned by going through the train/val folder and reading folder names in alphabetical order.
# So in our case cats would be first, dogs second and humans third class.

# Forward prop is as simple as calling model() function and passing in the input.

# Variables are basically wrappers on top of PyTorch tensors and all that they do is keep a track of every process that tensor goes through.
# The benefit of this is, that you don't need to write the equations for backpropogation, because the history of computations has been tracked
# and pytorch can automatically differentiate it! Thus, 2 things are SUPER important. ALWAYS check for these 2 things.
# 1) NEVER overwrite a pytorch variable, as all previous history will be lost and autograd won't work.
# 2) Variables can only undergo operations that ar


def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):

        for phase in ['train' , 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            counter=0

            for data in dset_loaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.float().cuda()),  Variable(labels.long().cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds  = torch.max(outputs.data, 1)
            loss = criterion(outputs ,labels)

#            if counter % 50 == 0:
#                print("Reached iteration ", counter)
#                counter += 1

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            try:
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            except:
                print('unexpected error, could not calculate loss or do a sum.')

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                print('New best accuracy', best_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model




def visualize_model(model, num_images=6):
    was_training = model.training
    print("############ VISUALIZATION STAGE ##########################\n")
    print("############ VISUALIZATION STAGE ##########################\n")
    print("############ VISUALIZATION STAGE ##########################\n")
    print("############ VISUALIZATION STAGE ##########################\n")

    print("model.training ", was_training)
    print("\n")

    model.eval()
    images_so_far = 0
    fig = plt.figure()
#    print("Inside the Visualize Model Called")

#    with torch.set_grad_disalbe():
    for i, (inputs, labels) in enumerate(dset_loaders['val']):
         #   inputs = inputs
         #   labels = labels
            inputs, labels = Variable(inputs.float().cuda(), requires_grad = False), Variable(labels.long().cuda(), requires_grad = False)
            #inputs = Variable(inputs.float())
            #labels = Variable(labels.long())
            print("The input to Model is" , inputs)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print("The prediction is -->", preds)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                index  = int(j)
                print("INDEX-->", dset_classes[int(preds[index])])
                ax.set_title('predicted: {}'.format(dset_classes[int(preds[index])]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
    model.train(mode=was_training)


### DEFINING MODEL ARCHITECTURE.
# We use Resnet18 here. If you have more computational power, feel free to swap it with Resnet50, Resnet100 or Resnet152.
# Since we are doing fine-tuning, or transfer learning we will use the pretrained net weights.
# In the last line, the number of classes has been specified.





model_ft = models.resnet18(pretrained=True)
number_of_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(number_of_features, 2)

criterion = nn.CrossEntropyLoss()

criterion.cuda()
model_ft.cuda()

optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)

print("#############################################\n")



### Visualizing Test Data
visualize_model(model_ft)

plt.ioff()
plt.show()






