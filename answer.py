import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


# %%
class NN(nn.Module):
    def __init__(self, arr=[]):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 30 * 3, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# %%
class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1568, 5)

    def forward(self, x):
        """
        Question 2

        TODO: fill this forward function for data flow
        """
        x = self.pool(F.relu(self.conv_layer(x)))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x


# %%
basic_transformer = transforms.Compose([transforms.ToTensor()])

"""
Question 3

TODO: Add color normalization to the transformer. For simplicity, let us use 0.5 for mean
      and 0.5 for standard deviation for each color channel.
"""
norm_transformer = transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize(0.5, 0.5)])


# %%
class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()
        """
        Question 4

        TODO: setup the structure of the network
        """
        self.conv_layer1 = nn.Conv2d(3, 8, 3) # setting up the layers based on the values in the array
        self.conv_layer2 = nn.Conv2d(8, 16, 3)
        self.conv_layer3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 12 * 12, 5)
        

    def forward(self, x):
        """
        Question 4

        TODO: setup the flow of data (tensor)
        """
        x = self.conv_layer1(x)
        x = F.relu(x)
        x = self.conv_layer2(x)
        x = F.relu(x)
        x = self.conv_layer3(x)
        x = F.relu(x)
        x = self.pool(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# %%
"""
Question 5

TODO:
    change the train_transformer to a tranformer with random horizontal flip,
    random crop, and random affine transformation

    1. It should randomly flip the image horizontally with probability 50%
    2. It should apply random affine transformation to the image, which randomly rotate the image 
        within 5 degrees, and shear the image within 10 degrees.
    3. It should include color normalization after data augmentation. Similar to question 3.
"""

"""Add random data augmentation to the transformer"""
aug_transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomCrop(size = 30),
                                      transforms.RandomAffine(degrees = 5, shear = 10),                  
                                      transforms.Normalize(0.5, 0.5)])


