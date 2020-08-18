## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 100,kernel_size = 5,stride = 1,padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 100,out_channels = 64, kernel_size = 5,stride = 1,padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 32,kernel_size = 3,stride = 1,padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32,out_channels = 16,kernel_size = 3,stride = 1,padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 16,out_channels = 16,kernel_size = 3,stride = 1,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 1)
        self.dropout1 = nn.Dropout(p = 0.4)
        self.dropout2 = nn.Dropout(p = 0.5)
        self.dropout3 = nn.Dropout(p = 0.2)
        self.Fc1 = nn.Linear(16*108*108,1000)
        self.Fc2 = nn.Linear(1000,1000)
        self.Fc3 = nn.Linear(1000,136)
        I.xavier_uniform_(self.conv1.weight)
        I.xavier_uniform_(self.conv2.weight)
        I.xavier_uniform_(self.conv3.weight)
        I.xavier_uniform_(self.conv4.weight)
        I.xavier_uniform_(self.conv5.weight)
        I.xavier_uniform_(self.Fc1.weight)
        I.xavier_uniform_(self.Fc2.weight)
        I.xavier_uniform_(self.Fc2.weight)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout3(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout3(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool2(F.relu(self.conv3(x))))
        x = self.dropout3(self.pool2(F.relu(self.conv4(x))))
        x = self.dropout3(self.pool2(F.relu(self.conv5(x))))
        x = x.view(x.shape[0],16*108*108)
        x = self.dropout1(F.relu(self.Fc1(x)))
        x = self.dropout1(F.relu(self.Fc2(x)))
        x = self.Fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
