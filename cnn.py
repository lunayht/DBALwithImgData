import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters: int=32, kernel_size: int=4, dense_layer: int=128, \
                 img_rows: int=28, img_cols: int=28, maxpool: int=2):
        """
        num_filters: Number of filters, out channel for 1st and 2nd conv layers,
        kernel_size: Kernel size of convolution,
        dense_layer: Dense layer units
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters*((img_rows-2*kernel_size+2)//2)*((img_cols-2*kernel_size+2)//2), \
                             dense_layer)
        self.fc2 = nn.Linear(dense_layer, 10)

    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out