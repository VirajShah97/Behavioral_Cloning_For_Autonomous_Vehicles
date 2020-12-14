import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
  def __init__(self):

    super(Net, self).__init__()

    self.conv_1 = nn.Conv2d(1,24, kernel_size=5, stride=2)
    self.conv_2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
    self.conv_3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
    self.conv_4 = nn.Conv2d(48, 64, kernel_size=3)
    self.conv_5 = nn.Conv2d(64, 64, kernel_size=3)

    # CAN ADD this and label the new architecture to be modified NVIDIA.
    #self.maxpool_1 = nn.MaxPool2d(2, stride=1) #######################

    self.linear_1 = nn.Linear(6656, 100) # input of 6656 is determined from output of last conv 2d layer
    self.linear_2 = nn.Linear(100, 50)
    self.linear_3 = nn.Linear(50, 10)
    self.linear_4 = nn.Linear(10, 2) # output is vehicle control: steering and throttle

    self.dropout = nn.Dropout(p=0.5) #reduce overfitting

    self.elu = torch.nn.ELU() # elu is used as activation to increase non-linearity

  def forward(self,x):
    # Conv 2d layers
    x = self.conv_1(x)
    x = self.elu(x)
    #x = self.maxpool_1(x) #########################

    x = self.conv_2(x)
    x = self.elu(x)

    x = self.conv_3(x)
    x = self.elu(x)

    x = self.conv_4(x)
    x = self.elu(x)

    x = self.conv_5(x)
    x = self.elu(x)

    x = x.reshape(x.size(0), -1) #flatten

    #linear dense layers (fully connected)
    x = self.linear_1(x)
    x = self.elu(x)
    x = self.dropout(x)

    x = self.linear_2(x)
    x = self.elu(x)
    x = self.dropout(x)

    x = self.linear_3(x)
    x = self.elu(x)
    x = self.dropout(x)

    pred = self.linear_4(x) #no softmax for output

    return (pred)

