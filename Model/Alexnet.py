import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear

class Alexnet(nn.Module):

    def __init__(self):

        self.conv_1 = Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv_2 = Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
        self.conv_3 = Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)

        # fully connected layers
        self.fc_1 =

