import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear

'''
This structure differs from what Alex had proposed in his work: https://arxiv.org/pdf/1404.5997.pdf
, which is clearly desribed here: 
https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/

The architecture seems to be modified in the Pytorch TORCHVISION.MODELS. The modification is mostly
related to the size of the filter and padding. For instance, the vannila paper used 96 kernels in the
first conv layer, here we use 64.
'''

class Alexnet(nn.Module):

    def __init__(self):

        self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1)
        self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)
        self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)

        # fully connected layers
        self.fc_1 = Linear(in_features=6*6*256, out_features=4096)
        self.fc_2 = Linear(in_features=4096, out_features=4096)
        self.fc_3 = Linear(in_features=4096, out_features=10)

    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = F.relu(self.conv_3(out))
        out = F.relu(self.conv_4(out))
        out = F.relu(self.conv_5(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)



