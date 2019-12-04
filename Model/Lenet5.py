import torch.nn as nn
from torch.nn import Conv2d, Linear, MaxPool2d
import torch.nn.functional as F

# following link shows the Resnet-5 architecture https://engmrk.com/lenet-5-a-classic-cnn-architecture/
class Resnet5(nn.Module):

    def __init__(self,n_classes):
        super().__init__()
        # resnet has 3 input channels (RGB) and 6 filters with kernel size 5
        self.conv_1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv_2 = Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        # create max pool layers
        self.p = MaxPool2d(kernel_size=2, stride=2)
        '''create the 3 fully connected layers, the input size at fc_1 is determined by 
        (W_1 - F)S + 1, where W_1 is the width of the channel from conv_2 layer. '''
        self.fc_1 = Linear(in_features=5*5*16, out_features=120)
        self.fc_2 = Linear(in_features=120,out_features=84)
        self.fc_3 = Linear(in_features=84,out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = self.p(out)
        out = F.relu(self.conv_2(out))
        out = self.p(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

