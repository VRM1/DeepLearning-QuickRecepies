import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear
# https://github.com/piEsposito/blitz-bayesian-deep-learning
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

'''
This structure differs from what Alex had proposed in his work: https://arxiv.org/pdf/1404.5997.pdf
, which is clearly desribed here: 
1. https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/


The architecture seems to be modified in the Pytorch TORCHVISION.MODELS. The modification is mostly
related to the size of the filter, the stride and padding. For instance, the vannila paper used 96 kernels in the
first conv layer, here we use 64. If you want to keep the same architecture as the original, you can simply upscale
the 32*32 image to 227*227. However, this does not yield the best results.

https://github.com/AbhishekTaur/AlexNet-CIFAR-10/blob/master/alexnet.py
'''

# alexnet for imagenet
class AlexnetImageNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        self.fc_1 = Linear(in_features=6*6*256, out_features=4096)
        self.fc_2 = Linear(in_features=4096, out_features=4096)
        self.fc_3 = Linear(in_features=4096, out_features=n_classes)

        # layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.fc_1, self.fc_2, self.fc_3]
        # self.layers = nn.ModuleList(layers)
    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.relu(self.conv_5(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = F.dropout(out, inplace=True)
        out = self.fc_2(out)
        out = F.dropout(out, inplace=True)
        out = self.fc_3(out)
        return out


# alexnet for CIFAR 10
class AlexnetCifar(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=2)
        self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.mx_pl = nn.MaxPool2d(kernel_size=3, stride=2)

        # fully connected layers
        self.fc_1 = Linear(in_features=4096, out_features=2048)
        self.fc_2 = Linear(in_features=2048, out_features=2048)
        self.fc_3 = Linear(in_features=2048, out_features=n_classes)

        # layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.fc_1, self.fc_2, self.fc_3]
        # self.layers = nn.ModuleList(layers)
    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)
        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.relu(self.conv_5(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out), inplace=True)
        out = F.dropout(out)
        out = self.fc_2(out)
        out = F.dropout(out)
        out = self.fc_3(out)
        return out

# bayesian alexnet for CIFAR 10
@variational_estimator
class BAlexnet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.conv_1 = BayesianConv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=2)
        self.conv_2 = BayesianConv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=2)
        self.conv_3 = BayesianConv2d(in_channels=192, out_channels=384, kernel_size=(3,3), padding=1)
        self.conv_4 = BayesianConv2d(in_channels=384, out_channels=256, kernel_size=(3,3), padding=1)
        self.conv_5 = BayesianConv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1)
        self.mx_pl = nn.MaxPool2d(kernel_size=3, stride=2)

        # fully connected layers
        self.fc_1 = BayesianLinear(in_features=4096, out_features=512)
        self.fc_2 = BayesianLinear(in_features=512, out_features=256)
        self.fc_3 = BayesianLinear(in_features=256, out_features=n_classes)

        # layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.fc_1, self.fc_2, self.fc_3]
        # self.layers = nn.ModuleList(layers)
    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out,kernel_size=2)
        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out))
        out = F.relu(self.conv_4(out))
        out = F.relu(self.conv_5(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

class AlexNetB(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc
