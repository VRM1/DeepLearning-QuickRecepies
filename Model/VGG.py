import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Note: you need to add the root directoy of the project to import modules in VS code to test this file
HOME = os.path.expanduser('~')
root_dir = HOME + '/Documents/GitWorkSpace/PytorchCNNs'
sys.path.append(root_dir)
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator


# configs for bayesian VGG
bcfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
bcfg_d = {'Dense1': [512, 256, 'C']}


@variational_estimator
class BVGG(nn.Module):
    def __init__(self, n_classes, i_channel, vgg_name):

        if i_channel == 1:
            i_dim = 512
        else:
            # change this
            i_dim = 512
        super().__init__()
        self.features = self._make_conv_layers(i_channel, bcfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        self.classifier = self._make_dense_layers(i_dim, n_classes, bcfg_d['Dense1'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = F.dropout(self.classifier(out), p=0.5)
        out = self.classifier(out)
        return out

    def _make_dense_layers(self, i_dim, o_dim, cfg):
        layers = []
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout(p=0.5)]
            elif x == 'C':
                layers += [BayesianLinear(i_dim, o_dim)]
            else:
                layers += [BayesianLinear(i_dim, x),
                           nn.ReLU()]
                i_dim = x
        return nn.Sequential(*layers)

    def _make_conv_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [BayesianConv2d(in_channels, x, kernel_size=(3,3), padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    i_channel = 3
    classes = 10
    net = BVGG(classes, i_channel, 'VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
