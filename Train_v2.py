'''
This program trains the following self.models with Cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

1. Resnet-5
'''
import argparse
import torch
import sys
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Model import Lenet5, AlexnetCifar, BAlexnet
from Model import BVGG
from Model import AlexNetB
import pkbar
import os
import numpy as np
from torchsummary import summary
import GPUtil
import torch as th
from Dataset import get_cifar10
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

torch.manual_seed(33)
np.random.seed(33)
if torch.cuda.is_available():
    # check which gpu is free and assign that gpu
    AVAILABLE_GPU = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, \
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])[0]
    th.cuda.set_device(AVAILABLE_GPU)
    print('Program will be executed on GPU:{}'.format(AVAILABLE_GPU))
    DEVICE = torch.device('cuda:' + str(AVAILABLE_GPU))
else:
    DEVICE = torch.device('cpu')

'''
This program uses CIFAR10 data: https://www.cs.toronto.edu/~kriz/cifar.html for image classification using
several popular self.models based on convolution neural network.
'''
if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')


class RunModel:

    def __init__(self, args):

        self.epochs = 150
        self.tr_b_sz = 256
        self.tst_b_sz = 512
        self.is_bayesian = args.is_bayesian
        self.criterion = nn.CrossEntropyLoss()
        self.optim = args.optimizer
        self.m_name = args.model
        self.lr = 0.001
        # path to write trained weights
        self.train_weight_path = 'trained_weights/' + self.m_name + '-' + args.dataset + '-' + str(self.epochs) + \
                                 '-' + str(self.tr_b_sz) + '.pth'

        if args.dataset == 'cifar10':
            self.n_classes, self.i_channel, self.i_dim, self.train_len, self.valid_len, self.test_len, \
            self.train_loader, self.valid_loader, self.test_loader = get_cifar10(self.tr_b_sz, self.tst_b_sz)

        if args.dataset == 'cifar100':
            self.n_classes = 100
            train_d = CIFAR100(
                root='datasets', train=True,
                download=True, transform=self.transform_train)
            test_d = CIFAR100(
                root='datasets', train=False,
                download=True, transform=self.transform_test)

        self.init_model()
        self.init_optimizer(self.lr)

    def get_validation_data(self, is_valid):

        indices = range(self.train_len)
        split = int(np.floor(0.1 * self.train_len))
        valid_indx = np.random.choice(indices, split)
        train_indx = set(indices).difference(set(valid_indx))
        train_sampler = SubsetRandomSampler(list(train_indx))
        valid_sampler = SubsetRandomSampler(valid_indx)
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=1)
        self.valid_loader = DataLoader(self.train_d, batch_size=256, sampler=valid_sampler, num_workers=1)

    def init_model(self):
        if self.m_name == 'lenet5' and not self.is_bayesian:
            self.model = Lenet5(self.n_classes).to(DEVICE)
            t_param = sum(p.numel() for p in self.model.parameters())
            print(summary(self.model, (3, 32, 32)))
        elif self.m_name == 'alexnet' and not self.is_bayesian:
            self.model = AlexnetCifar(self.n_classes).to(DEVICE)
            t_param = sum(p.numel() for p in self.model.parameters())
            print(summary(self.model, (3, 32, 32)))
        # bayesian alexnet
        elif self.m_name == 'alexnet' and self.is_bayesian:
            self.model = BAlexnet(self.n_classes).to(DEVICE)
            t_param = sum(p.numel() for p in self.model.parameters())

        elif self.m_name == 'VGG' and self.is_bayesian:
            self.model = BVGG(self.n_classes, self.i_channel, 'VGG11').to(DEVICE)
            self.model.features = torch.nn.DataParallel(self.model.features)
            t_param = sum(p.numel() for p in self.model.parameters())

        print('Running Mode:{}, #TrainingSamples:{}, #ValidationSamples:{}, #TestSamples:{}, #Parameters:{}'
              .format(self.m_name, self.train_len, self.valid_len, self.test_len, t_param))

    def init_optimizer(self, l_rate=0.001):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate, amsgrad=True)

    def train(self):

        train_loss = []
        correct = 0
        total = 0
        self.model.train()
        for batch_idx, (X, Y) in enumerate(tqdm(self.train_loader)):

            X, Y = X.to(DEVICE), Y.to(DEVICE)
            self.optimizer.zero_grad()
            if self.is_bayesian:
                loss = self.model.sample_elbo(inputs=X,
                                              labels=Y,
                                              criterion=self.criterion,
                                              sample_nbr=3,
                                              complexity_cost_weight=1 / 50000)
            else:
                outputs = self.model(X)
                loss = self.criterion(outputs, Y)

            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            if self.is_bayesian:
                outputs = self.model(X)

            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()

        t_accuracy = (100. * correct / total)
        avg_train_loss = np.average(train_loss)
        return avg_train_loss, t_accuracy

    def test(self, is_valid=False):

        correct = 0
        total = 0
        if is_valid:
            data = self.valid_loader
        else:
            data = self.test_loader
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tqdm(data)):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                outputs = self.model(X)
                _, predicted = outputs.max(1)
                total += Y.size(0)
                correct += predicted.eq(Y).sum().item()
        t_accuracy = (100. * correct / total)
        return t_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN self.models that use CIFAR10')
    parser.add_argument('-m', '--model', help='model name 1.lenet5 2.alexnet 3. VGG', default='lenet5')
    parser.add_argument('-d', '--dataset', help='dataset type', default='cifar10')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-ba', '--is_bayesian', help='to use bayesian layer or not', action='store_true')

    args = parser.parse_args()
    run_model = RunModel(args)
    for e in range(args.epochs):
        avg_train_loss, train_accuracy = run_model.train()
        valid_accuracy = run_model.test(is_valid=True)
        tst_accuracy = run_model.test()
        print('Epoch:{}, AvgTrainLoss:{}, TrainAccuracy:{}, ValidationAccuracy:{}, TestAccuracy:{}'
              .format(e, avg_train_loss, train_accuracy, valid_accuracy, tst_accuracy))
