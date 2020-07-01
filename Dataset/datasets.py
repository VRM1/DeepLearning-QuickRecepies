import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from os.path import expanduser
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import nn

HOME = expanduser('~')
# D_PTH = HOME + '/Documents/DataRepo'
D_PTH = HOME + '/data/DataRepo'


def get_mnist():
    n_classes = 10
    i_channel = 1
    i_dim = 28
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_d = MNIST(
        root=D_PTH, train=True,
        download=True, transform=transform_train)
    # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    test_d = MNIST(
        root=D_PTH, train=False,
        download=True, transform=transform_test)
    return (n_classes, i_channel, i_dim, train_d, test_d)


def get_fashion_mnist():
    n_classes = 10
    i_channel = 1
    i_dim = 28
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_d = FashionMNIST(
        root=D_PTH, train=True,
        download=True, transform=transform_train)
    # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    test_d = FashionMNIST(
        root=D_PTH, train=False,
        download=True, transform=transform_test)

    return (n_classes, i_channel, i_dim, train_d, test_d)


# def get_cifar10(self):

#     n_classes = 10
#     i_channel = 3
#     i_dim = 32

#     transform_train = transforms.Compose([ transforms.ToTensor(), \
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

#     transform_test = transforms.Compose([ transforms.ToTensor(), \
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

#     train_d = CIFAR10(
#         root=D_PTH, train=True,
#         download=True, transform=transform_train)
#     # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
#     test_d = CIFAR10(
#         root=D_PTH, train=False,
#         download=True, transform=transform_test)

#     return (n_classes, i_channel, i_dim, train_d, test_d)

def get_cifar10(train_batch_sz=256, test_batch_sz=512, is_valid=False):
    n_classes = 10
    i_channel = 3
    i_dim = 32

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_test = transforms.Compose([transforms.ToTensor(), \
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    train_d = CIFAR10(
        root=D_PTH, train=True,
        download=True, transform=transform_train)
    # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    test_d = CIFAR10(
        root=D_PTH, train=False,
        download=True, transform=transform_test)
    train_len = len(train_d)
    test_len = len(test_d)
    indices = range(train_len)
    # 10% of data is used for validation
    if is_valid:
        split = int(np.floor(0.1 * train_len))
    else:
        split = int(np.floor(0 * train_len))
    
    valid_indx = np.random.choice(indices, split)
    train_indx = set(indices).difference(set(valid_indx))
    train_sampler = SubsetRandomSampler(list(train_indx))
    train_len = len(train_indx)
    valid_len = len(valid_indx)
    valid_sampler = SubsetRandomSampler(valid_indx)
    train_loader = DataLoader(train_d, batch_size=train_batch_sz, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(train_d, batch_size=test_batch_sz, sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(test_d, batch_size=test_batch_sz, shuffle=True, num_workers=0)

    return n_classes, i_channel, i_dim, train_len, valid_len, \
           test_len, train_loader, valid_loader, test_loader


def get_cifar100():
    n_classes = 100
    i_channel = 3
    i_dim = 32
    # transforms taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    # transform_train = transforms.Compose([transforms.ToTensor(),\
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    # transform_test = transforms.Compose([transforms.ToTensor(),\
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([transforms.ToTensor(), \
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    train_d = CIFAR100(
        root=D_PTH, train=True,
        download=True, transform=transform_train)
    # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    test_d = CIFAR100(
        root=D_PTH, train=False,
        download=True, transform=transform_test)

    return (n_classes, i_channel, i_dim, train_d, test_d)
