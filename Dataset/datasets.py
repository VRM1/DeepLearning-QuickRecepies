import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from os.path import expanduser
from torch import nn

HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'

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

def get_cifar10():

    n_classes = 10
    i_channel = 3
    i_dim = 32
    # transforms taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    # transform_train = transforms.Compose([transforms.ToTensor(),\
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4), \
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # transform_test = transforms.Compose([transforms.ToTensor(),\
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([ transforms.ToTensor(), \
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_d = CIFAR10(
        root=D_PTH, train=True,
        download=True, transform=transform_train)
    # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    test_d = CIFAR10(
        root=D_PTH, train=False,
        download=True, transform=transform_test)

    return (n_classes, i_channel, i_dim, train_d, test_d)

def get_cifar100():

        n_classes = 100
        i_channel = 3
        i_dim = 32
        # transforms taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        # transform_train = transforms.Compose([transforms.ToTensor(),\
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4), \
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        # transform_test = transforms.Compose([transforms.ToTensor(),\
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([ transforms.ToTensor(), \
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_d = CIFAR100(
            root=D_PTH, train=True,
            download=True, transform=transform_train)
        # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
        test_d = CIFAR100(
            root=D_PTH, train=False,
            download=True, transform=transform_test)

        return (n_classes, i_channel, i_dim, train_d, test_d)
