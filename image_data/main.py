'''
This program trains the following self.models with Cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

1. Resnet-5
'''
import argparse
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
from Model import Lenet5, AlexnetCifar, BAlexnet
from Model import VGG, BVGG
import os
import numpy as np
from torchinfo import summary
import GPUtil
import torch as th
from Dataset import get_cifar10
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from Utils import EarlyStopping
import pdb
import timeit

torch.manual_seed(33)
np.random.seed(33)
if torch.cuda.is_available():
    # check which gpu is free and assign that gpu
    AVAILABLE_GPU = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, \
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])[0]
    th.cuda.set_device(AVAILABLE_GPU)
    print('Program will be executed on GPU:{}'.format(AVAILABLE_GPU))
    DEVICE = torch.device('cuda:' + str(AVAILABLE_GPU))
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
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
        # number of MCMC samples for Bayesian NN. If network is not bayesian, it is simply set as 1
        if args.is_bayesian:
            self.n_samples = 3
        else:
            self.n_samples = 1
        self.criterion = nn.CrossEntropyLoss()
        self.optim = args.optimizer
        self.m_name = args.model
        self.d_name = args.dataset
        self.lr = args.learning_rate
        self.resume = args.resume
        self.start_epoch = 0
        # path to write trained weights
        self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name + '-' + str(self.epochs) + \
                                 '-' + str(self.tr_b_sz) + '.pth'
        
        
        self.n_classes, self.i_channel, self.i_dim, self.train_len, self.valid_len, self.test_len, \
        self.train_loader, self.valid_loader, self.test_loader = get_cifar10(self.tr_b_sz, self.tst_b_sz)

        if self.d_name == 'cifar100':
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
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, \
             sampler=train_sampler, num_workers=1)
        self.valid_loader = DataLoader(self.train_d, batch_size=256, \
             sampler=valid_sampler, num_workers=1)

    def init_model(self, load_weights=False, res_round=None):
        if self.m_name == 'lenet5' and not self.is_bayesian:
            self.model = Lenet5(self.n_classes)
            summary(self.model, (self.tr_b_sz, 3, 32, 32))
            t_param = sum(p.numel() for p in self.model.parameters())

            summary(self.model, (self.tr_b_sz, 3, 32, 32))

            self.model = self.model.to(DEVICE)

        elif self.m_name == 'alexnet' and not self.is_bayesian:
            self.model = AlexnetCifar(self.n_classes)
            t_param = sum(p.numel() for p in self.model.parameters())
            summary(self.model, (self.tr_b_sz, 3, 32, 32))
            self.model = self.model.to(DEVICE)

        # bayesian alexnet
        elif self.m_name == 'alexnet' and self.is_bayesian:
            self.model = BAlexnet(self.n_classes).to(DEVICE)
            t_param = sum(p.numel() for p in self.model.parameters())
            summary(self.model, (self.tr_b_sz, 3, 32, 32))
        elif self.m_name == 'VGG' and not self.is_bayesian:
            self.model = VGG(self.n_classes, self.i_channel, 'VGG19').to(DEVICE)
            # torch.nn.DataParallel(self.model.features)
            t_param = sum(p.numel() for p in self.model.parameters())

        elif self.m_name == 'VGG' and self.is_bayesian:
            self.model = BVGG(self.n_classes, self.i_channel, 'VGG19').to(DEVICE)
            # torch.nn.DataParallel(self.model.features)
            t_param = sum(p.numel() for p in self.model.parameters())

        if self.resume:
            self.__load_pre_train_model(self.resume)
        print('Running Mode:{}, #TrainingSamples:{}, #ValidationSamples:{}, #TestSamples:{}, #Parameters:{} ResumingFromEpoch:{}'
              .format(self.m_name, self.train_len, self.valid_len, self.test_len, t_param, self.start_epoch))

    def __load_pre_train_model(self, resume):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel()
        # load the weights
        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.init_optimizer(self.lr)
        self.model.load_state_dict(state['weights'])
        self.start_epoch = state['epoch']
        # self.optimizer.load_state_dict(state['optimizer'])

    def init_optimizer(self, l_rate=0.001):
        if self.optim == 'SGD':
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9, 
            #                     weight_decay=5e-4)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate, amsgrad=True)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate, weight_decay=1e-5)

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
                                              sample_nbr=self.n_samples,
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

    def getTrainedmodel(self):
        retrain = 100
        if self.is_bayesian:
            net_typ = '_is_bayesian_1'
        else:
            net_typ = '_is_bayesian_0'
        self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name \
                                 + '-b' + str(self.tr_b_sz) + '-mcmc' + str(self.n_samples) + '-' \
                                 + net_typ + '-' + self.optim + '.pkl'
        return (self.model, self.train_weight_path)

def _initiate_arguments(parser):
    
    parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default="config.yml")
    parser.add_argument('-m', '--model', help='model name 1.lenet5 2.alexnet 3. VGG', default='lenet5')
    parser.add_argument('-d', '--dataset', help='dataset type', default='cifar10')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=150, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-ba', '--is_bayesian', help='to use bayesian layer or not', action='store_true')
    parser.add_argument('-v', '--is_valid', help='whether to use validation or not', action='store_true')
    parser.add_argument('-r', '--resume', help='if you want to resume from an epoch', action='store_true')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = _initiate_arguments(parser)

    parser.add_argument('-dev', '--device', help='device type can be cpu or gpu or mps (for M1 mac only) \
        If not arugment is given, the program auto-detects', default=None)

    args = parser.parse_args()
    if args.device:
        DEVICE = torch.device(args.device)
    run_model = RunModel(args)

    patience = 10
    start_epoch = 0
    if args.resume:
        start_epoch = run_model.start_epoch
        
    early_stopping = EarlyStopping(patience=patience, verbose=True, typ='loss')
    start = timeit.default_timer()
    for e in range(start_epoch, args.epochs):
        avg_train_loss, train_accuracy = run_model.train()
        if args.is_valid:
            valid_accuracy = run_model.test(is_valid=True)
        
        tst_accuracy = run_model.test()
        model, path_to_write = run_model.getTrainedmodel()
        early_stopping(e, avg_train_loss, model, run_model.optimizer, path_to_write)
        if early_stopping.early_stop:
            break
        if args.is_valid:
            print('Epoch:{}, AvgTrainLoss:{:.3f}, TrainAccuracy:{:.2f}, ValidationAccuracy:{:.2f}, TestAccuracy:{:.2f}'
                .format(e, avg_train_loss, train_accuracy, valid_accuracy, tst_accuracy))
        else:
            print('Epoch:{}, AvgTrainLoss:{:.3f}, TrainAccuracy:{:.2f}, TestAccuracy:{:.2f}'
                .format(e, avg_train_loss, train_accuracy, tst_accuracy))
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
