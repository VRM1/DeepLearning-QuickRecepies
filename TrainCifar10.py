'''
This program trains the following models with Cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

1. Resnet-5
'''
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Model import Resnet5
from Utils import progress_bar

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

TWEIGHT_PTH = 'trained_weights/cifar_net.pth'

def Train(m_name):

    n_classes = 10
    if m_name == 'Restnet5':
        model = Resnet5(n_classes)

    # beging by doing some pre-processing and scaling of data
    transform_train = transforms.Compose([transforms.ToTensor(),\
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),\
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_d = CIFAR10(
            root='datasets', train=True,
            download=True, transform=transform_train)
    train_loader = DataLoader(train_d, batch_size=128, shuffle=True, num_workers=4)

    test_d = CIFAR10(
            root='datasets', train=False,
            download=True, transform=transform_test)
    epochs = 10
    train_loss = 0
    total = 0
    correct = 0
    test_loader = DataLoader(test_d, batch_size=128, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(epochs):
        model.train()
        for batch_idx, (X, Y) in enumerate(train_loader):

            X,Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs,Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    torch.save(model.state_dict(), TWEIGHT_PTH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN models that use CIFAR10')
    parser.add_argument('-m','--model', help='Model name', default='Restnet5')
    args = parser.parse_args()
    Train(args.model)
