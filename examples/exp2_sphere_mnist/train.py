import math
import argparse
import sys
import numpy as np
import pickle, gzip

import sys
sys.path.append("../../meshcnn")
from utils import sparse2tensor, spmatmul, MNIST_S2_Loader
from ops import MeshConv
from model import LeNet

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel.distributed_cpu import DistributedDataParallelCPU
from torch.utils.data.distributed import DistributedSampler


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} \r'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \r'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, required=True,
                        help='path to mesh folder (default: mesh_files)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--datafile', type=str, default="mnist_ico4.gzip",
                        help='data file containing preprocessed spherical mnist data')
    parser.add_argument('--dist_cpu', action='store_true')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dist_cpu:
        dist.init_process_group(backend='mpi')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainset = MNIST_S2_Loader(args.datafile, "train")
    testset = MNIST_S2_Loader(args.datafile, "test")
    if args.dist_cpu:
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=(test_sampler is None), 
                              sampler=test_sampler, **kwargs)
    
    model = LeNet(mesh_folder=args.mesh_folder)
    model = nn.DataParallel(model)
    model.to(device)
    if args.dist_cpu:
        model = DistributedDataParallelCPU(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable model parameters: {0}".format(count_parameters(model)))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        if args.dist_cpu:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        
if __name__ == "__main__":
    main()
