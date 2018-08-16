import math
import argparse
import sys
sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import os
import shutil


from ops import MeshConv
from loader import ClimateSegLoader
from model import UNet

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pdb


def save_checkpoint(state, is_best, epoch, output_folder, filename):
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')

def iou_score(pred_cls, true_cls, nclass=3):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    iou = []
    for i in range(nclass):
        # intersect = ((pred_cls == i) + (true_cls == i)).eq(2).item()
        # union = ((pred_cls == i) + (true_cls == i)).ge(1).item()
        intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
        union = ((pred_cls == i) + (true_cls == i)).ge(1).sum().item()
        print("Intersect: ", intersect, " Union: ", union)
        iou_ = intersect / union
        iou.append(iou_)
    return np.array(iou)

def train(args, model, train_loader, optimizer, epoch):
    # w = torch.tensor([0.97731504, 0.00104697, 0.021638]).cuda()
    w = torch.tensor([0.001020786726132422, 0.9528737404907279, 0.04610547278313972]).cuda()
    model.train()
    tot_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, weight=w)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        count += data.size()[0]
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} \n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()
    tot_loss /= count
    return tot_loss

def test(args, model, test_loader):
    # w = torch.tensor([0.97731504, 0.00104697, 0.021638]).cuda()
    w = torch.tensor([0.001020786726132422, 0.9528737404907279, 0.04610547278313972]).cuda()
    model.eval()
    test_loss = 0
    ious = np.zeros(3)
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]
            test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            iou = iou_score(pred, target, nclass=3)
            ious += iou * n_data
            count += n_data
    ious /= count
    test_loss /= count

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, IoU: {:.4f}, {:.4f}, {:.4f}\r'.format(
        test_loss, iou[0], iou[1], iou[2]))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="../../mesh_files",
                        help='path to mesh folder (default: ../../mesh_files)')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoint",
                        help='directory to save checkpoint (default:checkpoint)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=7, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--fdim', type=int, default=16, help='filter dimensions')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    kwargs = {}
    trainset = ClimateSegLoader(args.data_folder, "train")
    testset = ClimateSegLoader(args.data_folder, "test")
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    model = UNet(mesh_folder=args.mesh_folder, in_ch=16, out_ch=3, 
        max_level=args.max_level, min_level=args.min_level, fdim=args.fdim)
    model = model.cuda()
    # model = torch.nn.DataParallel(model)
    # pdb.set_trace()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = np.inf
    checkpoint_path = os.path.join(args.ckpt_dir, 'checkpoint_latest.pth.tar')
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, train_loader, optimizer, epoch)
        test(args, model, test_loader)
        if loss < best_loss:
            best_loss = loss
            is_best = True
        else:
            is_best = False
        save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_UNet")

        
if __name__ == "__main__":
    main()