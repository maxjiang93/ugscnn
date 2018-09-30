import math
import argparse
import sys
sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import os
import shutil
import logging
from collections import OrderedDict

from ops import MeshConv
from loader import ClimateSegLoader
from model import UNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 1:
        os.remove(output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
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
        iou_ = intersect / union
        iou.append(iou_)
    return np.array(iou)

def average_precision(score_cls, true_cls, nclass=3):
    score = score_cls.cpu().numpy()
    true = label_binarize(true_cls.cpu().numpy().reshape(-1), classes=[0, 1, 2])
    score = np.swapaxes(score, 1, 2).reshape(-1, nclass)
    return average_precision_score(true, score)

def train(args, model, train_loader, optimizer, epoch, device, logger):
    # w = torch.tensor([0.001020786726132422, 0.9528737404907279, 0.04610547278313972]).to(device)
    # ratios: [0.97663559 0.00104578 0.02231863]
    w = torch.tensor([1.0,1.0,1.0]).to(device)
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
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    tot_loss /= count
    return tot_loss

def test(args, model, test_loader, device, logger):
    # w = torch.tensor([0.001020786726132422, 0.9528737404907279, 0.04610547278313972]).to(device)
    w = torch.tensor([1.0,1.0,1.0]).to(device)
    model.eval()
    test_loss = 0
    ious = np.zeros(3)
    aps = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]
            test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            iou = iou_score(pred, target, nclass=3)
            ap = average_precision(output, target)
            ious += iou * n_data
            aps += ap * n_data
            count += n_data
    ious /= count
    aps /= count
    test_loss /= count

    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Avg Precision: {:.4f}; MIoU: {:.4f}; IoU: {:.4f}, {:.4f}, {:.4f}; Avg loss: {:.4f}'.format(
        aps, np.mean(ious), ious[0], ious[1], ious[2], test_loss))
    return aps
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="../../mesh_files",
                        help='path to mesh folder (default: ../../mesh_files)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=7, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=16, help='filter dimensions')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # logger and snapshot current code
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    shutil.copy2(__file__, os.path.join(args.log_dir, "script.py"))
    shutil.copy2("model.py", os.path.join(args.log_dir, "model.py"))
    shutil.copy2("run.sh", os.path.join(args.log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    torch.manual_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    trainset = ClimateSegLoader(args.data_folder, "train")
    valset = ClimateSegLoader(args.data_folder, "val")
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    model = UNet(mesh_folder=args.mesh_folder, in_ch=16, out_ch=3, 
        max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
    model = nn.DataParallel(model)
    model.to(device)

    if args.resume:
        resume_dict = torch.load(args.resume)

        def load_my_state_dict(self, state_dict, exclude='out_layer'):
            from torch.nn.parameter import Parameter
     
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if exclude in name:
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)

        load_my_state_dict(model, resume_dict)  

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

    best_ap = 0
    checkpoint_path = os.path.join(args.log_dir, 'checkpoint_latest.pth.tar')
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, train_loader, optimizer, epoch, device, logger)
        ap = test(args, model, val_loader, device, logger)
        if ap > best_ap:
            best_ap = ap
            is_best = True
        else:
            is_best = False
        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

        save_checkpoint({
        'epoch': epoch,
        'state_dict': state_dict_no_sparse,
        'best_ap': best_ap,
        'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_UNet", logger)

if __name__ == "__main__":
    main()