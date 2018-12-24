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
from model import SphericalUNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

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
        # print("Intersect: ", intersect, " Union: ", union)
        iou_ = intersect / union
        iou.append(iou_)
    return np.array(iou)

def accuracy(pred_cls, true_cls, nclass=3):
    """
    compute per-node classification accuracy
    """
    accu = []
    for i in range(nclass):
        intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
        thiscls = (true_cls == i).sum().item()
        accu.append(intersect / thiscls)
    return np.array(accu)

def test(args, model, test_loader):
    w = torch.tensor([0.001020786726132422, 0.9528737404907279, 0.04610547278313972]).cuda()
    model.eval()
    test_loss = 0
    ious = np.zeros(3)
    accus = np.zeros(3)
    count = 0
    pred_ = []
    true_ = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]
            test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            iou = iou_score(pred, target, nclass=3)
            accu = accuracy(pred, target, nclass=3)
            ious += iou * n_data
            accus += accu * n_data
            count += n_data
            pred_.append(pred.cpu().numpy())
            true_.append(target.cpu().numpy())
    ious /= count
    accus /= count
    test_loss /= count
    pred_ = np.concatenate(pred_, axis=0)
    true_ = np.concatenate(true_, axis=0)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}; Accu: {:.4f}, {:.4f}, {:.4f}; IoU: {:.4f}, {:.4f}, {:.4f}\r'.format(
        test_loss, accus[0], accus[1], accus[2], iou[0], iou[1], iou[2]))

    # plot and save confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(true_.ravel(), pred_.ravel())
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["Background", "TC", "AR"], normalize=True, title='Normalized confusion matrix')
    plt.savefig("confusion_matrix.png", dpi=200)



def export(args, model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            break
    if args.export_file:
        print("Saving export file...")
        np.savez(args.export_file, data=data, labels=target, predict=pred)
        print("Success!")
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Climate Segmentation Example')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="../../mesh_files",
                        help='path to mesh folder (default: ../../mesh_files)')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_folder', type=str, default="data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--max_level', type=int, default=5, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=8, help='filter dimensions')
    parser.add_argument('--export_file', type=str, default='', help='file name for exporting samples')
    parser.add_argument('--partition', type=str, default='test', choices=['test, val, train'], help='data partition to use')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    model = SphericalUNet(mesh_folder=args.mesh_folder, in_ch=16, out_ch=3, 
        max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
    model = nn.DataParallel(model)
    model = model.cuda()

    # load checkpoint
    assert(os.path.isfile(args.ckpt))
    print("=> loading checkpoint '{}'".format(args.ckpt))
    resume_dict = torch.load(args.ckpt)
    def load_my_state_dict(self, state_dict, exclude='none'):
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

    load_my_state_dict(model, resume_dict['state_dict'])
    last_epoch = resume_dict['epoch']
    best_ap = resume_dict['best_ap']

    print("=> loaded checkpoint '{}' (epoch {} ap {:.03f}) "
          .format(args.ckpt, resume_dict['epoch'], best_ap))
    testset = ClimateSegLoader(args.data_folder, args.partition)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    if args.export_file:
        export(args, model, test_loader)
    else:
        test(args, model, test_loader)
        
if __name__ == "__main__":
    main()