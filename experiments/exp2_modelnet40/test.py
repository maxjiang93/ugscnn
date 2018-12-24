# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
from collections import OrderedDict
from time import time
import numpy as np

from dataset import ModelNet, CacheNPY, ToMesh, ProjectOnSphere


def main(sp_mesh_dir, sp_mesh_level, log_dir, data_dir, eval_time,
         dataset, partition, batch_size, jobs, tiny, feat, no_cuda, neval):
    torch.set_num_threads(jobs)
    print("Running on {} CPU(s)".format(torch.get_num_threads()))
    if no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    sp_mesh_file = os.path.join(sp_mesh_dir, "icosphere_{}.pkl".format(sp_mesh_level))


    # Load the model
    loader = importlib.machinery.SourceFileLoader('model',"model.py")
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    num_classes = int(dataset[-2:])
    if not tiny:
        model = mod.Model(num_classes, mesh_folder=sp_mesh_dir, feat=feat)
    else:
        model = mod.Model_tiny(num_classes, mesh_folder=sp_mesh_dir, feat=feat)

    # load checkpoint
    ckpt = os.path.join(log_dir, "state.pkl")
    if no_cuda:
        pretrained_dict = torch.load(ckpt, map_location=lambda storage, loc:storage)
    else:
        pretrained_dict = torch.load(ckpt)

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

    load_my_state_dict(model, pretrained_dict)  
    model.to(device)
  

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print("{} paramerters in the last layer".format(sum(x.numel() for x in model.out_layer.parameters())))

    # Load the dataset
    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(prefix="sp{}_".format(sp_mesh_level), transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=False, random_translation=0),
            ProjectOnSphere(meshfile=sp_mesh_file, dataset=dataset, normalize=True)
        ]
    ), sp_mesh_dir=sp_mesh_dir, sp_mesh_level=sp_mesh_level)

    transform_test = CacheNPY(prefix="sp{}_".format(sp_mesh_level), transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=False, random_translation=0),
            ProjectOnSphere(meshfile=sp_mesh_file, dataset=dataset, normalize=True)
        ]
    ), sp_mesh_dir=sp_mesh_dir, sp_mesh_level=sp_mesh_level)

    if dataset == 'modelnet10':
        def target_transform(x):
            classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
            return classes.index(x)
    elif dataset == 'modelnet40':
        def target_transform(x):
            classes = ['airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa', 'tv_stand', 'bathtub', 'car', 'door',
                       'lamp', 'piano', 'stairs', 'vase', 'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool',
                       'wardrobe', 'bench', 'cone', 'flower_pot', 'mantel', 'radio', 'table', 'xbox', 'bookshelf', 'cup',
                       'glass_box', 'monitor', 'range_hood', 'tent', 'bottle', 'curtain', 'guitar', 'night_stand', 'sink', 'toilet']
            return classes.index(x)
    else:
        print('invalid dataset. must be modelnet10 or modelnet40')
        assert(0)

    test_set = ModelNet(data_dir, dataset=dataset, partition='test', transform=transform_test, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=jobs, pin_memory=True, drop_last=False)

    def test_step(data, target):
        model.eval()
        data, target = data.to(device), target.to(device)

        t = time()
        prediction = model(data)
        dt = time() - t
        loss = F.nll_loss(prediction, target)

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item(), dt

    # test
    total_loss = 0
    total_correct = 0
    count = 0
    total_time = []
    for batch_idx, (data, target) in enumerate(test_loader):
        loss, correct, dt = test_step(data, target)
        total_time.append(dt)
        total_loss += loss
        total_correct += correct
        count += 1
        if eval_time and count >= neval:
            print("Time per batch: {} secs".format(np.mean(total_time[10:])))
            break
    if not eval_time:
        print("[Test] <LOSS>={:.2} <ACC>={:2}".format(total_loss / (count+1), total_correct / len(test_set)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--partition", choices={"test", "train"}, default="train")
    parser.add_argument("--dataset", choices={"modelnet10", "modelnet40"}, default="modelnet40")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--sp_mesh_dir", type=str, default="../../mesh_files")
    parser.add_argument("--sp_mesh_level", type=int, default=5)
    parser.add_argument("--feat", type=int, default=32)
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--eval_time", action='store_true')
    parser.add_argument('--neval', type=int, default=64, help='Number of evaluations to run for timing.')

    args = parser.parse_args()

    main(**args.__dict__)