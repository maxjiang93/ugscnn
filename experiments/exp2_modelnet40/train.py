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

from dataset import ModelNet, CacheNPY, ToMesh, ProjectOnSphere


def main(sp_mesh_dir, sp_mesh_level, log_dir, model_path, augmentation, decay, data_dir,
         dataset, partition, batch_size, learning_rate, num_workers, epochs, pretrain, feat, rand_rot):
    arguments = copy.deepcopy(locals())

    sp_mesh_file = os.path.join(sp_mesh_dir, "icosphere_{}.pkl".format(sp_mesh_level))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2(model_path, os.path.join(log_dir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    num_classes = int(dataset[-2:])
    # model = mod.Model(num_classes, mesh_folder=sp_mesh_dir, feat=feat)
    model = mod.Model_tiny(num_classes, mesh_folder=sp_mesh_dir, feat=feat)
    model = nn.DataParallel(model)
    model.cuda()

    if pretrain:
        pretrained_dict = torch.load(pretrain)

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

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in model.module.out_layer.parameters())))

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

    train_set = ModelNet(data_dir, dataset=dataset, partition='train', transform=transform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_set = ModelNet(data_dir, dataset=dataset, partition='test', transform=transform_test, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.7)


    def train_step(data, target):
        model.train()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    def test_step(data, target):
        model.eval()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    def get_learning_rate(epoch):
        limits = [100, 200]
        lrs = [1, 0.1, 0.01]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate

    for epoch in range(epochs):

        # lr = get_learning_rate(epoch)
        # logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
        # for p in optimizer.param_groups:
        #     p['lr'] = lr
        if decay:
            scheduler.step()
        # training
        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, correct = train_step(data, target)

            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / (batch_idx + 1),
                correct / len(data), total_correct / len(data) / (batch_idx + 1),
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

        # test
        total_loss = 0
        total_correct = 0
        count = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            loss, correct = test_step(data, target)
            total_loss += loss
            total_correct += correct
            count += 1
        logger.info("[Epoch {} Test] <LOSS>={:.2} <ACC>={:2}".format(
            epoch, total_loss / (count+1), total_correct / len(test_set)))

        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)
        torch.save(state_dict_no_sparse, os.path.join(log_dir, "state.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--partition", choices={"test", "train"}, default="train")
    parser.add_argument("--dataset", choices={"modelnet10", "modelnet40"}, default="modelnet40")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--sp_mesh_dir", type=str, default="../../mesh_files")
    parser.add_argument("--sp_mesh_level", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--feat", type=int, default=32)
    parser.add_argument("--rand_rot", action='store_true')
    parser.add_argument("--decay", action='store_true')
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()

    main(**args.__dict__)