import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("../../meshcnn")
from ops import MeshConv, DownSamp, ResBlock
import os


class Model(nn.Module):
    def __init__(self, nclasses, mesh_folder, feat=32):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_5.pkl")
        self.in_conv = MeshConv(6, feat, mesh_file=mf, stride=2)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(in_chan=feat, neck_chan=feat, out_chan=4*feat, level=4, coarsen=True, mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=4*feat, neck_chan=4*feat, out_chan=16*feat, level=3, coarsen=True, mesh_folder=mesh_folder)
        self.block3 = ResBlock(in_chan=16*feat, neck_chan=16*feat, out_chan=64*feat, level=2, coarsen=True, mesh_folder=mesh_folder)
        self.avg = nn.AvgPool1d(kernel_size=self.block3.nv_prev) # output shape batch x channels x 1
        self.out_layer = nn.Linear(64*feat, nclasses)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.squeeze(self.avg(x), dim=-1)
        x = F.dropout(x, training=self.training)
        x = self.out_layer(x)

        return F.log_softmax(x, dim=1)


# Uses a different progression of layer numbers for a smaller and leaner model
class Model_tiny(nn.Module):
    def __init__(self, nclasses, mesh_folder, feat=1):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_5.pkl")
        self.in_conv = MeshConv(6, feat, mesh_file=mf, stride=2)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(in_chan=feat, neck_chan=feat, out_chan=2*feat, level=4, coarsen=True, mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=2*feat, neck_chan=2*feat, out_chan=8*feat, level=3, coarsen=True, mesh_folder=mesh_folder)
        self.block3 = ResBlock(in_chan=8*feat, neck_chan=8*feat, out_chan=32*feat, level=2, coarsen=True, mesh_folder=mesh_folder)
        self.avg = nn.AvgPool1d(kernel_size=self.block3.nv_prev) # output shape batch x channels x 1
        self.out_layer = nn.Linear(32*feat, nclasses)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.squeeze(self.avg(x), dim=-1)
        x = F.dropout(x, training=self.training)
        x = self.out_layer(x)

        return F.log_softmax(x, dim=1)
