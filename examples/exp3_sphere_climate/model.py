from torch import nn
from ops import MeshConv, MeshConv_transpose
import torch.nn.functional as F
import os
import torch
import pickle


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, mesh_file, bias=True):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            MeshConv(in_ch, out_ch, mesh_file, stride=1),
            nn.BatchNorm1d(out_ch).cuda(),
            nn.ReLU(inplace=True),
            MeshConv(out_ch, out_ch, mesh_file, stride=1),
            nn.BatchNorm1d(out_ch).cuda(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, mesh_file, bias=True):
        """
        use mesh_file for the mesh of one-level up
        """
        super(Up, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = MeshConv_transpose(int(in_ch/2), int(in_ch/2), mesh_file, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, mesh_file)
        self.mesh_file = mesh_file

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, mesh_file, bias=True):
        """
        use mesh_file for the mesh of one-level down
        """
        super(Down, self).__init__()
        pkl = pickle.load(open(mesh_file, "rb"))
        self.nv_prev = pkl['V'].shape[0]
        self.conv = DoubleConv(in_ch, out_ch, mesh_file)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mesh_file = mesh_file

    def forward(self, x):
        x = x[:, :, :self.nv_prev]
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, mesh_folder, in_ch, out_ch, max_level=7, min_level=0, fdim=16):
        super(UNet, self).__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level
        self.down = []
        self.up = []
        self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(max_level), stride=1)
        self.out_conv = MeshConv(fdim, out_ch, self.__meshfile(max_level), stride=1)
        # Downward path
        for i in range(self.levels-1):
            self.down.append(Down(fdim*(2**i), fdim*(2**(i+1)), self.__meshfile(max_level-i-1)))
        self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), self.__meshfile(min_level)))
        # Upward path
        for i in range(self.levels-1):
            self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), self.__meshfile(min_level+i+1)))
        self.up.append(Up(fdim*2, fdim, self.__meshfile(max_level)))

    def forward(self, x):
        x_ = [self.in_conv(x)]
        for i in range(self.levels):
            x_.append(self.down[i](x_[-1]))
        x = self.up[0](x_[-1], x_[-2])
        for i in range(self.levels-1):
            x = self.up[i+1](x, x_[-3-i])
        x = self.out_conv(x)
        return x

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))