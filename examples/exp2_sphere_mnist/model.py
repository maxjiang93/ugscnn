from torch import nn
from ops import MeshConv, MeshIntegrate
import torch.nn.functional as F
import os


class LeNet(nn.Module):
    def __init__(self, mesh_folder, isotropic=False):
        super(LeNet, self).__init__()
        self.isotropic = isotropic
        self.b = 20
        self.conv1 = MeshConv(1, self.b, mesh_file=os.path.join(mesh_folder, "icosphere_4.pkl"), isotropic=self.isotropic, stride=2)
        self.conv2 = MeshConv(self.b, 2*self.b, mesh_file=os.path.join(mesh_folder, "icosphere_3.pkl"), isotropic=self.isotropic, stride=2)
        self.conv3 = MeshConv(2*self.b, 2*self.b, mesh_file=os.path.join(mesh_folder, "icosphere_2.pkl"), isotropic=self.isotropic, stride=2)
        self.csize = 2 * self.b * self.conv3.nv_prev

        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.csize, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_drop(self.conv3(x).unsqueeze(-1)))
        x = x.view(-1, self.csize)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
