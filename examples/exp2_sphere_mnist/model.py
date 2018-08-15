from torch import nn
from ops import MeshConv
import torch.nn.functional as F
import os


class LeNet(nn.Module):
    def __init__(self, device, mesh_folder):
        super(LeNet, self).__init__()
        self.b = 32
        self.conv1 = MeshConv(1, self.b, mesh_file=os.path.join(mesh_folder, "icosphere_3.pkl"), device=device, stride=2)
        self.conv2 = MeshConv(self.b, 2*self.b, mesh_file=os.path.join(mesh_folder, "icosphere_2.pkl"), device=device, stride=2)
        self.csize = 2*self.b*self.conv2.nv_prev
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.csize, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x).unsqueeze(-1)))
        x = x.view(-1, self.csize)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
