from torch import nn
from ops import MeshConv
import torch.nn.functional as F
import os


class LeNet(nn.Module):
    def __init__(self, device, mesh_folder):
        super(LeNet, self).__init__()
        self.b = 16
        self.conv1 = MeshConv(1, self.b, device=device, mesh_file=os.path.join(mesh_folder, "icosphere_4.pkl"), stride=2)
        self.conv2 = MeshConv(self.b, 2*self.b, device=device, mesh_file=os.path.join(mesh_folder, "icosphere_3.pkl"), stride=2)
        self.conv3 = MeshConv(2*self.b, 3*self.b, device=device, mesh_file=os.path.join(mesh_folder, "icosphere_2.pkl"), stride=2)
        self.conv4 = MeshConv(3*self.b, 4*self.b, device=device, mesh_file=os.path.join(mesh_folder, "icosphere_1.pkl"), stride=2)
        self.csize = 4*self.b*self.conv4.nv_prev
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.csize, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4_drop(self.conv4(x).unsqueeze(-1)))
        x = x.view(-1, self.csize)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
