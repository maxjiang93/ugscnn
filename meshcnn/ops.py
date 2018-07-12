import math
import pickle, gzip

import torch
from torch import nn
from torch.nn.parameter import Parameter

from utils import sparse2tensor, spmatmul


class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_file, device=torch.device("cuda"), stride=1, bias=True):
        assert stride in [1, 2]
        super(MeshConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.ncoeff = 4
        self.coeffs = torch.Tensor(out_channels, in_channels, self.ncoeff)
        if torch.cuda.is_available():
            self.coeffs = self.coeffs.cuda()
        self.coeffs = Parameter(self.coeffs)
        self.set_coeffs()
        # load mesh file
        pkl = pickle.load(open(mesh_file, "rb"))
        self.G = sparse2tensor(pkl['G']).to(device)  # gradient matrix V->F, 3#F x #V
        self.NS = torch.tensor(pkl['NS'], dtype=torch.float32).to(device)  # north-south vector field, #F x 3
        self.EW = torch.tensor(pkl['EW'], dtype=torch.float32).to(device)  # east-west vector field, #F x 3
        if stride == 2:
            self.nv_prev = pkl['nv_prev']
            self.L = sparse2tensor(pkl['L'].tocsr()[:self.nv_prev].tocoo()).to(device) # laplacian matrix V->V
            self.F2V = sparse2tensor(pkl['F2V'].tocsr()[:self.nv_prev].tocoo()).to(device)  # F->V, #V x #F
        else: # stride == 1
            self.nv_prev = pkl['V'].shape[0]
            self.L = sparse2tensor(pkl['L'].tocoo()).to(device)
            self.F2V = sparse2tensor(pkl['F2V'].tocoo()).to(device)
        
    def set_coeffs(self):
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
        # map gradient components to vertex
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)
        laplacian = spmatmul(input, self.L)
        identity = input[..., :self.nv_prev]
        res = torch.stack((identity, grad_vert_ew, grad_vert_ns, laplacian), dim=-1)
        output = torch.sum(torch.sum(torch.mul(res.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        return output


def MeshConv_transpose(nn.module):
    def __init__(self, in_channels, out_channels, mesh_file, device=torch.device("cuda"), stride=2, bias=True):
        assert(stride == 2)
        super(MeshConv_transpose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.ncoeff = 4
        self.coeffs = torch.Tensor(out_channels, in_channels, self.ncoeff)
        if torch.cuda.is_available():
            self.coeffs = self.coeffs.cuda()
        self.coeffs = Parameter(self.coeffs)
        self.set_coeffs()
        # load mesh file
        pkl = pickle.load(open(mesh_file, "rb"))
        self.G = sparse2tensor(pkl['G']).to(device)  # gradient matrix V->F, 3#F x #V
        self.NS = torch.tensor(pkl['NS'], dtype=torch.float32).to(device)  # north-south vector field, #F x 3
        self.EW = torch.tensor(pkl['EW'], dtype=torch.float32).to(device)  # east-west vector field, #F x 3
        self.nv_prev = pkl['nv_prev']
        self.nv = pkl['V'].shape[0]
        self.nv_pad = self.nv - self.nv_prev
        self.L = sparse2tensor(pkl['L'].tocoo()).to(device) # laplacian matrix V->V
        self.F2V = sparse2tensor(pkl['F2V'].tocoo()).to(device) # F->V, #V x #F
        
    def set_coeffs(self):
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        # pad input with zeros up to next mesh resolution
        input = torch.cat((input, torch.ones(*input.size()[:2], self.nv_pad)))
        # compute gradients
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
        # map gradient components to vertex
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)
        # compute laplacian
        laplacian = spmatmul(input, self.L)
        identity = input[..., :self.nv_prev]
        res = torch.stack((identity, grad_vert_ew, grad_vert_ns, laplacian), dim=-1)
        output = torch.sum(torch.sum(torch.mul(res.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        return output
