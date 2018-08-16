import math
import pickle, gzip

import torch
from torch import nn
from torch.nn.parameter import Parameter

from utils import sparse2tensor, spmatmul


class _MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        assert stride in [1, 2]
        super(_MeshConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.ncoeff = 4
        self.coeffs = Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        self.set_coeffs()
        # load mesh file
        pkl = pickle.load(open(mesh_file, "rb"))
        self.pkl = pkl
        G = sparse2tensor(pkl['G'])  # gradient matrix V->F, 3#F x #V
        NS = torch.tensor(pkl['NS'], dtype=torch.float32)  # north-south vector field, #F x 3
        EW = torch.tensor(pkl['EW'], dtype=torch.float32)  # east-west vector field, #F x 3
        self.register_buffer("G", G)
        self.register_buffer("NS", NS)
        self.register_buffer("EW", EW)
        
    def set_coeffs(self):
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class MeshConv(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        super(MeshConv, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        pkl = self.pkl
        if stride == 2:
            self.nv_prev = pkl['nv_prev']
            L = sparse2tensor(pkl['L'].tocsr()[:self.nv_prev].tocoo()) # laplacian matrix V->V
            F2V = sparse2tensor(pkl['F2V'].tocsr()[:self.nv_prev].tocoo())  # F->V, #V x #F
        else: # stride == 1
            self.nv_prev = pkl['V'].shape[0]
            L = sparse2tensor(pkl['L'].tocoo())
            F2V = sparse2tensor(pkl['F2V'].tocoo())
        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)
        
    def forward(self, input):
        # compute gradient
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        laplacian = spmatmul(input, self.L)
        identity = input[..., :self.nv_prev]
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        out = torch.stack(feat, dim=-1)
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        out += self.bias.unsqueeze(-1)
        return out


class MeshConv_transpose(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_file, stride=2, bias=True):
        assert(stride == 2)
        super(MeshConv_transpose, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        self.nv_prev = self.pkl['nv_prev']
        self.nv = self.pkl['V'].shape[0]
        self.nv_pad = self.nv - self.nv_prev
        L = sparse2tensor(pkl['L'].tocoo()) # laplacian matrix V->V
        F2V = sparse2tensor(pkl['F2V'].tocoo()) # F->V, #V x #F
        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)
        
    def forward(self, input):
        # pad input with zeros up to next mesh resolution
        ones_pad = torch.ones(*input.size()[:2], self.nv_pad).to(input.device)
        input = torch.cat((input, ones_pad), dim=-1)
        # compute gradient
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        laplacian = spmatmul(input, self.L)
        identity = input[..., :self.nv_prev]
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        out = torch.stack(feat, dim=-1)
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        out += self.bias.unsqueeze(-1)
        return out
