from __future__ import absolute_import
from lib2to3.refactor import get_fixers_from_package

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import *

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    
class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout, depth_emd=False):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)
        ### depth embedding ###
        self.depth_emd = depth_emd
        if depth_emd:
            self.depth_proj = torch.nn.Linear(emd_dim,hid_dim)


    def forward(self, x, temb, dpemb=None):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        if self.depth_emd and (dpemb is not None):
            out = out + self.depth_proj(nonlinearity(dpemb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

class GCNdiff(nn.Module):
    def __init__(self, adj, config):
        super(GCNdiff, self).__init__()
        
        self.adj = adj
        self.config = config
        ### load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4
                
        ### Generate Graphformer  ###
        self.n_layers = num_layers

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1, depth_emd=con_gcn.depth_emd))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)
        
        
        ### diffusion configuration  ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])

        ### 2d kpt embedding ###
        self.kpt2demb = nn.Module()
        self.kpt2demb.dense = nn.ModuleList([
            torch.nn.Linear(3,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])
        self.kpt2demb_proj = torch.nn.Linear(self.emd_dim, self.hid_dim)

        ### depth embedding ###
        self.depth_emd = con_gcn.depth_emd
        if con_gcn.depth_emd:
            self.dpemb = nn.Module()
            self.dpemb.dense = nn.ModuleList([
                torch.nn.Linear(self.hid_dim,self.emd_dim),
                torch.nn.Linear(self.emd_dim,self.emd_dim),
            ])

    def forward(self, x, mask, t, kpt2ds=None, x_depth=None):
        # timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        kpt2dembs=[]
        if kpt2ds is not None:
            for v in range(2):
                kpt2demb = self.kpt2demb.dense[0](kpt2ds[:,v])
                kpt2demb = nonlinearity(kpt2demb)
                kpt2demb = self.kpt2demb.dense[1](kpt2demb)
                kpt2demb = self.kpt2demb_proj(nonlinearity(kpt2demb))
                kpt2dembs.append(kpt2demb)
        
        out = self.gconv_input(x, self.adj)
        if kpt2ds is not None:
            for v in range(2):
                out = out + kpt2dembs[v]
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out, temb)
        out = self.gconv_output(out, self.adj)
        return out