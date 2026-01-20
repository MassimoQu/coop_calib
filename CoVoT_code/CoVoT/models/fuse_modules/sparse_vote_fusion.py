import torch

from torch.nn import functional as F
from torch.autograd import Variable

import copy
import spconv.pytorch.functional as spF
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from ...utils.spconv_utils import replace_feature
import torch.nn as nn
import numpy as np

class SparseVoteFusion(nn.Module):
    def __init__(self,fusion_channel):
        super(SparseVoteFusion,self).__init__()
        self.fuse_net = self.build_fusion_net(fusion_channel)

    def build_fusion_net(self,channels=[128,96,64]):
        fusion_net = spconv.SparseSequential(
            spconv.SubMConv2d(channels[0], channels[1], 3, stride=1, padding=1, bias=False, algo=ConvAlgo.Native),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(True),
            spconv.SubMConv2d(channels[1],channels[2], 3, stride=1, padding=1, bias=False, algo=ConvAlgo.Native),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(True),
        )
        return fusion_net

    def forward(self,pts_feats_ego, pts_feats_col):
        # pts_feats = pts_feats + pts_feats_inf
        # "cat zero and add"
        pts_feats_ego= replace_feature(pts_feats_ego,\
                        torch.cat([pts_feats_ego.features,\
                                    torch.zeros_like(pts_feats_ego.features)],dim=1))
        
        pts_feats_col= replace_feature(pts_feats_col,\
                        torch.cat([torch.zeros_like(pts_feats_col.features),\
                                    pts_feats_col.features],dim=1))                                                         
        pts_feats_ego=spF.sparse_add_hash_based(pts_feats_ego,pts_feats_col)
        pts_feats_ego= self.fuse_net(pts_feats_ego)

        return pts_feats_ego
