
# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import numpy as np
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.sub_modules.voxelnext import VoxelResBackBone8xVoxelNeXt
from opencood.models.sub_modules.height_compression import HeightCompression2D
import torch

class VoxelNeXTWhere2comm(nn.Module):
    def __init__(self, args):
        super(VoxelNeXTWhere2comm, self).__init__()
        
        self.batch_size = args['batch_size']

        # PIllar VFE
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        
        # sparse 3d backbone
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(4, args['backbone_3d'],
                                            args['grid_size'])
        # # height compression
        self.height_compression = HeightCompression2D(args['height_compression'])
        
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 256)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 256)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.mean_vfe.parameters():
            p.requires_grad = False

        if hasattr(self, "scatter"):
            for p in self.scatter.parameters():
                p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        
        aligned_gt_boxes = []
        for i in range(self.batch_size):
            num_cav_col = record_len[i]
            # if num_cav_col == 0:
            #     continue
            gt_boxes_col = data_dict['object_bbx_center'][i]
            # aligned_gt_boxes.append(gt_boxes_col)
            for _ in range(num_cav_col):
                aligned_gt_boxes.append(gt_boxes_col)
        aligned_gt_boxes = torch.stack(aligned_gt_boxes, dim=0)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'batch_size': torch.cumsum(record_len, dim=0)[-1].item(),
                      'gt_boxes': aligned_gt_boxes.float()}
        # n, 4 -> n, c
        batch_dict = self.mean_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict["batch_size"] = self.batch_size
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory，降采样
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        # if self.dcn:
        #     spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)
            
            
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm
                       }
        output_dict.update(result_dict)
        
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict
