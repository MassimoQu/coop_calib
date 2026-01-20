# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib
import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.voxelnext import VoxelResBackBone8xVoxelNeXt
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from ..utils.spconv_utils import spconv, replace_feature
from torch.nn.init import kaiming_normal_
import copy

import torch

def convert_to_anchor_format(pred_dict):
    """
    将从稠密张量预测出的 pred_dict 转换为 anchor 格式，返回与原始形状一致的张量 (B, C, H, W)。
    """
    anchor_outputs = []
    for pred in pred_dict:
        # 提取中心点信息 (center_x, center_y, center_z)
        center_dense = pred['center']  # shape: (B, 4, H, W)
        center_xy1 = center_dense[:, :2, :, :]  # (B, 2, H, W)
        center_z1 = pred['center_z'][:, 0:1, :, :]  # 取出 z 坐标 (B, 1, H, W)
        center1 = torch.cat([center_xy1, center_z1], dim=1)  # (B, 3, H, W)
        center_xy2 = center_dense[:, 2:4, :, :]  # (B, 2, H, W)
        center_z2 = pred['center_z'][:, 1:2, :, :]  # 取出 z 坐标 (B, 1, H, W)
        center2 = torch.cat([center_xy2, center_z2], dim=1)  # (B, 3, H, W)

        # 提取尺寸信息 (w, h, l)
        dim_dense = pred['dim']  # shape: (B, 6, H, W)
        size1 = torch.stack([
            dim_dense[:, 0, :, :],  # w
            dim_dense[:, 2, :, :],  # h
            dim_dense[:, 4, :, :],  # l
        ], dim=1)  # (B, 3, H, W)
        size2 = torch.stack([
            dim_dense[:, 1, :, :],  # w
            dim_dense[:, 3, :, :],  # h
            dim_dense[:, 5, :, :],  # l
        ], dim=1)

        # 提取旋转角度 (theta)
        rot_dense = pred['rot']  # shape: (B, 2, H, W)
        theta1 = torch.atan2(rot_dense[:, 0, :, :], rot_dense[:, 1, :, :]).unsqueeze(1)  # (B, 1, H, W)
        theta2 = torch.atan2(rot_dense[:, 2, :, :], rot_dense[:, 3, :, :]).unsqueeze(1)

        # 提取置信度 (confidence)
        psm = pred['hm']  # shape: (B, 2, H, W)

        # 拼接所有 anchor 信息
        anchor_output = torch.cat([center1, size1, theta1, center2, size2, theta2], dim=1)  # (B, 14, H, W)
        anchor_outputs.append(anchor_output)

    # 返回拼接后的 anchors，保持原始形状
    return psm, torch.cat(anchor_outputs, dim=0)  # (B, 14, H, W)


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, anchors, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels*anchors, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).dense()

        return ret_dict

class VoxelNeXtEarly(nn.Module):
    def __init__(self, args):
        super(VoxelNeXtEarly, self).__init__()

        self.batch_size = args['batch_size']
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(4, args['backbone_3d'],
                                            args['grid_size'])
        # # height compression
        # self.height_compression = HeightCompression(args['height_compression'])
        # # base ben backbone
        # self.backbone_2d = BaseBEVBackbone(args['base_bev_backbone'], 256)

        # head (modified from VoxelNeXtHead)
        # self.cls_head = nn.Conv2d(256 * 2, args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head = nn.Conv2d(256 * 2, 7 * args['anchor_num'],
        #                           kernel_size=1)
        self.class_names = args['class_names']
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in args['class_names_each_head']:
            self.class_names_each_head.append([x for x in cur_class_names if x in self.class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in self.class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = args.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = args['separate_head']
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            # print('head_dict:', self.separate_head_cfg)
            cur_head_dict = copy.deepcopy(self.separate_head_cfg['head_dict'])
            cur_head_dict['hm'] = dict(out_channels=int(len(cur_class_names)/args['anchor_num']), num_conv=args['num_hm_conv'])
            # print('cur_head_dict:', cur_head_dict)
            self.heads_list.append(
                SeparateHead(
                    input_channels=args.get('shared_conv_channel', 256),
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    anchors=args['anchor_num'],
                    init_bias=-2.66,
                    use_bias=args.get('use_bias_before_norm', False),
                )
            )

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels


    def forward(self, data_dict):
        # print('data_dict_labels:', data_dict['object_bbx_center'].shape)
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        # print('voxel_features:', voxel_features.shape)
        # print('voxel_coords:', voxel_coords.shape)
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': self.batch_size,
                      'gt_boxes': data_dict['object_bbx_center'].float()
                      }

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        # a bev-free version
        # batch_dict = self.height_compression(batch_dict)
        # batch_dict = self.backbone_2d(batch_dict)

        x = batch_dict['encoded_spconv_tensor']
        
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        forward_ret_dict = {}
        # print('x_features:', x.features.shape)
        pred_dict = []
        for head in self.heads_list:
            pred_dict.append(head(x))
            
        psm, rm = convert_to_anchor_format(pred_dict)

        forward_ret_dict['batch_size'] = self.batch_size
        forward_ret_dict['psm'] = psm.to_dense()
        forward_ret_dict['rm'] = rm.to_dense()
        forward_ret_dict['spatial_shape'] = spatial_shape
        forward_ret_dict['batch_index'] = batch_index
        forward_ret_dict['voxel_indices'] = voxel_indices
        forward_ret_dict['spatial_indices'] = spatial_indices
        forward_ret_dict['num_voxels'] = num_voxels
        forward_ret_dict['loss_box_of_pts_sprs']=batch_dict['loss_box_of_pts_sprs']



        return forward_ret_dict