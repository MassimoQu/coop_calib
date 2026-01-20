# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.voxelnext import VoxelResBackBone8xVoxelNeXt
from opencood.models.sub_modules.spconv_sps_quantisizer import VoxelResSPSQuantiseizer
from opencood.models.sub_modules.height_compression import HeightCompression2D
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

class PointQuantization(object):
    def __init__(self, voxel_size, quantize_coords_range, q_delta=1.6e-5):
        self.voxel_size = np.array(voxel_size)
        self.quantize_coords_range = quantize_coords_range
        self.low_bound = np.array(quantize_coords_range[:3])
        self.q_delta = q_delta
    # 2340x2304x16 = 8.5e7 log2(8.5e7) = 26.4 4+1=5B for each point
    def __call__(self, points):
        device = points.device
        low_bound = torch.tensor(self.low_bound).to(device)
        voxel_size = torch.tensor(self.voxel_size).to(device)
        points[:, :3] -= (low_bound + voxel_size / 2)
        points[:, :3] = torch.round(points[:, :3] / voxel_size)
        points[:, :3] *= voxel_size
        points[:, :3] += (low_bound + voxel_size / 2)
        "we assume that intensity is always ≥ 0 and quantisize uniformly"
        points[:,3] = torch.round(points[:,3]/self.q_delta)*self.q_delta
        return points
    
class VoxelNeXtSecondHybrid(nn.Module):
    def __init__(self, args):
        super(VoxelNeXtSecondHybrid, self).__init__()

        self.batch_size = args['batch_size']
        # mean_vfe
        self.pre_processer = build_preprocessor(args['preprocess'], False)
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.pruning = None
        if 'pruning' in args and 'pruning_ratio' in args:
            self.pruning = nn.ModuleList()
            for i in range(len(args['pruning_ratio'])):
                self.pruning.append(VoxelResSPSQuantiseizer(args['pruning'], args['pruning_ratio'][i]))
        self.pruning_mean_vfe = MeanVFE(args['mean_vfe'], 4)

        # quantization
        self.quant_level = args['quant_level']
        self.quantize = []
        for quant_level in self.quant_level:
            self.quantize.append(PointQuantization(quant_level,args['point_cloud_range']))
        self.quantize_switch = args['quantize_switch']
        # sparse 3d backbone
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(4, args['backbone_3d'],
                                            args['grid_size'])
        # # height compression
        self.height_compression = HeightCompression2D(args['height_compression'])
        # base ben backbone
        self.backbone_2d = BaseBEVBackbone(args['base_bev_backbone'], 256)

        # head
        self.cls_head = nn.Conv2d(256 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256 * 2, 7 * args['anchor_num'],
                                  kernel_size=1)
        self.raw = args['raw']
        if args['backbone_fix']:
            self.backbone_fix()
    
    def quantization(self, points, level):
        return self.quantize[level](points)
        
    def voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.cpu().numpy()
            res_voxels, res_coors, res_num_points = self.pre_processer.preprocess(res)
            res_voxels = torch.tensor(res_voxels, dtype=torch.float32).to(points[0].device)
            res_coors = torch.tensor(res_coors, dtype=torch.int32).to(points[0].device)
            res_num_points = torch.tensor(res_num_points, dtype=torch.int32).to(points[0].device)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)

        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def extract_pts_feats(self, point_list):
        voxels, num_points, coors = self.voxelize(point_list)
        batch_dict = dict()
        batch_dict['batch_size'] = len(point_list)
        batch_dict['voxels'] = voxels
        batch_dict['voxel_features'] = voxels
        batch_dict['voxel_coords'] =  coors
        batch_dict['voxel_num_points'] = num_points
        return batch_dict

    def backbone_fix(self):
        for param in self.mean_vfe.parameters():
            param.requires_grad = False

        for param in self.backbone_3d.parameters():
            param.requires_grad = False
        for param in self.height_compression.parameters():
            param.requires_grad = False
        for param in self.backbone_2d.parameters():
            param.requires_grad = False

        # for param in self.cls_head.parameters():
        #     param.requires_grad = False
        # for param in self.reg_head.parameters():
        #     param.requires_grad = False

    def cav_pruning_fusion(self, batch_dict_col, data_dict):
        if self.pruning is not None:
            for i in range(len(self.pruning)):
                batch_dict_col = self.pruning_mean_vfe(batch_dict_col)
                batch_dict_col = self.pruning[i](batch_dict_col)
            # im_coord_col, im_voxel_col, batch_dict_col = self.pruning(batch_dict_col)
            im_coord_col = batch_dict_col['voxel_coords']
            im_voxel_col = batch_dict_col['voxels']
            # align important voxels to batchsize of ego vehicle
            rec_len_cum = torch.cumsum(batch_dict_col['record_len'], dim=0)
            rec_len_cum = torch.cat([torch.tensor([0]).to(rec_len_cum.device), rec_len_cum], dim=0)
            aligned_batch_idx = im_coord_col[:, 0].clone()
            for ego_batch_id in range(batch_dict_col['record_len'].shape[0]):
                aligned_batch_idx[(im_coord_col[:, 0] >= rec_len_cum[ego_batch_id]) \
                                & (im_coord_col[:, 0] < rec_len_cum[ego_batch_id+1])] = ego_batch_id
            im_coord_col[:, 0] = aligned_batch_idx
            compressed_point_col = []
            # Nxmax_points_per_voxelx4 -> N*max_points_per_voxelx4
            im_point_col = im_voxel_col.reshape(-1, im_voxel_col.shape[-1])
            batch_size = batch_dict_col['batch_size']
            # print('im_point_col:',im_point_col)
            for b_id in range(int(batch_size)):
                im_point_col_bid = im_point_col[im_coord_col[:, 0] == b_id]
                if im_point_col_bid.shape[0] == 0:
                    compressed_point_col.append(im_point_col_bid)
                    continue
                if self.quantize_switch:
                    im_point_col_bid = self.quantization(im_point_col_bid,0)
                compressed_point_col.append(im_point_col_bid)
            points = data_dict['lidar_ego_list']
            for i in range(len(compressed_point_col)):
                # print('points[i]:',points[i])
                points[i] = torch.cat([points[i], compressed_point_col[i].to(points[i].device)], dim=0)
            batch_dict = self.extract_pts_feats(points)
            loss_box_of_pts_sprs_pruning = batch_dict_col['loss_box_of_pts_sprs_pruning']
            batch_dict.update({'loss_box_of_pts_sprs_pruning': loss_box_of_pts_sprs_pruning})

            # # cat ego vehicle and cavs
            # batch_dict['voxel_features'] = torch.cat([batch_dict['voxel_features'], im_voxel_col], dim=0)
            # batch_dict['voxel_coords'] = torch.cat([batch_dict['voxel_coords'], im_coord_col], dim=0)

        else:
            # align all voxels to batchsize of ego vehicle
            rec_len_cum = torch.cumsum(batch_dict_col['record_len'], dim=0)
            rec_len_cum = torch.cat([torch.tensor([0]).to(rec_len_cum.device), rec_len_cum], dim=0)
            aligned_batch_idx = batch_dict_col['voxel_coords'][:, 0].clone()
            for ego_batch_id in range(batch_dict_col['record_len'].shape[0]):
                aligned_batch_idx[(batch_dict_col['voxel_coords'][:, 0] >= rec_len_cum[ego_batch_id]) \
                                & (batch_dict_col['voxel_coords'][:, 0] < rec_len_cum[ego_batch_id+1])] = ego_batch_id
            batch_dict_col['voxel_coords'][:, 0] = aligned_batch_idx
            # print('aligned_batch_idx:',aligned_batch_idx)
            # cat ego vehicle and cavs
            points = data_dict['lidar_ego_list']
            batch_dict = self.extract_pts_feats(points)
            batch_dict['voxel_features'] = torch.cat([batch_dict['voxel_features'], batch_dict_col['voxel_features']], dim=0)
            batch_dict['voxel_coords'] = torch.cat([batch_dict['voxel_coords'], batch_dict_col['voxel_coords']], dim=0)

        # # sort voxels by batch index
        # _, indices = batch_dict['voxel_coords'][:, 0].sort()
        # batch_dict['voxel_coords'] = batch_dict['voxel_coords'][indices]
        # batch_dict['voxel_features'] = batch_dict['voxel_features'][indices]
        # print('voxel_features:',batch_dict['voxel_features'].shape)
        # # remove duplicate voxels and corresponding features
        # unique_coords, inverse_indices = torch.unique(batch_dict['voxel_coords'], \
        #             return_counts=False, return_inverse=True, dim=0)
        # batch_dict['voxel_coords'] = unique_coords
        # print('voxel_coords:',batch_dict['voxel_coords'])
        # batch_dict['voxel_features'] = batch_dict['voxel_features'][inverse_indices]        
        # batch_dict['voxel_num_points'] = torch.tensor(batch_dict['voxel_features'].shape[0],\
        #             dtype=torch.int32).to(batch_dict['voxel_features'].device)
        return batch_dict

    def forward(self, data_dict):
        if self.raw == True:
            # cat raw points of ego vehicle and cavs
            ego_lidar = data_dict['lidar_ego_list']
            col_lidar = data_dict['lidar_col_list']
            num_cav_col = data_dict['record_len']
            rec_len_cum = torch.cumsum(num_cav_col, dim=0)
            rec_len_cum = torch.cat([torch.tensor([0]).to(rec_len_cum.device), rec_len_cum], dim=0)
            fused_lidar_list = []
            for ego_batch_id in range(num_cav_col.shape[0]):
                fused_lidar_batch = []
                fused_lidar_batch.append(ego_lidar[ego_batch_id])
                if num_cav_col[-1] > 0:
                    if rec_len_cum[ego_batch_id] != rec_len_cum[ego_batch_id+1]:
                        fused_col_lidar_batch = torch.cat(col_lidar[rec_len_cum[ego_batch_id]:rec_len_cum[ego_batch_id+1]])
                        # if self.quantize_switch:
                        #     # uniformly sample points to 20% of original points
                        #     fused_col_lidar_batch = fused_col_lidar_batch[torch.randperm(fused_col_lidar_batch.shape[0])[:int(fused_col_lidar_batch.shape[0]*0.25)]]
                        fused_lidar_batch.append(fused_col_lidar_batch)
                fused_lidar = torch.cat(fused_lidar_batch, dim=0)
                fused_lidar_list.append(fused_lidar)
            batch_dict = self.extract_pts_feats(fused_lidar_list)
            assert batch_dict['batch_size'] == self.batch_size
            batch_dict.update({'batch_size': self.batch_size,
                            'gt_boxes': data_dict['object_bbx_center'].float()})



        else:
            # data processing for cavs and ego vehicle
            if len(data_dict['lidar_col_list']) > 0:
                # print('data_dict_col:',data_dict['lidar_col_list'])
                batch_dict_col = self.extract_pts_feats(data_dict['lidar_col_list'])

                record_len = data_dict['record_len'].cpu().numpy()
                batch_size = len(record_len)
                # repeat gtboxes to algn with record_len
                aligned_gt_boxes = []
                for i in range(batch_size):
                    num_cav_col = record_len[i]
                    if num_cav_col == 0:
                        continue
                    gt_boxes_col = data_dict['object_bbx_center'][i]
                    for _ in range(num_cav_col):
                        aligned_gt_boxes.append(gt_boxes_col)
                aligned_gt_boxes = torch.stack(aligned_gt_boxes, dim=0)
                # print('aligned_gt_boxes:',aligned_gt_boxes.shape)
                # print('record_len:',record_len)
                batch_dict_col.update({'record_len': data_dict['record_len'],
                                'gt_boxes': aligned_gt_boxes.float()})
                # print("gt_boxes:",batch_dict_col['gt_boxes'])
                batch_dict = self.cav_pruning_fusion(batch_dict_col,data_dict)
            else:
                batch_dict = self.extract_pts_feats(data_dict['lidar_ego_list'])
            assert batch_dict['batch_size'] == self.batch_size
            batch_dict.update({'batch_size': self.batch_size,
                            'gt_boxes': data_dict['object_bbx_center'].float()})
        
        # regular processing for fused voxels
        batch_dict = self.mean_vfe(batch_dict) 
        # print num voxels
        # print('voxel_features:',batch_dict['voxel_features'].shape)      
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'loss_box_of_pts_sprs': batch_dict['loss_box_of_pts_sprs'],}
                    #    'loss_box_of_pts_sprs_pruning': batch_dict['loss_box_of_pts_sprs_pruning']}
        if 'loss_box_of_pts_sprs_pruning' in batch_dict:
            output_dict.update({'loss_box_of_pts_sprs_pruning': batch_dict['loss_box_of_pts_sprs_pruning']})
        return output_dict