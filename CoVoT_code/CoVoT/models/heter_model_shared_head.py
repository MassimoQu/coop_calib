# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# In this heterogeneous version, feature align start before backbone.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
from opencood.models.comm_modules.where2comm import Communication
import torchvision
from opencood.models.sub_modules.codebook import ChannelCompressor
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.point_pillar import PointPillar

class HeterModelSharedhead(nn.Module):
    def __init__(self, args):
        super(HeterModelSharedhead, self).__init__()
        self.args = args

        self.cav_range = args['lidar_range']
        #communication
        self.naive_communication = Communication(args['fusion_args']['communication'])
        # setup each modality model
        
        self.sensor_backbone = PointPillar(args)

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1


        """
        Fusion, by default multiscale fusion: 
        """
        self.backbone = ResNetBEVBackbone(args['fusion_backbone'], 64)
        self.fusion_net = nn.ModuleList()

        for i in range(len(args['fusion_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))

        """
        Codebook
        """
        self.multi_channel_compressor_flag = False
        if 'multi_channel_compressor' in args and args['multi_channel_compressor']:
            # print('multi_channel_compressor_flag')
            self.multi_channel_compressor_flag = True

        channel = 64
        p_rate = 0.0
        seg_num = args['codebook']['seg_num']
        if args['codebook']['r'] == 1:
            dict_size = [args['codebook']['dict_size']]
        elif args['codebook']['r'] == 2:
            dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size']]
        else:
            dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size'], args['codebook']['dict_size']]
        self.multi_channel_compressor = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                          {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                           "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                           "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
        # print("codebook:", self.multi_channel_compressor_flag)
        # print("seg_num: ", seg_num)        
        # print("dict_size: ", args['codebook']['dict_size'])

        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_before_head = nn.Conv2d(64, args['anchor_number'],
                                  kernel_size=1)
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        if 'freeze_for_codebook' in args and args['freeze_for_codebook']:
            self.freeze_for_codebook()            
        if 'freeze_codebook' in args and args['freeze_codebook']:
            print('freezecodebook')
            self.freeze_codebook()

        # check again which module is not fixed.
        check_trainable_module(self)
        # print('----------- Training Parameters -----------')
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
        # print('----------- Training Parameters -----------')


    def regroup(self, x, record_len):
        #print(x)
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def freeze_codebook(self):
        for p in self.multi_channel_compressor.parameters():
            p.requires_grad_(False)
            
    def freeze_for_codebook(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.shrink_conv.parameters():
            p.requires_grad_(False)
        for p in self.cls_head.parameters():
            p.requires_grad_(False)
        for p in self.reg_head.parameters():
            p.requires_grad_(False)
        for p in self.dir_head.parameters():
            p.requires_grad_(False)        
        for p in self.fusion_net.parameters():
            p.requires_grad_(False)                
        for p in self.naive_communication.parameters():
            p.requires_grad_(False)                      
        for p in self.naive_communication.parameters():
            p.requires_grad_(False)

    def model_train_init(self):
        if self.stage2_added_modality is None:
            return
        """
        In stage 2, only ONE modality's aligner is trainable.
        We first fix all modules, and set the aligner trainable.
        """
        # fix all modules
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, data_dict):
        output_dict = {}
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len'] 
        
        # print(agent_modality_list)
        modality_feature_dict = {}
        modality_feature_dict['lidar'] = self.sensor_backbone(data_dict)['spatial_features_2d']

        heter_feature_2d = modality_feature_dict['lidar']
        cls_preds_before_fusion = self.cls_before_head(heter_feature_2d)


        """
        Codebook Part
        """
        '''get_feature_flag = True
        if get_feature_flag:
            save_path = "/GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/feature_folder/"
            print("get feature", num)
            torch.save(heter_feature_2d, os.path.join(save_path,'feature%d.pt' % (num)))
            #torch.save(record_len, os.path.join(save_path,'record_len%d.pt' % (num)))
            #codebook_loss = 0.0
        '''
        
        N, C, H, W = heter_feature_2d.shape
        #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
        # import pdb
        # pdb.set_trace()
        if self.multi_channel_compressor_flag:
            #print("------------Codebook information------------")
            heter_feature_2d_gt = heter_feature_2d.clone()
            heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
            heter_feature_2d, _, _, codebook_loss = self.multi_channel_compressor(heter_feature_2d)
            heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            heter_feature_2d_gt_split = self.regroup(heter_feature_2d_gt, record_len)
            shape_num = 0
            #print("record_len: ", record_len)
            #print("heter_feature_2d_gt_shape: ", heter_feature_2d_gt.shape)
            #print("heter_feature_2d_gt_split: ", len(heter_feature_2d_gt_split))
            for index in range(len(heter_feature_2d_gt_split)):
                #print("heter_feature_2d_gt_split_shape: ", heter_feature_2d_gt_split[index].shape)
                #print(heter_feature_2d_gt_split[index].shape[0])
                #print(shape_num)
                heter_feature_2d[shape_num] = heter_feature_2d_gt_split[index][0]
                shape_num = shape_num + heter_feature_2d_gt_split[index].shape[0]
                
            #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
            output_dict.update({'codebook_loss': codebook_loss})
            #print('codebook_loss', codebook_loss)
            #print("------------Codebook information------------")

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """

        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)

        batch_confidence_maps = self.regroup(cls_preds_before_fusion, record_len)

        
        #print('confidencemap:{}'.format(batch_confidence_maps[0].size()))
        #print("fusion2")
        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
        for i in range(len(feature_list)):
              #print(x.size())
              feature_list[i] = feature_list[i] * communication_masks
              communication_masks = F.max_pool2d(communication_masks, kernel_size=2)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

       

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'comm_rates': communication_rates})
        # print("comm_rates",output_dict['comm_rates'])
        return output_dict