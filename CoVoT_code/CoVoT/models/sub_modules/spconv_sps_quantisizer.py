from functools import partial
import torch
import torch.nn as nn
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from .pruning_block import DynamicFocalPruningDownsample
from .voxelnext import SparseSequentialBatchdict,PostActBlock,\
    SparseBasicBlock
"we use sparse pruning to choose important points"

class VoxelResSPSQuantiseizer(nn.Module):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "spconv", "spconv"]
    
    def __init__(self, model_cfg, pruning_ratio, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        input_channels = model_cfg['input_channels']
        grid_size = np.array(model_cfg['grid_size']) # Z, Y, X
        spconv_kernel_sizes = model_cfg['spconv_kernel_sizes'] if 'spconv_kernel_sizes' in model_cfg else [3, 3, 3]
        self.downsample_pruning_ratio = pruning_ratio 
        assert isinstance(self.downsample_pruning_ratio, list)
        channels = model_cfg['channels'] if 'channels' in model_cfg else [16, 32, 64]
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.point_cloud_range = model_cfg['point_cloud_range'] if 'point_cloud_range' in model_cfg else [-3, -46.08, 0, 1, 46.08, 92.16]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        # self.conv1 = SparseSequentialBatchdict(
        #     SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        #     SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        #     SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        # )

        self.conv1 = SparseSequentialBatchdict(
            PostActBlock(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, indice_key='spconv1'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            PostActBlock(channels[1], channels[2], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, \
                padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', pred_mode="pruning",\
                conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],loss_mode="focal_sprs",\
                point_cloud_range=self.point_cloud_range),
            # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.backbone_channels = {
            'x_conv1': channels[0],
            # 'x_conv2': channels[1],
            # 'x_conv3': channels[2],
            # 'x_conv4': channels[3]
        }
        self.forward_ret_dict = {}
        self.backbone_model = None

    # def bound_backbone(self,backbone):
    #     self.backbone_model = backbone

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        batch_dict['loss_box_of_pts_sprs']=0 if 'loss_box_of_pts_sprs' not in batch_dict else batch_dict['loss_box_of_pts_sprs']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor) # 41X1440X1440x16 33954 
        # x_conv0, batch_dict = self.conv0(x, batch_dict) # 41X1440X1440x16 33954
        x_conv1, batch_dict = self.conv1(x, batch_dict) # 41X1440X1440x16 33954 
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)
        pruning_ratio = self.downsample_pruning_ratio[0]
        voxel_importance = batch_dict['voxel_importance']

        # return x_conv2, batch_dict
        # if self.backbone_model is None:
        #     x = self.conv_input(input_sp_tensor) # 41X1440X1440x16 33954 
        #     x_conv1, batch_dict = self.conv1(x, batch_dict) # 41X1440X1440x16 33954 
        #     pruning_ratio = self.downsample_pruning_ratio[0]
        # else:
        #     x = self.backbone_model.conv_input(input_sp_tensor) # 41X1440X1440x16 33954
        #     x_conv1, batch_dict = self.backbone_model.conv1(x, batch_dict) # 41X1440X1440x16 33954
        #     pruning_ratio = self.backbone_model.downsample_pruning_ratio[0]
        # x_features = x_conv1.features
        # x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
        # sigmoid = nn.Sigmoid()
        # voxel_importance = sigmoid(x_attn_predict.view(-1, 1))
        _, indices = voxel_importance.view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):] #  比例越高越压缩
        # indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
        important_coords = x_conv1.indices[indices_im]#[:,1:] keep batch index
        # unimportant_coords = x_conv1.indices[indices_nim]#[:,1:]
        important_voxels = batch_dict['voxels'][indices_im]
        # unimportant_voxels = batch_dict['voxels'][indices_nim]
        important_voxel_num_points = batch_dict['voxel_num_points'][indices_im]

        # update batch_dict
        batch_dict['voxels'] = important_voxels
        batch_dict['voxel_coords'] = important_coords
        batch_dict['voxel_num_points'] = important_voxel_num_points

        # visulization
        vis_importance = False
        if vis_importance:
            all_coords = x_conv1.indices[:,1:]
            important_voxels = important_voxels.cpu().numpy()
            all_coords = all_coords.cpu().numpy()
            important_voxels.tofile('opencood/visualization/important_coords.bin')#,important_coords)
            # all_coords.tofile('opencood/visualization/all_coords.bin')

        if 'loss_box_of_pts_sprs_pruning' in batch_dict:
            batch_dict.update({'loss_box_of_pts_sprs_pruning':batch_dict['loss_box_of_pts_sprs'] + batch_dict['loss_box_of_pts_sprs_pruning']})
        else:
            batch_dict.update({'loss_box_of_pts_sprs_pruning':batch_dict['loss_box_of_pts_sprs']})
       
        # return important_coords, important_voxels, batch_dict
        return batch_dict
