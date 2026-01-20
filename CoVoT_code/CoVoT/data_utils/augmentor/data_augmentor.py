# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

import numpy as np

from opencood.data_utils.augmentor import augment_utils


class DataAugmentor(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue = []
        self.train = train
        self.config = augment_config

        for cur_cfg in self.config:
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        extra_points = None
        if 'lidar_col_list' in data_dict:
            if len(data_dict['lidar_col_list']) > 0:
                extra_points = data_dict['lidar_col_list']

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points, extra_points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points, extra_points=extra_points
            )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points
        if extra_points is not None:
            data_dict['lidar_col_list'] = extra_points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        extra_points = None
        if 'lidar_col_list' in data_dict:
            if len(data_dict['lidar_col_list']) > 0:
                extra_points = data_dict['lidar_col_list']
            else:
                extra_points = None

        gt_boxes_valid, points, extra_points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range=rot_range, extra_points=extra_points
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points
        if extra_points is not None:
            data_dict['lidar_col_list'] = extra_points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        extra_points = None
        if 'lidar_col_list' in data_dict:
            if len(data_dict['lidar_col_list']) > 0:
                extra_points = data_dict['lidar_col_list']
            else:
                extra_points = None

        gt_boxes_valid, points, extra_points = augment_utils.global_scaling(
            gt_boxes_valid, points, config['WORLD_SCALE_RANGE'], extra_points=extra_points
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points
        if extra_points is not None:
            data_dict['lidar_col_list'] = extra_points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
    
    def get_augment_transform(self):
        """
        Returns:
        the random augmentor transform for each augmentation
        """
        if not self.train:
            return None
        augmentor_trans = {}
        if "random_world_flip" in self.data_augmentor_queue:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        for cur_cfg in self.config:
            if cur_cfg['NAME'] == 'random_world_flip':
                enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                augmentor_trans.update({'random_world_flip': enable})
            if cur_cfg['NAME'] == 'random_world_rotation':
                rot_range = cur_cfg['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
                augmentor_trans.update({'random_world_rotation': noise_rotation})
            if cur_cfg['NAME'] == 'random_world_scaling':
                scale_range = cur_cfg['WORLD_SCALE_RANGE']
                if not isinstance(scale_range, list):
                    scale_range = [-scale_range, scale_range]
                noise_scale = np.random.uniform(scale_range[0], scale_range[1])
                augmentor_trans.update({'random_world_scaling': noise_scale})

        return augmentor_trans

    def forward_hybrid(self, data_dict, augmentor_trans):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict