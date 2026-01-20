# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
from opencood.data_utils.datasets.early_fusion_dataset_voxelnext import EarlyFusionDatasetVoxelNeXt
from opencood.data_utils.datasets.hybrid_fusion_dataset import HybridFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.hybrid_fusion_dataset_dair import HybridFusionDatasetDAIR
__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'EarlyFusionDatasetVoxelNeXt': EarlyFusionDatasetVoxelNeXt,
    'HybridFusionDataset': HybridFusionDataset,
    'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    'HybridFusionDatasetDAIR': HybridFusionDatasetDAIR
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
GT_RANGE_DAIR = [0, -46.08, -3, 92.16, 46.08, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    # assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
    #                         'IntermediateFusionDataset', 'IntermediateFusionDatasetV2'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
