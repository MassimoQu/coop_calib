# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
import warnings
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.transformation_utils import pose_to_tfm
from opencood.visualization import vis_utils, my_vis, simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\\..*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered in intersection.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*nn\\.functional\\.sigmoid is deprecated.*", category=UserWarning)

def _parse_float_list(raw: str):
    values = []
    for token in (raw or "").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _parse_hw(raw: str):
    raw = str(raw or "").strip()
    if not raw:
        return None
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return None
    try:
        side = int(raw)
    except Exception:
        return None
    return side, side


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument(
        "--pos-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated translation noise std list (meters).",
    )
    parser.add_argument(
        "--rot-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated yaw noise std list (degrees).",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["paired", "grid"],
        default="paired",
        help="paired: zip(pos,rot); grid: Cartesian product.",
    )
    parser.add_argument(
        "--noise-target",
        choices=["all", "ego", "non-ego"],
        default="non-ego",
        help="Which agent(s) receive synthetic pose noise. "
             "'all' matches legacy behavior; 'non-ego' makes relative error scale match the requested std.",
    )
    parser.add_argument(
        "--pose-correction",
        choices=["none", "v2xregpp_initfree", "v2xregpp_stable"],
        default="none",
        help="Optional extrinsic correction inside the dataset, before building pairwise transforms.",
    )
    parser.add_argument(
        "--stage1-result",
        type=str,
        default="",
        help="Stage1/detection cache JSON used by the pose corrector (required for v2xregpp_*).",
    )
    parser.add_argument(
        "--v2xregpp-config",
        type=str,
        default="configs/pipeline_midfusion_detection_occ.yaml",
        help="V2X-Reg++ pipeline config used by the pose corrector.",
    )
    parser.add_argument(
        "--v2xregpp-stage1-field",
        type=str,
        default="pred_corner3d_np_list",
        help="Stage1 result field name that contains per-agent 3D boxes (e.g., pred_corner3d_np_list, feature_corner3d_np_list).",
    )
    parser.add_argument(
        "--v2xregpp-bbox-type",
        type=str,
        default="detected",
        help="BBox type tag used inside V2X-Reg++ matching thresholds (e.g., detected, feature).",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-hint",
        action="store_true",
        help="Enable occ-hint seed when stage1_result contains occ_map_level0/paths.",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-pose",
        action="store_true",
        help="Treat occ-hint as a first-class pose candidate (and allow occ-only pose updates when boxes are sparse).",
    )
    parser.add_argument(
        "--v2xregpp-force-occ-pose",
        action="store_true",
        help="Always use occ-based pose (optionally refined), making the output independent of injected pose noise.",
    )
    parser.add_argument(
        "--v2xregpp-occ-from-lidar",
        action="store_true",
        help="Build BEV occupancy maps from raw lidar points (ignores stage1 occ_map fields).",
    )
    parser.add_argument(
        "--v2xregpp-occ-grid",
        type=str,
        default="256",
        help="Occupancy grid size for --v2xregpp-occ-from-lidar (int or 'H,W').",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-xy",
        type=float,
        default=20.0,
        help="Reject occ-hint if it differs from current pose by more than this translation (meters).",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-yaw",
        type=float,
        default=45.0,
        help="Reject occ-hint if it differs from current pose by more than this yaw (degrees).",
    )
    parser.add_argument(
        "--v2xregpp-icp-refine",
        action="store_true",
        help="Run Open3D ICP refinement on raw lidar points using the estimated transform as init.",
    )
    parser.add_argument("--v2xregpp-icp-voxel", type=float, default=1.0, help="ICP voxel downsample size (meters).")
    parser.add_argument("--v2xregpp-icp-max-corr", type=float, default=2.0, help="ICP max correspondence distance (meters).")
    parser.add_argument("--v2xregpp-icp-max-iter", type=int, default=30, help="ICP max iterations.")
    parser.add_argument("--v2xregpp-min-matches", type=int, default=3)
    parser.add_argument("--v2xregpp-min-stability", type=float, default=0.0)
    parser.add_argument("--v2xregpp-ema-alpha", type=float, default=0.5)
    parser.add_argument("--v2xregpp-max-step-xy", type=float, default=3.0)
    parser.add_argument("--v2xregpp-max-step-yaw", type=float, default=10.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. For v2xregpp_stable, keep this at 0 to preserve temporal state.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples per noise setting (debug only).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N samples (set 1 to print every sample).",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or 'V2XSET' in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    pos_std_list = _parse_float_list(opt.pos_std_list)
    rot_std_list = _parse_float_list(opt.rot_std_list)
    if not pos_std_list:
        pos_std_list = [0.0]
    if not rot_std_list:
        rot_std_list = [0.0]
    sweep_mode = opt.sweep_mode
    if sweep_mode == "paired":
        if len(pos_std_list) == 1 and len(rot_std_list) > 1:
            pos_std_list = pos_std_list * len(rot_std_list)
        if len(rot_std_list) == 1 and len(pos_std_list) > 1:
            rot_std_list = rot_std_list * len(pos_std_list)
        if len(pos_std_list) != len(rot_std_list):
            raise ValueError("paired sweep requires pos-std-list and rot-std-list to have equal length (or one of them length=1).")
        noise_pairs = list(zip(pos_std_list, rot_std_list))
    else:
        noise_pairs = [(p, r) for p in pos_std_list for r in rot_std_list]

    # Optional pose correction config injection.
    pose_correction = opt.pose_correction
    if pose_correction != "none":
        if not opt.stage1_result:
            raise ValueError("--stage1-result is required when --pose-correction != none")
        hypes.pop("box_align", None)
        mode = "stable" if pose_correction.endswith("stable") else "initfree"
        hypes["v2xregpp_align"] = {
            "train_result": opt.stage1_result,
            "val_result": opt.stage1_result,
            "args": {
                "config_path": opt.v2xregpp_config,
                "mode": mode,
                "use_occ_hint": bool(opt.v2xregpp_use_occ_hint),
                "use_occ_pose": bool(opt.v2xregpp_use_occ_pose),
                "force_occ_pose": bool(opt.v2xregpp_force_occ_pose),
                "occ_from_lidar": bool(opt.v2xregpp_occ_from_lidar),
                "occ_grid_hw": _parse_hw(opt.v2xregpp_occ_grid) or (256, 256),
                "occ_max_delta_xy_m": float(opt.v2xregpp_occ_max_delta_xy),
                "occ_max_delta_yaw_deg": float(opt.v2xregpp_occ_max_delta_yaw),
                "icp_refine": bool(opt.v2xregpp_icp_refine),
                "icp_voxel_size_m": float(opt.v2xregpp_icp_voxel),
                "icp_max_corr_dist_m": float(opt.v2xregpp_icp_max_corr),
                "icp_max_iterations": int(opt.v2xregpp_icp_max_iter),
                "min_matches": int(opt.v2xregpp_min_matches),
                "min_stability": float(opt.v2xregpp_min_stability),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
            },
        }

    
    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        rel_error_stats_all = []
        mean_pos = 0.0
        mean_rot = 0.0
        # Build the dataset once and only mutate the noise_setting per sweep entry.
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_setting['add_noise'] = True
        noise_setting['args'] = {
            'pos_std': 0.0,
            'rot_std': 0.0,
            'pos_mean': mean_pos,
            'rot_mean': mean_rot,
            'target': opt.noise_target,
        }
        if use_laplace:
            noise_setting['args']['laplace'] = True
        hypes.update({"noise_setting": noise_setting})
        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        num_workers = opt.num_workers
        if num_workers is None:
            num_workers = 0 if pose_correction.endswith("stable") else 4

        for pos_std, rot_std in noise_pairs:
            # setting noise
            noise_setting = OrderedDict()
            noise_args = {
                'pos_std': pos_std,
                'rot_std': rot_std,
                'pos_mean': mean_pos,
                'rot_mean': mean_rot,
                'target': opt.noise_target,
            }

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"

            opencood_dataset.params['noise_setting'] = noise_setting
            data_loader = DataLoader(
                opencood_dataset,
                batch_size=1,
                num_workers=int(num_workers),
                collate_fn=opencood_dataset.collate_batch_test,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )
            print(f"Noise Added: {pos_std}/{rot_std}/{mean_pos}/{mean_rot}.")
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            
            noise_level = f"{pos_std}_{rot_std}_{mean_pos}_{mean_rot}_" + opt.fusion_method + suffix + opt.note
            if pose_correction != "none":
                noise_level = f"{noise_level}_{pose_correction}"

            rel_trans_errors = []
            rel_yaw_errors = []


            for i, batch_data in enumerate(data_loader):
                if int(opt.log_interval or 0) > 0 and (i % int(opt.log_interval) == 0):
                    print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    
                    if opt.fusion_method == 'late':
                        infer_result = inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        infer_result = inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'single':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset,
                                                                        single_gt=True)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                'fusion is supported.')

                    pred_box_tensor = infer_result['pred_box_tensor']
                    gt_box_tensor = infer_result['gt_box_tensor']
                    pred_score = infer_result['pred_score']

                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)

                    try:
                        poses = batch_data['ego']['lidar_pose'].detach().cpu().numpy().reshape(-1, 6)
                        poses_clean = batch_data['ego']['lidar_pose_clean'].detach().cpu().numpy().reshape(-1, 6)
                        T_world = pose_to_tfm(poses)
                        T_world_clean = pose_to_tfm(poses_clean)
                        ego_T_world = T_world[0]
                        ego_T_world_clean = T_world_clean[0]
                        for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
                            rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
                            rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
                            err = np.linalg.inv(rel_clean) @ rel
                            rel_trans_errors.append(float(np.linalg.norm(err[:2, 3])))
                            yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
                            rel_yaw_errors.append(abs(yaw))
                    except Exception:
                        pass


                    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None) and (use_laplace is False):
                        vis_save_path_root = os.path.join(opt.model_dir, f'vis_{noise_level}')
                        if not os.path.exists(vis_save_path_root):
                            os.makedirs(vis_save_path_root)

                        """ If you want to 3d vis, uncomment lines below """
                        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                        # simple_vis.visualize(infer_result,
                        #                     batch_data['ego'][
                        #                         'origin_lidar'][0],
                        #                     hypes['postprocess']['gt_range'],
                        #                     vis_save_path,
                        #                     method='3d',
                        #                     left_hand=left_hand)
                        
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand)

                torch.cuda.empty_cache()
                if opt.max_eval_samples is not None and (i + 1) >= int(opt.max_eval_samples):
                    break

            ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                        opt.model_dir, noise_level)
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)

            rel_stats = {}
            if rel_trans_errors:
                arr = np.asarray(rel_trans_errors, dtype=np.float64)
                rel_stats['rel_trans_m'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
                thresholds_m = [1.0, 2.0, 3.0, 5.0, 10.0]
                rel_stats["rel_success_at_m"] = {
                    (str(int(thr)) if float(thr).is_integer() else str(thr)): float(np.mean(arr < float(thr)))
                    for thr in thresholds_m
                }
            if rel_yaw_errors:
                arr = np.asarray(rel_yaw_errors, dtype=np.float64)
                rel_stats['rel_yaw_deg'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
            rel_error_stats_all.append(
                {
                    'pos_std': float(pos_std),
                    'rot_std': float(rot_std),
                    **rel_stats,
                }
            )

            dump_dict = {
                'pos_std_list': [float(p) for p, _ in noise_pairs],
                'rot_std_list': [float(r) for _, r in noise_pairs],
                'noise_target': str(opt.noise_target or "all"),
                'ap30': AP30,
                'ap50': AP50,
                'ap70': AP70,
                'pose_correction': pose_correction,
                'rel_error_stats': rel_error_stats_all,
            }
            tag_safe = str(opt.note or "").replace("/", "_").replace(" ", "")
            corr_safe = str(pose_correction or "none")
            yaml_utils.save_yaml(
                dump_dict,
                os.path.join(opt.model_dir, f'AP030507_{corr_safe}{suffix}{tag_safe}.yaml'),
            )


if __name__ == '__main__':
    main()
