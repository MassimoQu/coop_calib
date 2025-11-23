# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import copy
from collections import OrderedDict
import json

import numpy as np
import math


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        default="/root/OpenCOODv2/opencood/hypes_yaml/v2xset/lidar_only/coalign/precalc.yaml",
                        help='data generation yaml file needed ')
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Optional override for stage1 model checkpoint when box_align_pre_calc is missing.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional override for stage1 box export directory.")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma separated dataset splits to export (default: train,val,test).")
    parser.add_argument("--comm_range_override", type=float, default=None,
                        help="Override comm_range to force exporting all agents (useful for detection cache generation).")
    parser.add_argument("--force_ego_cav", type=str, default=None,
                        help="Force dataset to treat either 'vehicle' or 'infrastructure' as ego.")
    parser.add_argument("--per_agent", action='store_true',
                        help="Enable dual single-agent export (vehicle + infrastructure).")
    parser.add_argument("--vehicle_hypes", type=str, default=None,
                        help="Hypes yaml for vehicle-side single-agent detector.")
    parser.add_argument("--vehicle_checkpoint", type=str, default=None,
                        help="Checkpoint path for vehicle-side detector.")
    parser.add_argument("--vehicle_output", type=str, default=None,
                        help="Output directory for vehicle detector boxes.")
    parser.add_argument("--vehicle_force_ego", type=str, default="vehicle",
                        help="Which CAV should be treated as ego for vehicle export (default: vehicle).")
    parser.add_argument("--infra_hypes", type=str, default=None,
                        help="Hypes yaml for infrastructure-side detector.")
    parser.add_argument("--infra_checkpoint", type=str, default=None,
                        help="Checkpoint path for infrastructure-side detector.")
    parser.add_argument("--infra_output", type=str, default=None,
                        help="Output directory for infrastructure detector boxes.")
    parser.add_argument("--infra_force_ego", type=str, default="infrastructure",
                        help="Which CAV should be treated as ego for infrastructure export (default: infrastructure).")
    parser.add_argument("--merged_output", type=str, default=None,
                        help="Destination directory for merged vehicle/infra stage1 boxes.")
    parser.add_argument("--single_agent_comm_range", type=float, default=0.0,
                        help="Communication range override for single-agent exports (default: 0).")
    parser.add_argument("--feature_topk", type=int, default=0,
                        help="Top-K BEV anchors per agent to export as feature boxes (0 disables feature export).")
    parser.add_argument("--feature_min_score", type=float, default=0.2,
                        help="Minimum sigmoid score for a feature anchor to be kept.")
    parser.add_argument("--feature_box_dims", type=float, nargs=3, default=(0.8, 0.8, 0.5),
                        metavar=('L', 'W', 'H'),
                        help="Micro-box dimensions (meters) for feature objects (default: 0.8 0.8 0.5).")
    parser.add_argument("--dump_bev_features", action='store_true',
                        help="Enable dense BEV feature peak export from PyramidFusion.")
    parser.add_argument("--bev_feature_topk", type=int, default=0,
                        help="Number of BEV feature peaks per agent to export when dump_bev_features is on.")
    parser.add_argument("--bev_feature_level", type=int, default=0,
                        help="Pyramid level index (0=highest resolution) used for peak extraction.")
    parser.add_argument("--bev_feature_score_min", type=float, default=0.3,
                        help="Minimum sigmoid score for BEV feature peaks.")
    parser.add_argument("--bev_descriptor_dim", type=int, default=32,
                        help="(deprecated) kept for backward compatibility; shape descriptors ignore this.")
    parser.add_argument("--bev_descriptor_norm", action='store_true',
                        help="(deprecated) kept for backward compatibility.")
    parser.add_argument("--bev_feature_box_dims", type=float, nargs=3, default=(0.6, 0.6, 0.5),
                        metavar=('L', 'W', 'H'),
                        help="Micro-box dimensions for BEV feature peaks.")
    parser.add_argument("--bev_feature_min_separation", type=int, default=0,
                        help="Minimum grid distance between BEV peaks (applied in row/col units).")
    parser.add_argument("--bev_descriptor_neighbors", type=int, default=6,
                        help="Number of nearest-neighbor distances to encode as descriptor.")
    parser.add_argument("--bev_descriptor_on_detections", action='store_true',
                        help="Attach BEV descriptors to detection boxes in pred_corner3d_np_list.")
    parser.add_argument("--max_export_samples", type=int, default=None,
                        help="Optional limit on number of samples to export per split (useful for quick tests).")
    opt = parser.parse_args()
    return opt

SAVE_BOXES = True


def _parse_splits(splits_str):
    splits = [s.strip() for s in splits_str.split(',')] if splits_str else []
    splits = [s for s in splits if s]
    return splits or ['train', 'val', 'test']


def _micro_box_from_center(center, dims, yaw_rad):
    l, w, h = dims
    dx = l / 2.0
    dy = w / 2.0
    dz = h / 2.0
    corners = np.array([
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=np.float32)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    rot = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y,  cos_y, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float32)
    rotated = corners @ rot.T
    translated = rotated + np.asarray(center, dtype=np.float32)
    return translated.tolist()


def _extract_feature_boxes(cls_preds, anchor_tensor, topk, min_score, box_dims):
    if topk <= 0 or cls_preds is None:
        return []
    anchor_tensor = anchor_tensor.to(cls_preds.device)
    probs = torch.sigmoid(cls_preds).permute(0, 2, 3, 1).contiguous()
    flat_anchor = anchor_tensor.view(-1, anchor_tensor.shape[-1])
    num_agents = probs.shape[0]
    results = []
    for agent_idx in range(num_agents):
        flat_scores = probs[agent_idx].reshape(-1)
        k = min(topk, flat_scores.shape[0])
        if k <= 0:
            results.append(([], []))
            continue
        values, indices = torch.topk(flat_scores, k=k)
        agent_boxes = []
        agent_scores = []
        for score, flat_idx in zip(values.tolist(), indices.tolist()):
            if score < min_score:
                break
            anchor_state = flat_anchor[flat_idx].cpu().numpy()
            center = anchor_state[:3]
            yaw = float(anchor_state[6])
            corners = _micro_box_from_center(center, box_dims, yaw)
            agent_boxes.append({
                'type': 'feature',
                'score': float(score),
                'corners': corners,
            })
            agent_scores.append(float(score))
        results.append((agent_boxes, agent_scores))
    return results


def _record_len_to_list(record_len):
    if record_len is None:
        return []
    if torch.is_tensor(record_len):
        values = record_len.detach().cpu().numpy().astype(int).tolist()
    else:
        values = [int(x) for x in record_len]
    return [v for v in values if v > 0]


def _chunk_entries(entries, record_len_list):
    chunks = []
    start = 0
    for count in record_len_list:
        chunks.append(entries[start:start + count])
        start += count
    return chunks


def _apply_neighbor_descriptor(entries, neighbor_k):
    for agent_idx, feats in enumerate(entries):
        if not feats:
            continue
        coords = []
        for feat in feats:
            center = feat.pop('center', None)
            if center is None:
                corners = feat.get('corners')
                if corners is not None:
                    arr = np.asarray(corners, dtype=np.float32)
                    center = arr.mean(axis=0)
            if center is None:
                coords.append([0.0, 0.0])
            else:
                coords.append([float(center[0]), float(center[1])])
        coords = np.asarray(coords, dtype=np.float32)
        if len(coords) < 2:
            for feat in feats:
                feat['descriptor'] = [0.0] * neighbor_k
            continue
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        for idx, feat in enumerate(feats):
            dists = np.sort(dist[idx][dist[idx] > 0])
            if dists.size < neighbor_k:
                pad = np.zeros(neighbor_k, dtype=np.float32)
                pad[:dists.size] = dists
                dists = pad
            else:
                dists = dists[:neighbor_k]
            feat['descriptor'] = dists.tolist()
    return entries


def _extract_bev_feature_peaks(runtime_store, cav_range, cfg):
    if not runtime_store or cfg.get('topk', 0) <= 0:
        return None
    bev_tensor = runtime_store.get('agent_bev')
    occ_map_list = runtime_store.get('occ_map_list')
    record_len = runtime_store.get('record_len')
    if bev_tensor is None or occ_map_list is None:
        return None
    record_len_list = _record_len_to_list(record_len)
    total_agents = bev_tensor.shape[0]
    if not record_len_list:
        record_len_list = [total_agents]
    level = int(cfg.get('level', 0))
    level = max(0, min(level, len(occ_map_list) - 1))
    occ_tensor = occ_map_list[level]
    if occ_tensor.shape[0] != total_agents:
        # fall back to broadcasting if pyramid level omits agents
        occ_tensor = occ_tensor[:total_agents]
    extent_x = cav_range[3] - cav_range[0]
    extent_y = cav_range[4] - cav_range[1]
    topk = int(cfg.get('topk', 0))
    min_score = float(cfg.get('score_min', 0.0))
    box_dims = tuple(cfg.get('box_dims', (0.6, 0.6, 0.5)))
    min_sep = int(cfg.get('min_separation', 0))
    neighbor_k = max(1, int(cfg.get('neighbor_k', 6)))
    per_agent_entries = []
    _, _, occ_H, occ_W = occ_tensor.shape
    _, _, feat_H, feat_W = bev_tensor.shape
    for agent_idx in range(total_agents):
        occ_map = occ_tensor[agent_idx, 0]
        scores = torch.sigmoid(occ_map).reshape(-1)
        k = min(topk, scores.shape[0])
        if k <= 0:
            per_agent_entries.append([])
            continue
        values, indices = torch.topk(scores, k=k)
        feat_map = bev_tensor[agent_idx]
        if (feat_H, feat_W) != (occ_H, occ_W):
            feat_map = F.interpolate(feat_map.unsqueeze(0), size=(occ_H, occ_W), mode='bilinear', align_corners=False).squeeze(0)
        x_res = extent_x / occ_W
        y_res = extent_y / occ_H
        agent_features = []
        selected_coords = []
        for value, index in zip(values.tolist(), indices.tolist()):
            if value < min_score:
                break
            row = index // occ_W
            col = index % occ_W
            if min_sep > 0:
                too_close = False
                for r_sel, c_sel in selected_coords:
                    if abs(row - r_sel) <= min_sep and abs(col - c_sel) <= min_sep:
                        too_close = True
                        break
                if too_close:
                    continue
            center_x = cav_range[0] + (col + 0.5) * x_res
            center_y = cav_range[1] + (row + 0.5) * y_res
            center = [float(center_x), float(center_y), 0.0]
            agent_features.append({
                'type': 'feature',
                'score': float(value),
                'corners': _micro_box_from_center(center, box_dims, 0.0),
                'center': center,
                'level': level,
                'grid': [int(row), int(col)],
            })
            selected_coords.append((row, col))
        per_agent_entries.append(agent_features)
    per_agent_entries = _apply_neighbor_descriptor(per_agent_entries, neighbor_k)
    return _chunk_entries(per_agent_entries, record_len_list)


def _annotate_detection_descriptors(runtime_store, cav_range, pred_corner3d_list, cfg):
    if not cfg.get('attach_detection'):
        return None
    bev_tensor = runtime_store.get('agent_bev')
    if bev_tensor is None:
        return None
    neighbor_k = max(1, int(cfg.get('neighbor_k', 6)))
    annotated = []
    for cav_idx, detections in enumerate(pred_corner3d_list):
        cav_entries = []
        if not isinstance(detections, list):
            annotated.append([])
            continue
        for det in detections:
            arr = np.asarray(det, dtype=np.float32)
            if arr.ndim >= 2:
                center = arr.mean(axis=0)
            else:
                center = arr
            cav_entries.append({
                'type': 'detected',
                'score': 1.0,
                'corners': arr.tolist(),
                'center': center.tolist(),
            })
        annotated.append(cav_entries)
    return _apply_neighbor_descriptor(annotated, neighbor_k)


def _ensure_box_align_config(hypes, stage1_checkpoint_path, output_dir):
    if not stage1_checkpoint_path:
        raise ValueError("stage1_checkpoint must be provided for export.")
    postprocessor_name = hypes['postprocess'].get('core_method', 'VoxelPostprocessor')
    if postprocessor_name.lower() == 'voxelpostprocessor':
        postprocessor_name = 'uncertainty_voxel_postprocessor'
    hypes['box_align_pre_calc'] = {
        'stage1_model': hypes['model']['core_method'],
        'stage1_model_config': copy.deepcopy(hypes['model']['args']),
        'stage1_model_path': stage1_checkpoint_path,
        'stage1_postprocessor_name': postprocessor_name,
        'stage1_postprocessor_config': copy.deepcopy(hypes['postprocess']),
        'output_save_path': output_dir,
    }


def _prepare_hypes(base_hypes, stage1_checkpoint_path, output_dir,
                   comm_range_override=None, force_ego_cav=None):
    hypes = copy.deepcopy(base_hypes)
    if comm_range_override is not None:
        hypes['comm_range'] = comm_range_override
    if force_ego_cav:
        hypes['force_ego_cav'] = force_ego_cav
    _ensure_box_align_config(hypes, stage1_checkpoint_path, output_dir)
    return hypes


def _load_state_dict_with_spconv_fix(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    for key, value in list(state_dict.items()):
        if isinstance(value, torch.Tensor) and value.ndim == 5:
            state_dict[key] = value.permute(3, 0, 1, 2, 4).contiguous()
    return state_dict


def export_stage1_boxes(prepared_hypes, splits, feature_cfg=None, max_samples=None, bev_feature_cfg=None):
    feature_cfg = feature_cfg or {}
    bev_feature_cfg = bev_feature_cfg or {}
    feature_topk = max(0, int(feature_cfg.get('topk', 0)))
    feature_min_score = float(feature_cfg.get('min_score', 0.0))
    feature_box_dims = tuple(feature_cfg.get('box_dims', (0.8, 0.8, 0.5)))
    bev_feature_enabled = bool(bev_feature_cfg.get('enabled', False))
    splits_list = _parse_splits(splits)
    hypes = yaml_utils.load_voxel_params(prepared_hypes)

    pos_std_list = [0]
    rot_std_list = [0]
    pos_mean_list = [0]
    rot_mean_list = [0]
    exported_paths = {}

    for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
        noise_setting = OrderedDict()
        noise_args = {'pos_std': pos_std,
                      'rot_std': rot_std,
                      'pos_mean': pos_mean,
                      'rot_mean': rot_mean}

        noise_setting['add_noise'] = True
        noise_setting['args'] = noise_args

        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

        print('Dataset Building')
        opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
        hypes_ = copy.deepcopy(hypes)
        hypes_['validate_dir'] = hypes_['test_dir']
        opencood_test_dataset = build_dataset(hypes_, visualize=False, train=False)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=1,
                                  num_workers=4,
                                  collate_fn=opencood_train_dataset.collate_batch_test,
                                  shuffle=False,
                                  pin_memory=False,
                                  drop_last=False)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

        test_loader = DataLoader(opencood_test_dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 collate_fn=opencood_train_dataset.collate_batch_test,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        stage1_model_name = hypes['box_align_pre_calc']['stage1_model']
        stage1_model_config = hypes['box_align_pre_calc']['stage1_model_config']
        stage1_checkpoint_path = hypes['box_align_pre_calc']['stage1_model_path']

        model_filename = "opencood.models." + stage1_model_name
        model_lib = importlib.import_module(model_filename)
        stage1_model_class = None
        target_model_name = stage1_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                stage1_model_class = cls

        stage1_model = stage1_model_class(stage1_model_config)
        runtime_dump_enabled = bev_feature_enabled or bev_feature_cfg.get('attach_detection', False)
        if hasattr(stage1_model, 'record_runtime_features'):
            stage1_model.record_runtime_features = runtime_dump_enabled
        state_dict = _load_state_dict_with_spconv_fix(stage1_checkpoint_path)
        stage1_model.load_state_dict(state_dict, strict=False)

        stage1_postprocessor_name = hypes['box_align_pre_calc']['stage1_postprocessor_name']
        stage1_postprocessor_config = hypes['box_align_pre_calc']['stage1_postprocessor_config']
        postprocessor_lib = importlib.import_module('opencood.data_utils.post_processor')
        stage1_postprocessor_class = None
        target_postprocessor_name = stage1_postprocessor_name.replace('_', '').lower()

        for name, cls in postprocessor_lib.__dict__.items():
            if name.lower() == target_postprocessor_name:
                stage1_postprocessor_class = cls

        stage1_postprocessor = stage1_postprocessor_class(stage1_postprocessor_config, train=False)
        cav_lidar_range = prepared_hypes['model']['args']['lidar_range']

        for p in stage1_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
            stage1_model.to(device)

        stage1_model.eval()
        stage1_anchor_box = torch.from_numpy(stage1_postprocessor.generate_anchor_box())

        loader_map = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }

        for split in splits_list:
            if split not in loader_map:
                continue

            stage1_boxes_dict = dict()
            stage1_boxes_save_dir = f"{hypes['box_align_pre_calc']['output_save_path']}/{split}"
            if not os.path.exists(stage1_boxes_save_dir):
                os.makedirs(stage1_boxes_save_dir)
            stage1_boxes_save_path = os.path.join(stage1_boxes_save_dir, "stage1_boxes.json")

            loader = loader_map[split]
            processed_split = 0
            for i, batch_data in enumerate(loader):
                if max_samples is not None and processed_split >= max_samples:
                    break
                if batch_data is None:
                    continue

                batch_data = train_utils.to_device(batch_data, device)
                print(i, batch_data['ego']['sample_idx'], batch_data['ego']['cav_id_list'])
                output_stage1 = stage1_model(batch_data['ego'])
                if isinstance(output_stage1, dict) and 'unc_preds' not in output_stage1:
                    cls_preds_tensor = output_stage1.get('cls_preds')
                    if cls_preds_tensor is not None:
                        B, A, H, W = cls_preds_tensor.shape
                        unc_channels = A * 2
                        output_stage1['unc_preds'] = cls_preds_tensor.new_zeros(
                            (B, unc_channels, H, W), dtype=cls_preds_tensor.dtype, device=cls_preds_tensor.device
                        )
                pred_corner3d_list, pred_box3d_list, uncertainty_list = \
                    stage1_postprocessor.post_process_stage1(output_stage1, stage1_anchor_box)

                if pred_corner3d_list is None:
                    continue
                runtime_store = getattr(stage1_model, '_runtime_feature_store', {})
                feature_entries = None
                if feature_topk > 0 and isinstance(output_stage1, dict) and 'cls_preds' in output_stage1:
                    feature_entries = _extract_feature_boxes(
                        output_stage1['cls_preds'].detach(),
                        stage1_anchor_box.float(),
                        feature_topk,
                        feature_min_score,
                        feature_box_dims,
                    )
                bev_feature_entries = None
                if bev_feature_enabled:
                    bev_chunks = _extract_bev_feature_peaks(runtime_store, cav_lidar_range, bev_feature_cfg)
                    if bev_chunks:
                        bev_feature_entries = bev_chunks[0]

                if SAVE_BOXES:
                    sample_idx = batch_data['ego']['sample_idx']
                    pred_corner3d_np_list = [x.cpu().numpy().tolist() for x in pred_corner3d_list]
                    uncertainty_np_list = [x.cpu().numpy().tolist() for x in uncertainty_list]
                    lidar_pose_clean_np = batch_data['ego']['lidar_pose_clean'].cpu().numpy().tolist()
                    stage1_boxes_dict[sample_idx] = OrderedDict()

                    detection_with_descriptor = None
                    if runtime_store and bev_feature_cfg.get('attach_detection'):
                        detection_with_descriptor = _annotate_detection_descriptors(
                            runtime_store,
                            cav_lidar_range,
                            pred_corner3d_np_list,
                            bev_feature_cfg,
                        )
                    detections_to_store = detection_with_descriptor or pred_corner3d_np_list

                    stage1_boxes_dict[sample_idx]['pred_corner3d_np_list'] = detections_to_store
                    stage1_boxes_dict[sample_idx]['uncertainty_np_list'] = uncertainty_np_list
                    stage1_boxes_dict[sample_idx]['lidar_pose_clean_np'] = lidar_pose_clean_np
                    stage1_boxes_dict[sample_idx]['cav_id_list'] = batch_data['ego']['cav_id_list']
                    stage1_boxes_dict[sample_idx]['bev_range'] = cav_lidar_range
                    combined_features = None
                    if feature_entries:
                        combined_features = [list(boxes) for boxes, _ in feature_entries]
                    if bev_feature_entries:
                        if combined_features is None:
                            combined_features = [[] for _ in range(len(bev_feature_entries))]
                        if len(combined_features) < len(bev_feature_entries):
                            combined_features.extend(
                                [[] for _ in range(len(bev_feature_entries) - len(combined_features))]
                            )
                        for cav_idx, bev_boxes in enumerate(bev_feature_entries):
                            combined_features[cav_idx].extend(bev_boxes)
                    if combined_features:
                        stage1_boxes_dict[sample_idx]['feature_corner3d_np_list'] = combined_features
                    if runtime_store and runtime_store.get('occ_map_list'):
                        occ_tensor_list = runtime_store['occ_map_list']
                        if occ_tensor_list:
                            occ_level0 = occ_tensor_list[0].detach().cpu().numpy().tolist()
                            stage1_boxes_dict[sample_idx]['occ_map_level0'] = occ_level0
                    processed_split += 1

            if SAVE_BOXES:
                with open(stage1_boxes_save_path, "w") as f:
                    json.dump(stage1_boxes_dict, f, indent=4, sort_keys=True)
                exported_paths[split] = stage1_boxes_save_path
    return exported_paths


def _load_stage1_json(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'stage1_boxes.json')
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Stage1 export must be a dict, got {type(data)}")
    return {str(k): v for k, v in data.items()}, path


def _first_list(entry, key):
    value = entry.get(key)
    if isinstance(value, list) and value:
        return value[0]
    return []


def merge_stage1_outputs(infra_stage1_path, vehicle_stage1_path, merged_output_dir):
    infra_dict, infra_stage1_path = _load_stage1_json(infra_stage1_path)
    veh_dict, vehicle_stage1_path = _load_stage1_json(vehicle_stage1_path)
    keys = sorted(set(infra_dict.keys()) & set(veh_dict.keys()), key=lambda x: int(x))
    merged = OrderedDict()
    for key in keys:
        infra_entry = infra_dict[key]
        veh_entry = veh_dict[key]
        if infra_entry is None or veh_entry is None:
            continue
        record = OrderedDict()
        record['cav_id_list'] = [
            _first_list(infra_entry, 'cav_id_list'),
            _first_list(veh_entry, 'cav_id_list'),
        ]
        record['pred_corner3d_np_list'] = [
            _first_list(infra_entry, 'pred_corner3d_np_list'),
            _first_list(veh_entry, 'pred_corner3d_np_list'),
        ]
        record['uncertainty_np_list'] = [
            _first_list(infra_entry, 'uncertainty_np_list'),
            _first_list(veh_entry, 'uncertainty_np_list'),
        ]
        if 'occ_map_level0' in infra_entry or 'occ_map_level0' in veh_entry:
            record['occ_map_level0'] = [
                infra_entry.get('occ_map_level0'),
                veh_entry.get('occ_map_level0'),
            ]
        if 'feature_corner3d_np_list' in infra_entry or 'feature_corner3d_np_list' in veh_entry:
            record['feature_corner3d_np_list'] = [
                _first_list(infra_entry, 'feature_corner3d_np_list'),
                _first_list(veh_entry, 'feature_corner3d_np_list'),
            ]
        for pose_key in ('lidar_pose_clean_np', 'lidar_pose_np'):
            record[pose_key] = [
                _first_list(infra_entry, pose_key),
                _first_list(veh_entry, pose_key),
            ]
        if 'bev_range' in infra_entry:
            record['bev_range'] = infra_entry['bev_range']
        elif 'bev_range' in veh_entry:
            record['bev_range'] = veh_entry['bev_range']
        merged[key] = record
    if merged_output_dir is None:
        merged_output_dir = os.path.join(os.path.dirname(vehicle_stage1_path), "merged_stage1")
    if os.path.isdir(merged_output_dir) or not merged_output_dir.endswith('.json'):
        os.makedirs(merged_output_dir, exist_ok=True)
        output_path = os.path.join(merged_output_dir, "stage1_boxes.json")
    else:
        os.makedirs(os.path.dirname(merged_output_dir), exist_ok=True)
        output_path = merged_output_dir
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote merged stage1 boxes for {len(merged)} samples to {output_path}")
    return output_path


def run_per_agent_mode(opt, feature_cfg, bev_feature_cfg):
    split_names = _parse_splits(opt.splits)
    target_split = 'test' if 'test' in split_names else split_names[0]

    specs = [
        ('vehicle', opt.vehicle_hypes or opt.hypes_yaml, opt.vehicle_checkpoint or opt.stage1_checkpoint,
         opt.vehicle_output, opt.vehicle_force_ego or 'vehicle'),
        ('infrastructure', opt.infra_hypes, opt.infra_checkpoint,
         opt.infra_output, opt.infra_force_ego or 'infrastructure'),
    ]

    exported_paths = {}
    for name, hypes_path, ckpt_path, output_dir, force_ego in specs:
        if hypes_path is None or ckpt_path is None:
            raise ValueError(f"{name} export requires both hypes and checkpoint paths.")
        base_hypes = yaml_utils.load_yaml(hypes_path, Namespace(model_dir=""))
        default_output_dir = output_dir or os.path.join(
            os.path.dirname(ckpt_path), f"stage1_{name}_export")
        prepared_hypes = _prepare_hypes(
            base_hypes,
            ckpt_path,
            default_output_dir,
            comm_range_override=opt.single_agent_comm_range,
            force_ego_cav=force_ego)
        exported_paths[name] = export_stage1_boxes(
            prepared_hypes,
            opt.splits,
            feature_cfg=feature_cfg,
            max_samples=opt.max_export_samples,
            bev_feature_cfg=bev_feature_cfg,
        )

    veh_stage1_path = exported_paths['vehicle'].get(target_split)
    infra_stage1_path = exported_paths['infrastructure'].get(target_split)
    if veh_stage1_path is None or infra_stage1_path is None:
        raise RuntimeError(f"Missing exported stage1 boxes for split '{target_split}'.")
    merge_stage1_outputs(infra_stage1_path, veh_stage1_path, opt.merged_output)


def main():
    opt = train_parser()
    feature_cfg = {
        'topk': max(0, int(opt.feature_topk)),
        'min_score': float(opt.feature_min_score),
        'box_dims': tuple(opt.feature_box_dims),
    }
    bev_feature_cfg = {
        'enabled': bool(opt.dump_bev_features),
        'topk': max(0, int(opt.bev_feature_topk)),
        'level': max(0, int(opt.bev_feature_level)),
        'score_min': float(opt.bev_feature_score_min),
        'descriptor_dim': max(0, int(opt.bev_descriptor_dim)),
        'descriptor_norm': bool(opt.bev_descriptor_norm),
        'box_dims': tuple(opt.bev_feature_box_dims),
        'min_separation': max(0, int(opt.bev_feature_min_separation)),
        'attach_detection': bool(opt.bev_descriptor_on_detections),
        'neighbor_k': max(1, int(opt.bev_descriptor_neighbors)),
    }

    if opt.per_agent:
        run_per_agent_mode(opt, feature_cfg, bev_feature_cfg)
        return

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    checkpoint_path = opt.stage1_checkpoint
    if 'box_align_pre_calc' in hypes and not checkpoint_path:
        checkpoint_path = hypes['box_align_pre_calc'].get('stage1_model_path')
    if checkpoint_path is None:
        raise ValueError("stage1_checkpoint must be provided.")

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), 'stage1_boxes_export')

    prepared_hypes = _prepare_hypes(
        hypes,
        checkpoint_path,
        output_dir,
        comm_range_override=opt.comm_range_override,
        force_ego_cav=opt.force_ego_cav)
    export_stage1_boxes(
        prepared_hypes,
        opt.splits,
        feature_cfg=feature_cfg,
        max_samples=opt.max_export_samples,
        bev_feature_cfg=bev_feature_cfg,
    )


if __name__ == '__main__':
    main()
