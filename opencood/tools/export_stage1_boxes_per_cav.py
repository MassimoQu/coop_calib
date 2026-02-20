#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Codex (based on pose_graph_pre_calc.py)
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import copy
import importlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.pose_utils import attach_pose_confidence
from opencood.utils.transformation_utils import get_pairwise_transformation


def _load_state_dict_with_spconv_fix(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    for key, value in list(state_dict.items()):
        if isinstance(value, torch.Tensor) and value.ndim == 5:
            if value.shape[0] <= 4:
                state_dict[key] = value.permute(1, 0, 2, 3, 4).contiguous()
    return state_dict


def _ensure_box_align_config(hypes, stage1_checkpoint_path, output_dir):
    if not stage1_checkpoint_path:
        raise ValueError("stage1_checkpoint must be provided for export.")
    postprocessor_name = hypes["postprocess"].get("core_method", "VoxelPostprocessor")
    if postprocessor_name.lower() == "voxelpostprocessor":
        postprocessor_name = "uncertainty_voxel_postprocessor"
    hypes["box_align_pre_calc"] = {
        "stage1_model": hypes["model"]["core_method"],
        "stage1_model_config": copy.deepcopy(hypes["model"]["args"]),
        "stage1_model_path": stage1_checkpoint_path,
        "stage1_postprocessor_name": postprocessor_name,
        "stage1_postprocessor_config": copy.deepcopy(hypes["postprocess"]),
        "output_save_path": output_dir,
    }


def _build_single_cav_sample(dataset, base_data_dict, cav_id, sample_idx):
    selected_cav_base = copy.deepcopy(base_data_dict[cav_id])
    selected_cav_base["ego"] = True
    params = selected_cav_base.get("params") or {}
    if "lidar_pose_clean" not in params:
        params["lidar_pose_clean"] = params.get("lidar_pose")
        selected_cav_base["params"] = params
    base_single = OrderedDict([(cav_id, selected_cav_base)])
    attach_pose_confidence(base_single)

    ego_cav_base = selected_cav_base
    selected_cav_processed = dataset.get_item_single_car(selected_cav_base, ego_cav_base)

    object_stack = [selected_cav_processed["object_bbx_center"]]
    object_id_stack = list(selected_cav_processed["object_ids"])
    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
    object_stack = np.vstack(object_stack)
    object_stack = object_stack[unique_indices]

    max_num = int(dataset.params["postprocess"]["max_num"])
    object_bbx_center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    object_bbx_center[: object_stack.shape[0], :] = object_stack
    mask[: object_stack.shape[0]] = 1

    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=dataset.anchor_box, mask=mask
    )

    lidar_pose_list = [selected_cav_base["params"]["lidar_pose"]]
    lidar_pose_clean_list = [selected_cav_base["params"]["lidar_pose_clean"]]
    pose_confidence_list = [
        base_single[cav_id]["params"].get("pose_confidence", 1.0)
    ]

    pairwise_t_matrix = get_pairwise_transformation(
        base_single, dataset.max_cav, dataset.proj_first
    )

    processed_data_dict = OrderedDict()
    processed_data_dict["ego"] = {}
    if hasattr(dataset, "modality_name_list"):
        for name in dataset.modality_name_list:
            processed_data_dict["ego"][f"input_{name}"] = None
        modality_name = selected_cav_base.get("modality_name")
        sensor_type = dataset.sensor_type_dict.get(modality_name)
        if sensor_type == "lidar":
            merged_feature_dict = merge_features_to_dict(
                [selected_cav_processed[f"processed_features_{modality_name}"]]
            )
            processed_data_dict["ego"][f"input_{modality_name}"] = merged_feature_dict
        elif sensor_type == "camera":
            merged_image_inputs_dict = merge_features_to_dict(
                [selected_cav_processed[f"image_inputs_{modality_name}"]], merge="stack"
            )
            processed_data_dict["ego"][f"input_{modality_name}"] = merged_image_inputs_dict
        processed_data_dict["ego"]["agent_modality_list"] = [modality_name]
    else:
        if dataset.load_lidar_file:
            merged_feature_dict = merge_features_to_dict(
                [selected_cav_processed["processed_features"]]
            )
            processed_data_dict["ego"].update({"processed_lidar": merged_feature_dict})
        if dataset.load_camera_file:
            merged_image_inputs_dict = merge_features_to_dict(
                [selected_cav_processed["image_inputs"]], merge="stack"
            )
            processed_data_dict["ego"].update({"image_inputs": merged_image_inputs_dict})

    processed_data_dict["ego"].update(
        {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [object_id_stack[i] for i in unique_indices],
            "anchor_box": dataset.anchor_box,
            "label_dict": label_dict,
            "cav_num": 1,
            "pairwise_t_matrix": pairwise_t_matrix,
            "lidar_poses_clean": np.array(lidar_pose_clean_list).reshape(-1, 6),
            "lidar_poses": np.array(lidar_pose_list).reshape(-1, 6),
            "pose_confidence": np.array(pose_confidence_list, dtype=np.float32).reshape(-1),
            "sample_idx": sample_idx,
            "cav_id_list": [cav_id],
        }
    )
    if getattr(dataset, "supervise_single", False) or getattr(dataset, "heterogeneous", False):
        single_label_dicts = dataset.post_processor.collate_batch(
            [selected_cav_processed["single_label_dict"]]
        )
        single_object_bbx_center = torch.from_numpy(
            np.array([selected_cav_processed["single_object_bbx_center"]])
        )
        single_object_bbx_mask = torch.from_numpy(
            np.array([selected_cav_processed["single_object_bbx_mask"]])
        )
        processed_data_dict["ego"].update(
            {
                "single_label_dict_torch": single_label_dicts,
                "single_object_bbx_center_torch": single_object_bbx_center,
                "single_object_bbx_mask_torch": single_object_bbx_mask,
            }
        )
    return processed_data_dict


def _build_stage1_model(hypes):
    stage1_model_name = hypes["box_align_pre_calc"]["stage1_model"]
    stage1_model_config = hypes["box_align_pre_calc"]["stage1_model_config"]
    stage1_checkpoint_path = hypes["box_align_pre_calc"]["stage1_model_path"]

    model_filename = "opencood.models." + stage1_model_name
    model_lib = importlib.import_module(model_filename)
    stage1_model_class = None
    target_model_name = stage1_model_name.replace("_", "")
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            stage1_model_class = cls

    stage1_model = stage1_model_class(stage1_model_config)
    state_dict = _load_state_dict_with_spconv_fix(stage1_checkpoint_path)
    stage1_model.load_state_dict(state_dict, strict=False)

    stage1_postprocessor_name = hypes["box_align_pre_calc"]["stage1_postprocessor_name"]
    stage1_postprocessor_config = hypes["box_align_pre_calc"]["stage1_postprocessor_config"]
    postprocessor_lib = importlib.import_module("opencood.data_utils.post_processor")
    stage1_postprocessor_class = None
    target_postprocessor_name = stage1_postprocessor_name.replace("_", "").lower()
    for name, cls in postprocessor_lib.__dict__.items():
        if name.lower() == target_postprocessor_name:
            stage1_postprocessor_class = cls
    stage1_postprocessor = stage1_postprocessor_class(stage1_postprocessor_config, train=False)

    return stage1_model, stage1_postprocessor


def _resolve_split_hypes(hypes, split):
    split = str(split or "test").lower()
    if split == "test":
        hypes = copy.deepcopy(hypes)
        hypes["validate_dir"] = hypes["test_dir"]
        train = False
    elif split in {"val", "validate", "validation"}:
        train = False
    else:
        train = True
    return hypes, train, split


def _parse_args():
    parser = argparse.ArgumentParser(description="Export per-CAV stage1 boxes.")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--comm_range_override", type=float, default=None)
    parser.add_argument(
        "--dump_occ_map",
        action="store_true",
        help="Dump dense occ_map_level0 inline into stage1_boxes.json (very large).",
    )
    parser.add_argument(
        "--dump_occ_map_path",
        action="store_true",
        help=(
            "Store occ_map_level0 as separate .npz files and write occ_map_level0_path into stage1_boxes.json "
            "(recommended)."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards for parallel export.")
    parser.add_argument("--log-interval", type=int, default=50, help="Print progress every N processed samples.")
    return parser.parse_args()


def _extract_occ_map_level0(output_stage1, runtime_store):
    """
    Produce a 2D occupancy-like map for occ-hint correlation.

    Priority:
      1) model runtime store (if available): occ_map_list[0]
      2) detection head scores: sigmoid(cls_preds).max over anchors
    """
    # 1) Runtime store path used by pyramid models.
    try:
        occ_list = (runtime_store or {}).get("occ_map_list")
        if isinstance(occ_list, (list, tuple)) and occ_list:
            occ0 = occ_list[0]
            if torch.is_tensor(occ0):
                occ0 = occ0.detach()
            occ0 = torch.as_tensor(occ0)
            # Expected shapes: [B,1,H,W] or [B,H,W] or [1,H,W] etc.
            if occ0.ndim >= 3:
                occ0 = occ0.squeeze()
            if occ0.ndim == 2:
                return occ0.float().cpu().numpy()
    except Exception:
        pass

    # 2) Fallback: derive occ-like map from cls_preds.
    if isinstance(output_stage1, dict):
        cls_preds = output_stage1.get("cls_preds")
        if torch.is_tensor(cls_preds) and cls_preds.numel() > 0:
            try:
                scores = torch.sigmoid(cls_preds.float())
                if scores.ndim == 4:  # [B, A, H, W]
                    occ = scores.max(dim=1)[0]  # [B,H,W]
                elif scores.ndim == 3:  # [B,H,W]
                    occ = scores
                else:
                    return None
                occ = occ[0] if occ.ndim == 3 else occ
                occ = occ.detach().cpu()
                if occ.ndim == 2:
                    return occ.numpy()
            except Exception:
                return None
    return None


def main():
    opt = _parse_args()
    if int(opt.num_shards) <= 0:
        raise ValueError("num_shards must be >= 1")
    if int(opt.shard_index) < 0 or int(opt.shard_index) >= int(opt.num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    if opt.comm_range_override is not None:
        hypes["comm_range"] = float(opt.comm_range_override)

    hypes, train_flag, split_name = _resolve_split_hypes(hypes, opt.split)

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    _ensure_box_align_config(hypes, opt.stage1_checkpoint, output_dir)

    dataset = build_dataset(hypes, visualize=False, train=train_flag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model, stage1_postprocessor = _build_stage1_model(hypes)
    stage1_anchor_box = torch.from_numpy(stage1_postprocessor.generate_anchor_box())
    for p in stage1_model.parameters():
        p.requires_grad_(False)
    stage1_model.to(device)
    stage1_model.eval()

    stage1_boxes_dict = {}
    max_samples = opt.max_samples
    total_len = len(dataset)
    processed = 0
    dump_occ_inline = bool(opt.dump_occ_map)
    dump_occ_path = bool(opt.dump_occ_map_path)
    split_dir = os.path.join(output_dir, split_name)
    occ_dir = os.path.join(split_dir, "occ_map_level0")
    if dump_occ_inline or dump_occ_path:
        os.makedirs(occ_dir, exist_ok=True)
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if t0 is not None:
        t0.record()
    for idx in range(total_len):
        if max_samples is not None and idx >= int(max_samples):
            break
        if int(opt.num_shards) > 1 and (idx % int(opt.num_shards)) != int(opt.shard_index):
            continue
        base_data_dict = dataset.retrieve_base_data(idx)
        cav_id_list = sorted(list(base_data_dict.keys()), key=lambda x: str(x))
        if not cav_id_list:
            continue

        pred_corner3d_np_list = []
        pred_box3d_np_list = []
        uncertainty_np_list = []
        score_np_list = []
        occ_map_level0_list = []
        occ_map_level0_path_list = []

        for cav_id in cav_id_list:
            sample = _build_single_cav_sample(dataset, base_data_dict, cav_id, idx)
            batch_data = dataset.collate_batch_test([sample])
            if batch_data is None:
                # Keep list alignment with cav_id_list.
                pred_corner3d_np_list.append([])
                pred_box3d_np_list.append([])
                uncertainty_np_list.append([])
                score_np_list.append([])
                occ_map_level0_list.append([])
                occ_map_level0_path_list.append("")
                continue
            batch_data = train_utils.to_device(batch_data, device)
            with torch.no_grad():
                output_stage1 = stage1_model(batch_data["ego"])
                if isinstance(output_stage1, dict) and "unc_preds" not in output_stage1:
                    cls_preds_tensor = output_stage1.get("cls_preds")
                    if cls_preds_tensor is not None:
                        bsz, anchors, height, width = cls_preds_tensor.shape
                        unc_channels = anchors * 2
                        output_stage1["unc_preds"] = cls_preds_tensor.new_zeros(
                            (bsz, unc_channels, height, width),
                            dtype=cls_preds_tensor.dtype,
                            device=cls_preds_tensor.device,
                        )
                # Optional: export dense occ map for V2X-Reg++ occ-hint/occ-pose.
                # Do this even when post-processed boxes are empty so occ-only modes still work.
                occ_inline = []
                occ_path = ""
                if dump_occ_inline or dump_occ_path:
                    runtime_store = getattr(stage1_model, "_runtime_feature_store", {})
                    occ = _extract_occ_map_level0(output_stage1, runtime_store)
                    if occ is not None:
                        try:
                            occ = np.asarray(occ, dtype=np.float32)
                        except Exception:
                            occ = None
                    if occ is not None:
                        if dump_occ_path:
                            occ_path = os.path.join(
                                occ_dir,
                                "{}_{}.npz".format(str(idx), str(cav_id)),
                            )
                            np.savez_compressed(occ_path, occ_map_level0=occ)
                        else:
                            occ_inline = occ.tolist()
                stage1_outputs = stage1_postprocessor.post_process_stage1(
                    output_stage1, stage1_anchor_box
                )
            if not stage1_outputs:
                pred_corner3d_np_list.append([])
                pred_box3d_np_list.append([])
                uncertainty_np_list.append([])
                score_np_list.append([])
                occ_map_level0_list.append(occ_inline if dump_occ_inline else [])
                occ_map_level0_path_list.append(occ_path if dump_occ_path else "")
                continue

            if isinstance(stage1_outputs, (list, tuple)) and len(stage1_outputs) == 3:
                pred_corner3d_list, pred_box3d_list, uncertainty_list = stage1_outputs
                score_list = None
            elif isinstance(stage1_outputs, (list, tuple)) and len(stage1_outputs) >= 4:
                pred_corner3d_list, pred_box3d_list, uncertainty_list, score_list = stage1_outputs[:4]
            else:
                raise ValueError(
                    "Unexpected stage1_postprocessor.post_process_stage1 output: {}".format(
                        type(stage1_outputs)
                    )
                )

            if pred_corner3d_list is None:
                pred_corner3d_np_list.append([])
                pred_box3d_np_list.append([])
                uncertainty_np_list.append([])
                score_np_list.append([])
                occ_map_level0_list.append(occ_inline if dump_occ_inline else [])
                occ_map_level0_path_list.append(occ_path if dump_occ_path else "")
                continue

            pred_corner3d_np_list.append(pred_corner3d_list[0].cpu().numpy().tolist())
            pred_box3d_np_list.append(pred_box3d_list[0].cpu().numpy().tolist())
            uncertainty_np_list.append(uncertainty_list[0].cpu().numpy().tolist())
            if score_list is not None:
                score_np_list.append(score_list[0].cpu().numpy().tolist())
            else:
                score_np_list.append([])

            occ_map_level0_list.append(occ_inline if dump_occ_inline else [])
            occ_map_level0_path_list.append(occ_path if dump_occ_path else "")

        sample_idx_str = str(idx)
        stage1_boxes_dict[sample_idx_str] = OrderedDict()
        stage1_boxes_dict[sample_idx_str]["pred_corner3d_np_list"] = pred_corner3d_np_list
        stage1_boxes_dict[sample_idx_str]["pred_box3d_np_list"] = pred_box3d_np_list
        stage1_boxes_dict[sample_idx_str]["pred_score_np_list"] = score_np_list
        stage1_boxes_dict[sample_idx_str]["uncertainty_np_list"] = uncertainty_np_list
        stage1_boxes_dict[sample_idx_str]["lidar_pose_np"] = [
            np.asarray(
                base_data_dict[cav_id]["params"].get("lidar_pose"),
                dtype=np.float32,
            ).tolist()
            for cav_id in cav_id_list
        ]
        stage1_boxes_dict[sample_idx_str]["lidar_pose_clean_np"] = [
            np.asarray(
                base_data_dict[cav_id]["params"].get(
                    "lidar_pose_clean", base_data_dict[cav_id]["params"].get("lidar_pose")
                ),
                dtype=np.float32,
            ).tolist()
            for cav_id in cav_id_list
        ]
        stage1_boxes_dict[sample_idx_str]["cav_id_list"] = [str(x) for x in cav_id_list]
        if dump_occ_inline:
            stage1_boxes_dict[sample_idx_str]["occ_map_level0"] = occ_map_level0_list
        if dump_occ_path:
            stage1_boxes_dict[sample_idx_str]["occ_map_level0_path"] = occ_map_level0_path_list
        stage1_boxes_dict[sample_idx_str]["veh_frame_id"] = (
            base_data_dict.get("veh_frame_id")
            or next(
                (
                    base_data_dict[cav_id].get("veh_frame_id")
                    for cav_id in cav_id_list
                    if isinstance(base_data_dict.get(cav_id), dict)
                    and base_data_dict[cav_id].get("veh_frame_id") is not None
                ),
                None,
            )
        )
        stage1_boxes_dict[sample_idx_str]["infra_frame_id"] = (
            base_data_dict.get("infra_frame_id")
            or next(
                (
                    base_data_dict[cav_id].get("infra_frame_id")
                    for cav_id in cav_id_list
                    if isinstance(base_data_dict.get(cav_id), dict)
                    and base_data_dict[cav_id].get("infra_frame_id") is not None
                ),
                None,
            )
        )
        stage1_boxes_dict[sample_idx_str]["bev_range"] = (
            np.asarray(dataset.params["model"]["args"]["lidar_range"], dtype=np.float32).tolist()
        )

        processed += 1
        if int(opt.log_interval) > 0 and (processed % int(opt.log_interval) == 0):
            if t0 is not None and t1 is not None:
                t1.record()
                torch.cuda.synchronize()
                ms = t0.elapsed_time(t1)
                rate = processed / max(ms / 1000.0, 1e-9)
                print(
                    "[stage1-export] shard {}/{} processed {} samples (idx={} / {}) rate={:.2f} samples/s".format(
                        int(opt.shard_index),
                        int(opt.num_shards),
                        processed,
                        idx,
                        total_len,
                        rate,
                    ),
                    flush=True,
                )
                t0.record()
            else:
                print(
                    "[stage1-export] shard {}/{} processed {} samples (idx={} / {})".format(
                        int(opt.shard_index), int(opt.num_shards), processed, idx, total_len
                    ),
                    flush=True,
                )

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    if int(opt.num_shards) > 1:
        output_name = "stage1_boxes_shard{:02d}of{:02d}.json".format(int(opt.shard_index), int(opt.num_shards))
    else:
        output_name = "stage1_boxes.json"
    output_path = os.path.join(split_dir, output_name)
    with open(output_path, "w") as f:
        json.dump(stage1_boxes_dict, f, sort_keys=True)
    print("Wrote per-CAV stage1 boxes for {} samples to {}".format(len(stage1_boxes_dict), output_path), flush=True)


if __name__ == "__main__":
    main()
