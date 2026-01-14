#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.pose.pgc import PGCInferConfig, PGCNet, infer_pose_and_confidence
from opencood.utils.transformation_utils import tfm_to_pose


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PGC on a HEAL/OpenCOOD dataset and export per-frame poses/confidences.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hypes_yaml", "-y", type=str, help="HEAL/OpenCOOD yaml for dataset paths.")
    g.add_argument("--model_dir", type=str, help="HEAL checkpoint folder containing config.yaml (dataset).")

    p.add_argument("--pgc_ckpt", type=str, required=True, help="PGC checkpoint (from train_v2vloc_pgc.py).")
    p.add_argument("--out_json", type=str, required=True, help="Output json mapping sample_idx -> predicted poses.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max_samples", type=int, default=0, help="Limit number of frames (0=all).")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--num_points", type=int, default=4096)
    p.add_argument("--rsd_voxel_size", type=float, default=0.2)
    p.add_argument("--ransac_iter", type=int, default=64)
    p.add_argument("--ransac_inlier_th", type=float, default=1.0)
    p.add_argument("--ransac_min_inliers", type=int, default=16)
    return p.parse_args()


def _load_pgc(ckpt_path: str, device: torch.device) -> PGCNet:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        feat_dim = int((state.get("args") or {}).get("feat_dim", 256))
    else:
        sd = state
        feat_dim = 256
    model = PGCNet(in_dim=4, feat_dim=int(feat_dim))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    device = torch.device(str(args.device))

    if args.model_dir:
        hypes = yaml_utils.load_yaml(str(Path(args.model_dir) / "config.yaml"), None)
    else:
        hypes = yaml_utils.load_yaml(args.hypes_yaml, None)

    train = args.split == "train"
    if args.split == "test" and "test_dir" in hypes:
        hypes = dict(hypes)
        hypes["validate_dir"] = hypes["test_dir"]
        train = False

    ds = build_dataset(hypes, visualize=False, train=train)
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else len(ds)

    model = _load_pgc(args.pgc_ckpt, device)
    cfg = PGCInferConfig(
        num_points=int(args.num_points),
        rsd_voxel_size=float(args.rsd_voxel_size),
        ransac_iter=int(args.ransac_iter),
        ransac_inlier_th=float(args.ransac_inlier_th),
        ransac_min_inliers=int(args.ransac_min_inliers),
    )

    out: Dict[str, Dict[str, object]] = {}
    for idx in range(int(max_samples)):
        base = ds.retrieve_base_data(int(idx))
        cav_id_list = list(base.keys())
        poses_pred = []
        conf_list = []
        for cav_id in cav_id_list:
            pts = base[cav_id].get("lidar_np")
            if pts is None:
                poses_pred.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                conf_list.append(0.0)
                continue
            T, conf, _ = infer_pose_and_confidence(model, pts, cfg=cfg, device=device)
            pose = tfm_to_pose(np.asarray(T, dtype=np.float64))
            poses_pred.append([float(x) for x in pose])
            conf_list.append(float(conf))

        out[str(idx)] = {
            "cav_id_list": [str(x) for x in cav_id_list],
            "lidar_pose_pred_np": poses_pred,
            "pose_confidence_np": conf_list,
        }

        if (idx + 1) % 50 == 0:
            print(f"[pgc-infer] {idx+1}/{int(max_samples)}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[pgc-infer] saved: {out_path}")


if __name__ == "__main__":
    main()
