#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator
from opencood.utils.box_utils import project_box3d
from opencood.utils.transformation_utils import x1_to_x2


def _as_T_from_rt(rotation, translation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj


_dair_cache: Dict[Tuple[str, str], np.ndarray] = {}


def _dair_T_infra_to_vehicle(dair_root: Path, infra_frame_id: str, veh_frame_id: str) -> np.ndarray:
    key = (str(infra_frame_id), str(veh_frame_id))
    if key in _dair_cache:
        return _dair_cache[key]

    infra_frame = str(infra_frame_id)
    veh_frame = str(veh_frame_id)
    infra_calib = dair_root / "infrastructure-side" / "calib" / "virtuallidar_to_world" / f"{infra_frame}.json"
    veh_world = dair_root / "vehicle-side" / "calib" / "novatel_to_world" / f"{veh_frame}.json"
    veh_lidar = dair_root / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{veh_frame}.json"

    infra_obj = _read_json(infra_calib)
    veh_world_obj = _read_json(veh_world)
    veh_lidar_obj = _read_json(veh_lidar)

    T_world_infra = _as_T_from_rt(infra_obj["rotation"], infra_obj["translation"])
    T_world_novatel = _as_T_from_rt(veh_world_obj["rotation"], veh_world_obj["translation"])
    # lidar_to_novatel stores parent=novatel, child=lidar.
    T_novatel_lidar = _as_T_from_rt(
        veh_lidar_obj["transform"]["rotation"],
        [x[0] for x in veh_lidar_obj["transform"]["translation"]],
    )
    T_world_veh_lidar = T_world_novatel @ T_novatel_lidar
    T_veh_lidar_infra = np.linalg.inv(T_world_veh_lidar) @ T_world_infra

    _dair_cache[key] = T_veh_lidar_infra
    return T_veh_lidar_infra


def _yaw_deg_from_T(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(T[1, 0]), float(T[0, 0]))))


def _wrap_angle_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


def _load_stage1(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _extract_single_agent(
    entry: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[List[float]], List[float]]:
    corners_all = entry.get("pred_corner3d_np_list") or []
    scores_all = entry.get("pred_score_np_list") or []
    poses_all = entry.get("lidar_pose_clean_np") or []
    if not isinstance(corners_all, list) or not corners_all:
        return np.zeros((0, 8, 3), dtype=np.float32), None, []
    corners_list = corners_all[0] or []
    corners = np.asarray(corners_list, dtype=np.float32).reshape(-1, 8, 3) if corners_list else np.zeros((0, 8, 3), dtype=np.float32)

    scores = None
    if isinstance(scores_all, list) and scores_all and isinstance(scores_all[0], list):
        try:
            scores = [float(x) for x in scores_all[0]]
        except Exception:
            scores = None

    pose = []
    if isinstance(poses_all, list) and poses_all:
        try:
            pose = [float(x) for x in poses_all[0]]
        except Exception:
            pose = []
    return corners, scores, pose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veh_stage1", type=str, required=True, help="Path to vehicle stage1_boxes.json")
    ap.add_argument("--infra_stage1", type=str, required=True, help="Path to infrastructure stage1_boxes.json")
    ap.add_argument(
        "--dair_root",
        type=str,
        default="~/datasets/data2/DAIR-V2X-C/cooperative-vehicle-infrastructure",
        help="DAIR-V2X-C root dir; used to compute T_true from calib (set to empty to disable).",
    )
    ap.add_argument("--seed_strategy", type=str, default="topk_radius", choices=["topk_radius", "all"])
    ap.add_argument("--use_gnn", action="store_true", help="Use learned edge features (requires meaningful ckpt)")
    ap.add_argument("--ckpt_path", type=str, default=None, help="Optional EdgeGAT checkpoint path")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_boxes", type=int, default=60)
    ap.add_argument("--anchor_topk", type=int, default=10)
    ap.add_argument("--anchor_max_count", type=int, default=4)
    ap.add_argument("--min_nodes", type=int, default=5)
    ap.add_argument("--sim_threshold", type=float, default=0.6)
    ap.add_argument("--anchor_threshold", type=float, default=None, help="Optional anchor threshold; defaults to sim_threshold.")
    ap.add_argument("--box_threshold", type=float, default=None, help="Optional subgraph threshold; defaults to sim_threshold.")
    ap.add_argument("--anchor_consistency_multiplier", type=float, default=2.0)
    ap.add_argument(
        "--selection_mode",
        type=str,
        default="max_size_then_min_eps",
        choices=["max_size_then_min_eps", "min_eps_then_max_size"],
    )
    ap.add_argument("--affine_method", type=str, default="lmeds", choices=["lmeds", "ransac"])
    ap.add_argument("--ransac_reproj_threshold", type=float, default=1.0)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--refine_iters", type=int, default=10)
    ap.add_argument("--refit_rigid", action="store_true", help="Refit a rigid SE(2) after robust affine estimation.")
    ap.add_argument("--no_refit_rigid", action="store_true", help="Disable rigid refit and return affine transform.")
    ap.add_argument("--min_inliers", type=int, default=3)
    ap.add_argument("--full_consistency", action="store_true", help="Require new nodes consistent with all matched nodes.")
    ap.add_argument("--no_full_consistency", action="store_true", help="Disable full-consistency check (faster, less strict).")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument(
        "--gt_match_dist",
        type=float,
        default=5.0,
        help="Diagnostics only: count GT-overlap by nearest-center matching after projecting with T_true (meters).",
    )
    ap.add_argument("--out", type=str, default=None, help="Optional path to write per-frame results json")
    args = ap.parse_args()

    if args.refit_rigid and args.no_refit_rigid:
        raise SystemExit("Cannot set both --refit_rigid and --no_refit_rigid")
    if args.full_consistency and args.no_full_consistency:
        raise SystemExit("Cannot set both --full_consistency and --no_full_consistency")

    refit_rigid = True if not args.no_refit_rigid else False
    if args.refit_rigid:
        refit_rigid = True

    full_consistency = True if not args.no_full_consistency else False
    if args.full_consistency:
        full_consistency = True

    veh_path = resolve_repo_path(args.veh_stage1)
    infra_path = resolve_repo_path(args.infra_stage1)
    veh_data = _load_stage1(veh_path)
    infra_data = _load_stage1(infra_path)

    cfg = FreeAlignPaperConfig(
        ckpt_path=None if not args.ckpt_path else str(resolve_repo_path(args.ckpt_path)),
        device=str(args.device),
        use_gnn=bool(args.use_gnn),
        max_boxes=int(args.max_boxes),
        anchor_topk=int(args.anchor_topk),
        seed_strategy=str(args.seed_strategy),
        min_nodes=int(args.min_nodes),
        anchor_max_count=int(args.anchor_max_count),
        sim_threshold=float(args.sim_threshold),
        anchor_threshold=None if args.anchor_threshold is None else float(args.anchor_threshold),
        box_threshold=None if args.box_threshold is None else float(args.box_threshold),
        anchor_consistency_multiplier=float(args.anchor_consistency_multiplier),
        selection_mode=str(args.selection_mode),
        affine_method=str(args.affine_method),
        ransac_reproj_threshold=float(args.ransac_reproj_threshold),
        max_iters=int(args.max_iters),
        confidence=float(args.confidence),
        refine_iters=int(args.refine_iters),
        refit_rigid=bool(refit_rigid),
        min_inliers=int(args.min_inliers),
        full_consistency=bool(full_consistency),
        p=float(args.p),
    )
    estimator = FreeAlignPaperEstimator(cfg)

    keys = sorted(set(veh_data.keys()) & set(infra_data.keys()), key=lambda x: int(x))
    if not keys:
        raise SystemExit("No shared sample keys between the two stage1 files.")

    records = []
    failures = 0
    re_list = []
    te_list = []
    matches_list = []
    gt_overlap_list = []
    gt_overlap_ge3 = 0

    for key in keys:
        veh_entry = veh_data.get(key) or {}
        infra_entry = infra_data.get(key) or {}
        veh_corners, veh_scores, veh_pose = _extract_single_agent(veh_entry)
        infra_corners, infra_scores, infra_pose = _extract_single_agent(infra_entry)

        veh_boxes = corners_to_bbox3d_list(veh_corners, bbox_type="detected", scores=veh_scores)
        infra_boxes = corners_to_bbox3d_list(infra_corners, bbox_type="detected", scores=infra_scores)

        T_true = None
        dair_root = str(args.dair_root or "").strip()
        if dair_root:
            try:
                T_true = _dair_T_infra_to_vehicle(Path(dair_root).expanduser(), str(infra_entry.get("infra_frame_id")), str(veh_entry.get("veh_frame_id")))
            except Exception:
                T_true = None
        if T_true is None and len(veh_pose) >= 6 and len(infra_pose) >= 6:
            T_true = x1_to_x2(infra_pose, veh_pose)  # infra -> vehicle (fallback)

        T_est, stability, matches, meta = estimator.estimate(cav_boxes=infra_boxes, ego_boxes=veh_boxes, T_init=None)

        rec = {
            "key": key,
            "veh_frame_id": veh_entry.get("veh_frame_id"),
            "infra_frame_id": veh_entry.get("infra_frame_id"),
            "veh_boxes": int(veh_corners.shape[0]),
            "infra_boxes": int(infra_corners.shape[0]),
            "success": bool(T_est is not None),
            "stability": float(stability),
            "matches": int(matches),
            "meta": meta or {},
        }

        if T_true is not None and veh_corners.shape[0] and infra_corners.shape[0]:
            infra_proj = project_box3d(infra_corners, T_true)
            veh_cent = veh_corners.mean(1)[:, :2]
            infra_cent = infra_proj.mean(1)[:, :2]
            used = set()
            gt_match = 0
            thr = float(args.gt_match_dist)
            if veh_cent.size and infra_cent.size and thr > 0:
                for j in range(infra_cent.shape[0]):
                    d = np.linalg.norm(veh_cent - infra_cent[j : j + 1], axis=1)
                    i = int(d.argmin())
                    if float(d[i]) <= thr and i not in used:
                        used.add(i)
                        gt_match += 1
            rec["gt_overlap_nn"] = int(gt_match)
            gt_overlap_list.append(float(gt_match))
            if gt_match >= 3:
                gt_overlap_ge3 += 1
        else:
            rec["gt_overlap_nn"] = None

        if T_est is None or T_true is None:
            failures += 1
            rec.update({"RE_deg": None, "TE_m": None})
        else:
            yaw_est = _yaw_deg_from_T(np.asarray(T_est))
            yaw_true = _yaw_deg_from_T(np.asarray(T_true))
            RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
            t_est = np.asarray(T_est, dtype=np.float64)[0:2, 3]
            t_true = np.asarray(T_true, dtype=np.float64)[0:2, 3]
            TE = float(np.linalg.norm(t_est - t_true))
            rec.update({"RE_deg": float(RE), "TE_m": float(TE)})
            re_list.append(float(RE))
            te_list.append(float(TE))
            matches_list.append(int(matches))

        records.append(rec)

    def _summ(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary = {
        "total": int(len(keys)),
        "failures": int(failures),
        "success_rate": float((len(keys) - failures) / max(len(keys), 1)),
        "RE_deg": _summ(re_list),
        "TE_m": _summ(te_list),
        "matches": _summ([float(x) for x in matches_list]),
        "gt_overlap_nn": _summ(gt_overlap_list),
        "gt_overlap_nn_ge3_rate": float(gt_overlap_ge3 / max(len(keys), 1)),
        "config": cfg.__dict__,
        "inputs": {"veh_stage1": str(veh_path), "infra_stage1": str(infra_path)},
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
