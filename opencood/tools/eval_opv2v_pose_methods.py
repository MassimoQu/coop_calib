#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.late_fusion.v2xregpp import V2XRegPPEstimator
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator


def _wrap_angle_deg(angle_deg: float) -> float:
    return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)


def _angle_abs_deg(angle_deg: float) -> float:
    return abs(_wrap_angle_deg(angle_deg))


def _se2_from_pose(pose6: Sequence[float]) -> np.ndarray:
    x = float(pose6[0])
    y = float(pose6[1])
    yaw_deg = float(pose6[4])
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 2] = x
    T[1, 2] = y
    return T


def _inv_se2(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2:3]
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = R.T
    out[:2, 2:3] = -R.T @ t
    return out


def _rel_se2(ego_pose6: Sequence[float], cav_pose6: Sequence[float]) -> np.ndarray:
    T_e = _se2_from_pose(ego_pose6)
    T_c = _se2_from_pose(cav_pose6)
    return _inv_se2(T_e) @ T_c


def _se2_error(T_rel_est: np.ndarray, T_rel_true: np.ndarray) -> Tuple[float, float]:
    delta = _inv_se2(T_rel_true) @ T_rel_est
    te = float(np.hypot(float(delta[0, 2]), float(delta[1, 2])))
    re = _angle_abs_deg(math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0]))))
    return te, re


def _se2_from_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = T[:2, :2]
    out[0, 2] = float(T[0, 3])
    out[1, 2] = float(T[1, 3])
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return data


def _sorted_keys(data: Mapping[str, Any]) -> List[str]:
    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    return sorted(data.keys(), key=_key)


def _summ(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


@dataclass(frozen=True)
class MethodMetrics:
    pairs: int
    success: int
    failures: int
    error_rate_te_gt_3m: float
    te_m: Dict[str, Optional[float]]
    re_deg: Dict[str, Optional[float]]


def _compute_metrics(
    *,
    errors: List[Tuple[Optional[float], Optional[float]]],
    te_error_threshold_m: float,
) -> MethodMetrics:
    pairs = int(len(errors))
    te_values: List[float] = []
    re_values: List[float] = []
    failures = 0
    err = 0
    for te, re in errors:
        if te is None or re is None:
            failures += 1
            err += 1
            continue
        te_values.append(float(te))
        re_values.append(float(re))
        if float(te) > float(te_error_threshold_m):
            err += 1
    success = pairs - failures
    return MethodMetrics(
        pairs=pairs,
        success=int(success),
        failures=int(failures),
        error_rate_te_gt_3m=float(err / max(pairs, 1)),
        te_m=_summ(te_values),
        re_deg=_summ(re_values),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate OPV2V pose estimation from stage1 boxes (FreeAlign-paper vs V2XReg++).")
    ap.add_argument("--stage1_cache", type=str, required=True, help="Path to stage1_boxes.json (must include lidar_pose_clean_np).")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap on frames to evaluate (in key order).")
    ap.add_argument("--comm_range", type=float, default=70.0, help="Only evaluate pairs within this clean range (meters). Set <=0 to disable.")
    ap.add_argument("--te_error_threshold_m", type=float, default=3.0, help="Paper Table-II error rate threshold on translation error (meters).")
    ap.add_argument(
        "--methods",
        type=str,
        default="freealign_paper,v2xregpp",
        help="CSV subset of {freealign_paper,v2xregpp} to evaluate.",
    )

    # FreeAlign-paper knobs (distance-only edges unless --use_gnn).
    ap.add_argument("--freealign_max_boxes", type=int, default=30)
    ap.add_argument("--freealign_anchor_topk", type=int, default=10)
    ap.add_argument("--freealign_seed_strategy", type=str, default="topk_radius", choices=["topk_radius", "all"])
    ap.add_argument("--freealign_min_nodes", type=int, default=5)
    ap.add_argument("--freealign_anchor_max_count", type=int, default=4)
    ap.add_argument("--freealign_sim_threshold", type=float, default=0.3)
    ap.add_argument("--freealign_selection_mode", type=str, default="max_size_then_min_eps", choices=["max_size_then_min_eps", "min_eps_then_max_size"])
    ap.add_argument("--freealign_affine_method", type=str, default="lmeds", choices=["lmeds", "ransac"])
    ap.add_argument("--freealign_refit_rigid", action="store_true", default=True)
    ap.add_argument("--freealign_no_refit_rigid", action="store_true")
    ap.add_argument("--freealign_use_gnn", action="store_true")
    ap.add_argument("--freealign_ckpt_path", type=str, default=None)
    ap.add_argument("--freealign_device", type=str, default="cpu")

    # V2XReg++ knobs.
    ap.add_argument("--v2xregpp_config", type=str, default="configs/pipeline_detection_pp_ft.yaml")
    ap.add_argument("--v2xregpp_min_matches", type=int, default=3)
    ap.add_argument("--v2xregpp_min_stability", type=float, default=0.0)

    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = ap.parse_args()
    methods = {s.strip().lower() for s in str(args.methods or "").split(",") if s.strip()}
    if not methods:
        raise SystemExit("Empty --methods")
    allowed_methods = {"freealign_paper", "v2xregpp"}
    unknown = sorted(methods - allowed_methods)
    if unknown:
        raise SystemExit(f"Unknown methods in --methods: {unknown}")

    if bool(args.freealign_refit_rigid) and bool(args.freealign_no_refit_rigid):
        raise SystemExit("Cannot set both --freealign_refit_rigid and --freealign_no_refit_rigid.")
    refit_rigid = bool(args.freealign_refit_rigid) and not bool(args.freealign_no_refit_rigid)

    stage1_path = resolve_repo_path(args.stage1_cache)
    data = _read_json(stage1_path)
    keys = _sorted_keys(data)
    if args.max_frames:
        keys = keys[: int(args.max_frames)]

    fa_cfg = FreeAlignPaperConfig(
        ckpt_path=None if not args.freealign_ckpt_path else str(resolve_repo_path(args.freealign_ckpt_path)),
        device=str(args.freealign_device),
        use_gnn=bool(args.freealign_use_gnn),
        max_boxes=int(args.freealign_max_boxes),
        anchor_topk=int(args.freealign_anchor_topk),
        seed_strategy=str(args.freealign_seed_strategy),
        min_nodes=int(args.freealign_min_nodes),
        anchor_max_count=int(args.freealign_anchor_max_count),
        sim_threshold=float(args.freealign_sim_threshold),
        selection_mode=str(args.freealign_selection_mode),
        affine_method=str(args.freealign_affine_method),
        refit_rigid=bool(refit_rigid),
    )
    freealign = FreeAlignPaperEstimator(fa_cfg)

    v2xregpp = None
    if "v2xregpp" in methods:
        v2xregpp = V2XRegPPEstimator(config_path=str(args.v2xregpp_config))

    te_thr = float(args.te_error_threshold_m)
    comm_range = float(args.comm_range)

    fa_errors: List[Tuple[Optional[float], Optional[float]]] = []
    v2_errors: List[Tuple[Optional[float], Optional[float]]] = []
    pairs_total = 0
    pairs_in_range = 0

    for key in keys:
        entry = data.get(key)
        if entry is None:
            continue

        corners_all = entry.get("pred_corner3d_np_list") or []
        scores_all = entry.get("pred_score_np_list") or []
        poses_all = entry.get("lidar_pose_clean_np") or []
        if not isinstance(corners_all, list) or not isinstance(poses_all, list):
            continue
        if len(corners_all) < 2 or len(poses_all) < 2:
            continue

        ego_pose = poses_all[0]
        def _scores_for(agent_idx: int) -> Optional[List[float]]:
            if not isinstance(scores_all, list) or agent_idx < 0 or agent_idx >= len(scores_all):
                return None
            scores = scores_all[agent_idx]
            if not isinstance(scores, list) or not scores:
                return None
            try:
                return [float(x) for x in scores]
            except Exception:
                return None

        ego_corners = np.asarray(corners_all[0] or [], dtype=np.float32).reshape(-1, 8, 3)
        ego_scores = _scores_for(0)
        if ego_scores is not None and len(ego_scores) != int(ego_corners.shape[0]):
            ego_scores = None
        ego_boxes = corners_to_bbox3d_list(ego_corners, bbox_type="detected", scores=ego_scores)

        for cav_idx in range(1, len(corners_all)):
            cav_pose = poses_all[cav_idx]
            if not isinstance(cav_pose, (list, tuple)) or len(cav_pose) < 6:
                continue
            if not isinstance(ego_pose, (list, tuple)) or len(ego_pose) < 6:
                continue

            T_true = _rel_se2(ego_pose, cav_pose)
            dist = float(np.hypot(float(T_true[0, 2]), float(T_true[1, 2])))
            pairs_total += 1
            if comm_range > 0 and dist > comm_range:
                continue
            pairs_in_range += 1

            cav_corners = np.asarray(corners_all[cav_idx] or [], dtype=np.float32).reshape(-1, 8, 3)
            cav_scores = _scores_for(int(cav_idx))
            if cav_scores is not None and len(cav_scores) != int(cav_corners.shape[0]):
                cav_scores = None
            cav_boxes = corners_to_bbox3d_list(cav_corners, bbox_type="detected", scores=cav_scores)

            # FreeAlign-paper (prior-free).
            if "freealign_paper" in methods:
                T_fa, stability, matches, _ = freealign.estimate(cav_boxes=cav_boxes, ego_boxes=ego_boxes, T_init=None)
                if T_fa is None:
                    fa_errors.append((None, None))
                else:
                    del stability, matches
                    te, re = _se2_error(_se2_from_T(T_fa), T_true)
                    fa_errors.append((float(te), float(re)))

            # V2XReg++ (init-free).
            if "v2xregpp" in methods:
                assert v2xregpp is not None
                est = v2xregpp.estimate(infra_boxes=cav_boxes, veh_boxes=ego_boxes, init=None, ctx=None)
                if (
                    not bool(est.success)
                    or est.T is None
                    or int(len(est.matches or [])) < int(args.v2xregpp_min_matches)
                    or float(est.stability) < float(args.v2xregpp_min_stability)
                ):
                    v2_errors.append((None, None))
                else:
                    te, re = _se2_error(_se2_from_T(est.T), T_true)
                    v2_errors.append((float(te), float(re)))

    payload: Dict[str, Any] = {
        "inputs": {
            "stage1_cache": str(stage1_path),
            "frames": int(len(keys)),
            "comm_range": float(comm_range),
            "te_error_threshold_m": float(te_thr),
            "methods": sorted(methods),
        },
        "counts": {
            "pairs_total": int(pairs_total),
            "pairs_in_range": int(pairs_in_range),
        },
        "freealign_paper": None
        if "freealign_paper" not in methods
        else asdict(_compute_metrics(errors=fa_errors, te_error_threshold_m=te_thr)),
        "v2xregpp": None if "v2xregpp" not in methods else asdict(_compute_metrics(errors=v2_errors, te_error_threshold_m=te_thr)),
        "freealign_config": asdict(fa_cfg),
        "v2xregpp_config": str(resolve_repo_path(args.v2xregpp_config)),
    }

    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
