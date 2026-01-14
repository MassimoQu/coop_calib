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


def _infer_pair_indices(cav_id_list: Any) -> Tuple[int, int]:
    if isinstance(cav_id_list, list):
        s = [str(x).lower() for x in cav_id_list]
        if "vehicle" in s and "infrastructure" in s:
            return int(s.index("infrastructure")), int(s.index("vehicle"))
    return 0, 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_cache", type=str, required=True, help="Dual-agent stage1 cache json (dict of entries).")
    ap.add_argument(
        "--dair_root",
        type=str,
        default="~/datasets/data2/DAIR-V2X-C/cooperative-vehicle-infrastructure",
        help="DAIR-V2X-C root dir; used to compute T_true from calib.",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on samples to evaluate (for quick runs).")
    ap.add_argument("--seed_strategy", type=str, default="topk_radius", choices=["topk_radius", "all"])
    ap.add_argument("--use_gnn", action="store_true")
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_boxes", type=int, default=60)
    ap.add_argument("--anchor_topk", type=int, default=10)
    ap.add_argument("--anchor_max_count", type=int, default=6)
    ap.add_argument("--min_nodes", type=int, default=5)
    ap.add_argument("--sim_threshold", type=float, default=1.0)
    ap.add_argument("--selection_mode", type=str, default="max_size_then_min_eps", choices=["max_size_then_min_eps", "min_eps_then_max_size"])
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stage1_path = resolve_repo_path(args.stage1_cache)
    dair_root = Path(str(args.dair_root)).expanduser().resolve()
    data = json.loads(stage1_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("stage1_cache must be a dict JSON")

    keys = sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if args.max_samples:
        keys = keys[: int(args.max_samples)]

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
        selection_mode=str(args.selection_mode),
        p=float(args.p),
    )
    estimator = FreeAlignPaperEstimator(cfg)

    failures = 0
    re_list: List[float] = []
    te_list: List[float] = []
    matches_list: List[int] = []
    records = []

    for key in keys:
        entry = data.get(key) or {}
        cav_ids = entry.get("cav_id_list")
        infra_idx, veh_idx = _infer_pair_indices(cav_ids)

        corners_all = entry.get("pred_corner3d_np_list") or []
        scores_all = entry.get("pred_score_np_list") or []
        if not isinstance(corners_all, list) or len(corners_all) < 2:
            failures += 1
            continue

        infra_c = np.asarray(corners_all[infra_idx], dtype=np.float32).reshape(-1, 8, 3)
        veh_c = np.asarray(corners_all[veh_idx], dtype=np.float32).reshape(-1, 8, 3)
        infra_s = scores_all[infra_idx] if isinstance(scores_all, list) and len(scores_all) > infra_idx else None
        veh_s = scores_all[veh_idx] if isinstance(scores_all, list) and len(scores_all) > veh_idx else None

        infra_boxes = corners_to_bbox3d_list(infra_c, bbox_type="detected", scores=infra_s)
        veh_boxes = corners_to_bbox3d_list(veh_c, bbox_type="detected", scores=veh_s)

        T_true = None
        try:
            T_true = _dair_T_infra_to_vehicle(dair_root, str(entry.get("infra_frame_id")), str(entry.get("veh_frame_id")))
        except Exception:
            T_true = None

        T_est, stability, matches, meta = estimator.estimate(cav_boxes=infra_boxes, ego_boxes=veh_boxes, T_init=None)

        rec = {
            "key": key,
            "veh_frame_id": entry.get("veh_frame_id"),
            "infra_frame_id": entry.get("infra_frame_id"),
            "veh_boxes": int(veh_c.shape[0]),
            "infra_boxes": int(infra_c.shape[0]),
            "success": bool(T_est is not None),
            "stability": float(stability),
            "matches": int(matches),
            "meta": meta or {},
        }

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

    summary = {
        "total": int(len(keys)),
        "failures": int(failures),
        "success_rate": float((len(keys) - failures) / max(len(keys), 1)),
        "RE_deg": _summ(re_list),
        "TE_m": _summ(te_list),
        "matches": _summ([float(x) for x in matches_list]),
        "config": cfg.__dict__,
        "inputs": {"stage1_cache": str(stage1_path)},
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()

