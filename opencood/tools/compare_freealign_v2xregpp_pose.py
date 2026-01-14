#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.late_fusion.v2xregpp import V2XRegPPEstimator
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator
from opencood.pose.freealign_repo import FreeAlignRepoConfig, FreeAlignRepoEstimator


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


def _extract_single_agent(entry: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[float]]]:
    corners_all = entry.get("pred_corner3d_np_list") or []
    scores_all = entry.get("pred_score_np_list") or []
    if not isinstance(corners_all, list) or not corners_all:
        return np.zeros((0, 8, 3), dtype=np.float32), None
    corners_list = corners_all[0] or []
    corners = (
        np.asarray(corners_list, dtype=np.float32).reshape(-1, 8, 3)
        if corners_list
        else np.zeros((0, 8, 3), dtype=np.float32)
    )
    scores = None
    if isinstance(scores_all, list) and scores_all and isinstance(scores_all[0], list):
        try:
            scores = [float(x) for x in scores_all[0]]
        except Exception:
            scores = None
    return corners, scores


def _extract_dual_agent(entry: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[float]], np.ndarray, Optional[List[float]]]:
    cav_ids = entry.get("cav_id_list")
    infra_idx, veh_idx = _infer_pair_indices(cav_ids)
    corners_all = entry.get("pred_corner3d_np_list") or []
    scores_all = entry.get("pred_score_np_list") or []
    if not isinstance(corners_all, list) or len(corners_all) < 2:
        return np.zeros((0, 8, 3), dtype=np.float32), None, np.zeros((0, 8, 3), dtype=np.float32), None
    infra_c = np.asarray(corners_all[infra_idx], dtype=np.float32).reshape(-1, 8, 3)
    veh_c = np.asarray(corners_all[veh_idx], dtype=np.float32).reshape(-1, 8, 3)
    infra_s = scores_all[infra_idx] if isinstance(scores_all, list) and len(scores_all) > infra_idx else None
    veh_s = scores_all[veh_idx] if isinstance(scores_all, list) and len(scores_all) > veh_idx else None
    return infra_c, infra_s, veh_c, veh_s


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage1_cache", type=str, help="Dual-agent stage1 cache json (dict of entries).")
    group.add_argument("--stage1_exports", action="store_true", help="Use separate infra/veh stage1_boxes.json files.")
    ap.add_argument("--veh_stage1", type=str, default=None, help="Required when using --stage1_exports.")
    ap.add_argument("--infra_stage1", type=str, default=None, help="Required when using --stage1_exports.")
    ap.add_argument(
        "--dair_root",
        type=str,
        default="~/datasets/data2/DAIR-V2X-C/cooperative-vehicle-infrastructure",
        help="DAIR-V2X-C root dir; used to compute T_true from calib.",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on evaluated samples.")

    # FreeAlign (paper reconstruction: EdgeGAT + MASS).
    ap.add_argument("--freealign_paper_seed_strategy", type=str, default="all", choices=["topk_radius", "all"])
    ap.add_argument("--freealign_paper_max_boxes", type=int, default=60)
    ap.add_argument("--freealign_paper_anchor_topk", type=int, default=10)
    ap.add_argument("--freealign_paper_anchor_max_count", type=int, default=4)
    ap.add_argument("--freealign_paper_min_nodes", type=int, default=6)
    ap.add_argument("--freealign_paper_sim_threshold", type=float, default=0.8)
    ap.add_argument("--freealign_paper_p", type=float, default=1.0)
    ap.add_argument("--freealign_paper_use_gnn", action="store_true")
    ap.add_argument("--freealign_paper_ckpt_path", type=str, default=None)
    ap.add_argument("--freealign_paper_device", type=str, default="cpu")

    # FreeAlign (released repo matching: match_v7_with_detection).
    ap.add_argument("--freealign_repo_max_boxes", type=int, default=60)
    ap.add_argument("--freealign_repo_min_nodes", type=int, default=3)
    ap.add_argument("--freealign_repo_min_anchors", type=int, default=3)
    ap.add_argument("--freealign_repo_anchor_error", type=float, default=0.3)
    ap.add_argument("--freealign_repo_box_error", type=float, default=0.5)

    # V2XReg++ config.
    ap.add_argument("--v2xregpp_config", type=str, default="configs/pipeline_detection_pp.yaml")
    ap.add_argument("--v2xregpp_detected_thr", type=float, default=None, help="Override matching.distance_thresholds.detected")
    ap.add_argument("--v2xregpp_filter_threshold", type=int, default=None, help="Override matching.filter_threshold")
    ap.add_argument("--v2xregpp_min_stability", type=float, default=0.0, help="Treat as failure if stability < this.")
    ap.add_argument("--v2xregpp_min_matches", type=int, default=1, help="Treat as failure if matched pairs < this.")

    # Evaluation thresholds.
    ap.add_argument("--te_thresholds", type=str, default="1,2,3", help="CSV meters; used for success-rate matrix.")
    ap.add_argument("--re_thresholds", type=str, default="1,2,3", help="CSV degrees; used for success-rate matrix.")

    ap.add_argument("--out", type=str, default=None, help="Optional path to write per-sample results json")
    args = ap.parse_args()

    dair_root = Path(str(args.dair_root)).expanduser().resolve()
    te_thrs = [float(x) for x in str(args.te_thresholds).split(",") if str(x).strip()]
    re_thrs = [float(x) for x in str(args.re_thresholds).split(",") if str(x).strip()]
    if not te_thrs or not re_thrs:
        raise SystemExit("Empty thresholds list.")

    freealign_paper_cfg = FreeAlignPaperConfig(
        ckpt_path=None
        if not args.freealign_paper_ckpt_path
        else str(resolve_repo_path(args.freealign_paper_ckpt_path)),
        device=str(args.freealign_paper_device),
        use_gnn=bool(args.freealign_paper_use_gnn),
        max_boxes=int(args.freealign_paper_max_boxes),
        anchor_topk=int(args.freealign_paper_anchor_topk),
        seed_strategy=str(args.freealign_paper_seed_strategy),
        min_nodes=int(args.freealign_paper_min_nodes),
        anchor_max_count=int(args.freealign_paper_anchor_max_count),
        sim_threshold=float(args.freealign_paper_sim_threshold),
        p=float(args.freealign_paper_p),
    )
    freealign_paper_est = FreeAlignPaperEstimator(freealign_paper_cfg)

    freealign_repo_cfg = FreeAlignRepoConfig(
        max_boxes=int(args.freealign_repo_max_boxes),
        min_anchors=int(args.freealign_repo_min_anchors),
        anchor_error=float(args.freealign_repo_anchor_error),
        box_error=float(args.freealign_repo_box_error),
        min_nodes=int(args.freealign_repo_min_nodes),
    )
    freealign_repo_est = FreeAlignRepoEstimator(freealign_repo_cfg)

    overrides: Dict[str, Any] = {}
    if args.v2xregpp_detected_thr is not None:
        overrides["distance_thresholds"] = {"detected": float(args.v2xregpp_detected_thr)}
    if args.v2xregpp_filter_threshold is not None:
        overrides["filter_threshold"] = int(args.v2xregpp_filter_threshold)
    v2xregpp_est = V2XRegPPEstimator(config_path=str(args.v2xregpp_config), matching_overrides=overrides or None)

    samples: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    if args.stage1_cache:
        cache_path = resolve_repo_path(args.stage1_cache)
        data = json.loads(cache_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("stage1_cache must be a dict JSON")
        keys = sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if args.max_samples:
            keys = keys[: int(args.max_samples)]
        for k in keys:
            entry = data.get(k) or {}
            samples.append((str(k), entry, entry))
    else:
        if not args.stage1_exports or not args.veh_stage1 or not args.infra_stage1:
            raise SystemExit("When using --stage1_exports, set --veh_stage1 and --infra_stage1.")
        veh_path = resolve_repo_path(args.veh_stage1)
        infra_path = resolve_repo_path(args.infra_stage1)
        veh_data = json.loads(veh_path.read_text())
        infra_data = json.loads(infra_path.read_text())
        if not isinstance(veh_data, dict) or not isinstance(infra_data, dict):
            raise ValueError("stage1 export json must be dict")
        keys = sorted(set(veh_data.keys()) & set(infra_data.keys()), key=lambda x: int(x))
        if args.max_samples:
            keys = keys[: int(args.max_samples)]
        for k in keys:
            samples.append((str(k), infra_data.get(k) or {}, veh_data.get(k) or {}))

    if not samples:
        raise SystemExit("No samples.")

    per_sample = []
    for key, infra_entry, veh_entry in samples:
        if args.stage1_cache:
            infra_c, infra_s, veh_c, veh_s = _extract_dual_agent(infra_entry)
            infra_frame = infra_entry.get("infra_frame_id")
            veh_frame = infra_entry.get("veh_frame_id")
        else:
            infra_c, infra_s = _extract_single_agent(infra_entry)
            veh_c, veh_s = _extract_single_agent(veh_entry)
            infra_frame = infra_entry.get("infra_frame_id")
            veh_frame = veh_entry.get("veh_frame_id")

        infra_boxes = corners_to_bbox3d_list(infra_c, bbox_type="detected", scores=infra_s)
        veh_boxes = corners_to_bbox3d_list(veh_c, bbox_type="detected", scores=veh_s)

        T_true = None
        try:
            T_true = _dair_T_infra_to_vehicle(dair_root, str(infra_frame), str(veh_frame))
        except Exception:
            T_true = None

        rec: Dict[str, Any] = {
            "key": key,
            "veh_frame_id": veh_frame,
            "infra_frame_id": infra_frame,
            "veh_boxes": int(veh_c.shape[0]),
            "infra_boxes": int(infra_c.shape[0]),
        }

        # FreeAlign (paper reconstruction)
        fa_paper_T, _, fa_paper_matches, fa_paper_meta = freealign_paper_est.estimate(
            cav_boxes=infra_boxes, ego_boxes=veh_boxes, T_init=None
        )
        rec["freealign_paper"] = {
            "success": bool(fa_paper_T is not None),
            "matches": int(fa_paper_matches),
            "meta": fa_paper_meta or {},
        }
        if fa_paper_T is not None and T_true is not None:
            yaw_est = _yaw_deg_from_T(np.asarray(fa_paper_T))
            yaw_true = _yaw_deg_from_T(np.asarray(T_true))
            RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
            TE = float(
                np.linalg.norm(
                    np.asarray(fa_paper_T, dtype=np.float64)[:2, 3] - np.asarray(T_true, dtype=np.float64)[:2, 3]
                )
            )
            rec["freealign_paper"].update({"RE_deg": float(RE), "TE_m": float(TE)})
        else:
            rec["freealign_paper"].update({"RE_deg": None, "TE_m": None})

        # FreeAlign (released repo matching)
        fa_repo_T, fa_repo_stability, fa_repo_matches, fa_repo_meta = freealign_repo_est.estimate(
            cav_boxes=infra_boxes, ego_boxes=veh_boxes
        )
        rec["freealign_repo"] = {
            "success": bool(fa_repo_T is not None),
            "stability": float(fa_repo_stability),
            "matches": int(fa_repo_matches),
            "meta": fa_repo_meta or {},
        }
        if fa_repo_T is not None and T_true is not None:
            yaw_est = _yaw_deg_from_T(np.asarray(fa_repo_T))
            yaw_true = _yaw_deg_from_T(np.asarray(T_true))
            RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
            TE = float(
                np.linalg.norm(
                    np.asarray(fa_repo_T, dtype=np.float64)[:2, 3] - np.asarray(T_true, dtype=np.float64)[:2, 3]
                )
            )
            rec["freealign_repo"].update({"RE_deg": float(RE), "TE_m": float(TE)})
        else:
            rec["freealign_repo"].update({"RE_deg": None, "TE_m": None})

        # V2X-Reg++
        v2x = v2xregpp_est.estimate(infra_boxes, veh_boxes, init=None, ctx=None)
        v2x_matches = int(len(v2x.matches or []))
        v2x_stability = float(v2x.stability or 0.0)
        v2x_ok = bool(v2x.success and v2x.T is not None)
        if v2x_ok:
            if v2x_matches < int(args.v2xregpp_min_matches):
                v2x_ok = False
            if v2x_stability < float(args.v2xregpp_min_stability):
                v2x_ok = False

        rec["v2xregpp"] = {"success": bool(v2x_ok), "stability": float(v2x_stability), "matches": int(v2x_matches)}
        if v2x_ok and v2x.T is not None and T_true is not None:
            yaw_est = _yaw_deg_from_T(np.asarray(v2x.T))
            yaw_true = _yaw_deg_from_T(np.asarray(T_true))
            RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
            TE = float(np.linalg.norm(np.asarray(v2x.T, dtype=np.float64)[:2, 3] - np.asarray(T_true, dtype=np.float64)[:2, 3]))
            rec["v2xregpp"].update({"RE_deg": float(RE), "TE_m": float(TE)})
        else:
            rec["v2xregpp"].update({"RE_deg": None, "TE_m": None})

        per_sample.append(rec)

    def _method_summary(method: str) -> Dict[str, Any]:
        total = len(per_sample)
        succ = [r for r in per_sample if (r.get(method) or {}).get("TE_m") is not None]
        te = [float(r[method]["TE_m"]) for r in succ]
        re = [float(r[method]["RE_deg"]) for r in succ]
        out: Dict[str, Any] = {
            "total": int(total),
            "success_rate": float(len(succ) / max(total, 1)),
            "TE_m": _summ(te),
            "RE_deg": _summ(re),
        }
        thr_mat = {}
        for re_thr in re_thrs:
            row = {}
            for te_thr in te_thrs:
                good = 0
                for r in per_sample:
                    m = r.get(method) or {}
                    te_v = m.get("TE_m")
                    re_v = m.get("RE_deg")
                    if te_v is None or re_v is None:
                        continue
                    if float(te_v) <= float(te_thr) and float(re_v) <= float(re_thr):
                        good += 1
                row[str(te_thr)] = float(good / max(total, 1))
            thr_mat[str(re_thr)] = row
        out["good_rate"] = thr_mat
        return out

    summary = {
        "freealign_paper": _method_summary("freealign_paper"),
        "freealign_repo": _method_summary("freealign_repo"),
        "v2xregpp": _method_summary("v2xregpp"),
        "config": {
            "freealign_paper": freealign_paper_cfg.__dict__,
            "freealign_repo": freealign_repo_cfg.__dict__,
            "v2xregpp": {"config_path": str(resolve_repo_path(args.v2xregpp_config)), "matching_overrides": overrides or {}},
        },
        "inputs": {
            "stage1_cache": None if not args.stage1_cache else str(resolve_repo_path(args.stage1_cache)),
            "veh_stage1": args.veh_stage1,
            "infra_stage1": args.infra_stage1,
        },
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "records": per_sample}, indent=2))


if __name__ == "__main__":
    main()
