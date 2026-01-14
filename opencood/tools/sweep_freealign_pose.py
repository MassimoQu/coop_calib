#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator
from opencood.tools.eval_freealign_pose import _dair_T_infra_to_vehicle


def _parse_csv(values: str, *, cast=float) -> List[Any]:
    raw = [v.strip() for v in str(values or "").split(",") if v.strip()]
    if not raw:
        return []
    if cast is str:
        return raw
    return [cast(v) for v in raw]


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


def evaluate(
    *,
    veh_data: Dict[str, Any],
    infra_data: Dict[str, Any],
    dair_root: Optional[Path],
    cfg: FreeAlignPaperConfig,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    estimator = FreeAlignPaperEstimator(cfg)

    keys = sorted(set(veh_data.keys()) & set(infra_data.keys()), key=lambda x: int(x))
    if max_samples:
        keys = keys[: int(max_samples)]

    failures = 0
    re_list: List[float] = []
    te_list: List[float] = []
    matches_list: List[int] = []

    for key in keys:
        veh_entry = veh_data.get(key) or {}
        infra_entry = infra_data.get(key) or {}
        veh_corners, veh_scores = _extract_single_agent(veh_entry)
        infra_corners, infra_scores = _extract_single_agent(infra_entry)
        veh_boxes = corners_to_bbox3d_list(veh_corners, bbox_type="detected", scores=veh_scores)
        infra_boxes = corners_to_bbox3d_list(infra_corners, bbox_type="detected", scores=infra_scores)

        T_true = None
        if dair_root is not None:
            try:
                T_true = _dair_T_infra_to_vehicle(
                    dair_root,
                    str(infra_entry.get("infra_frame_id")),
                    str(veh_entry.get("veh_frame_id")),
                )
            except Exception:
                T_true = None

        T_est, _, matches, _ = estimator.estimate(cav_boxes=infra_boxes, ego_boxes=veh_boxes, T_init=None)
        if T_est is None or T_true is None:
            failures += 1
            continue

        yaw_est = _yaw_deg_from_T(np.asarray(T_est))
        yaw_true = _yaw_deg_from_T(np.asarray(T_true))
        RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
        t_est = np.asarray(T_est, dtype=np.float64)[0:2, 3]
        t_true = np.asarray(T_true, dtype=np.float64)[0:2, 3]
        TE = float(np.linalg.norm(t_est - t_true))
        re_list.append(float(RE))
        te_list.append(float(TE))
        matches_list.append(int(matches))

    return {
        "total": int(len(keys)),
        "failures": int(failures),
        "success_rate": float((len(keys) - failures) / max(len(keys), 1)),
        "RE_deg": _summ(re_list),
        "TE_m": _summ(te_list),
        "matches": _summ([float(x) for x in matches_list]),
        "config": cfg.__dict__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veh_stage1", type=str, required=True)
    ap.add_argument("--infra_stage1", type=str, required=True)
    ap.add_argument("--dair_root", type=str, default="~/datasets/data2/DAIR-V2X-C/cooperative-vehicle-infrastructure")
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--seed_strategy", type=str, default="all", help="CSV: all,topk_radius")
    ap.add_argument("--min_nodes", type=str, default="5,6", help="CSV ints")
    ap.add_argument("--sim_threshold", type=str, default="0.8,1.0,1.2", help="CSV floats")
    ap.add_argument("--anchor_max_count", type=str, default="4,6,8", help="CSV ints")
    ap.add_argument("--anchor_topk", type=str, default="10", help="CSV ints")
    ap.add_argument("--selection_mode", type=str, default="max_size_then_min_eps", help="CSV")
    ap.add_argument("--max_boxes", type=str, default="60", help="CSV ints")
    ap.add_argument("--p", type=str, default="1.0", help="CSV floats")

    ap.add_argument("--use_gnn", action="store_true")
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    veh_path = resolve_repo_path(args.veh_stage1)
    infra_path = resolve_repo_path(args.infra_stage1)
    veh_data = _load_stage1(veh_path)
    infra_data = _load_stage1(infra_path)
    dair_root = Path(str(args.dair_root)).expanduser().resolve() if str(args.dair_root).strip() else None

    grid = {
        "seed_strategy": _parse_csv(args.seed_strategy, cast=str),
        "min_nodes": _parse_csv(args.min_nodes, cast=int),
        "sim_threshold": _parse_csv(args.sim_threshold, cast=float),
        "anchor_max_count": _parse_csv(args.anchor_max_count, cast=int),
        "anchor_topk": _parse_csv(args.anchor_topk, cast=int),
        "selection_mode": _parse_csv(args.selection_mode, cast=str),
        "max_boxes": _parse_csv(args.max_boxes, cast=int),
        "p": _parse_csv(args.p, cast=float),
    }
    for k, v in list(grid.items()):
        if not v:
            raise SystemExit(f"Empty sweep list for {k}.")

    results = []
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        cfg = FreeAlignPaperConfig(
            ckpt_path=None if not args.ckpt_path else str(resolve_repo_path(args.ckpt_path)),
            device=str(args.device),
            use_gnn=bool(args.use_gnn),
            max_boxes=int(params["max_boxes"]),
            anchor_topk=int(params["anchor_topk"]),
            seed_strategy=str(params["seed_strategy"]),
            min_nodes=int(params["min_nodes"]),
            anchor_max_count=int(params["anchor_max_count"]),
            sim_threshold=float(params["sim_threshold"]),
            selection_mode=str(params["selection_mode"]),
            p=float(params["p"]),
        )
        summary = evaluate(
            veh_data=veh_data,
            infra_data=infra_data,
            dair_root=dair_root,
            cfg=cfg,
            max_samples=args.max_samples,
        )
        results.append(summary)

    def _score(item: Dict[str, Any]) -> Tuple[float, float]:
        te_med = item.get("TE_m", {}).get("median")
        if te_med is None:
            te_med = float("inf")
        return float(te_med), -float(item.get("success_rate") or 0.0)

    results.sort(key=_score)

    payload = {
        "best": results[0] if results else None,
        "results": results,
        "inputs": {"veh_stage1": str(veh_path), "infra_stage1": str(infra_path)},
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

