#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import pose_to_tfm


def _wrap_angle_deg(angle_deg: float) -> float:
    return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)


def _angle_abs_deg(angle_deg: float) -> float:
    return abs(_wrap_angle_deg(angle_deg))


def _yaw_from_T(T: np.ndarray) -> float:
    # SE(2) yaw from the XY plane rotation.
    return float(math.degrees(math.atan2(float(T[1, 0]), float(T[0, 0]))))


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
class RelPoseMetrics:
    pairs: int
    success: int
    failures: int
    te_m: Dict[str, Optional[float]]
    re_deg: Dict[str, Optional[float]]
    success_at_m: Dict[str, Optional[float]]


def _compute_rel_pose_metrics(errors: List[Tuple[Optional[float], Optional[float]]]) -> RelPoseMetrics:
    pairs = int(len(errors))
    te_values: List[float] = []
    re_values: List[float] = []
    failures = 0
    for te, re in errors:
        if te is None or re is None:
            failures += 1
            continue
        te_values.append(float(te))
        re_values.append(float(re))

    success = pairs - failures
    thresholds = [1.0, 2.0, 3.0, 5.0, 10.0]
    success_at: Dict[str, Optional[float]] = {}
    if te_values:
        arr = np.asarray(te_values, dtype=np.float64)
        for thr in thresholds:
            key = str(int(thr)) if float(thr).is_integer() else str(thr)
            success_at[key] = float(np.mean(arr < float(thr)))
    else:
        for thr in thresholds:
            key = str(int(thr)) if float(thr).is_integer() else str(thr)
            success_at[key] = None

    return RelPoseMetrics(
        pairs=pairs,
        success=int(success),
        failures=int(failures),
        te_m=_summ(te_values),
        re_deg=_summ(re_values),
        success_at_m=success_at,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PGC pose JSON vs GT poses (relative ego->cav).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hypes_yaml", "-y", type=str, help="HEAL/OpenCOOD yaml for dataset paths.")
    g.add_argument("--model_dir", type=str, help="HEAL checkpoint folder containing config.yaml (dataset).")

    p.add_argument("--pgc_json", type=str, required=True, help="PGC pose json (from infer_v2vloc_pgc_pose.py).")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max_samples", type=int, default=0, help="Limit number of frames (0=all).")
    p.add_argument("--out", type=str, default="", help="Optional JSON output path for the metrics report.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

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

    pgc_path = Path(args.pgc_json)
    with pgc_path.open("r", encoding="utf-8") as f:
        pose_result = json.load(f)
    if not isinstance(pose_result, dict):
        raise SystemExit(f"Expected dict JSON at {pgc_path}")

    errors: List[Tuple[Optional[float], Optional[float]]] = []
    for idx in range(int(max_samples)):
        key = str(idx)
        entry = pose_result.get(key)
        if entry is None or not isinstance(entry, Mapping):
            # No prediction for this index.
            base = ds.retrieve_base_data(int(idx))
            if len(base) >= 2:
                errors.append((None, None))
            continue

        cav_ids_pred = entry.get("cav_id_list") or []
        poses_pred = entry.get("lidar_pose_pred_np") or []
        if not isinstance(cav_ids_pred, list) or not isinstance(poses_pred, list):
            continue
        cav_ids_pred = [str(x) for x in cav_ids_pred]

        base = ds.retrieve_base_data(int(idx))
        cav_ids = [str(k) for k in base.keys()]
        if len(cav_ids) < 2:
            continue
        ego_id = cav_ids[0]

        # Map predictions by cav_id.
        pred_by_id: Dict[str, np.ndarray] = {}
        for j, cid in enumerate(cav_ids_pred):
            if j < 0 or j >= len(poses_pred):
                continue
            try:
                pose6 = np.asarray(poses_pred[j], dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if pose6.size == 6:
                pred_by_id[str(cid)] = pose6
            elif pose6.size == 3:
                full = np.zeros((6,), dtype=np.float64)
                full[[0, 1, 4]] = pose6
                pred_by_id[str(cid)] = full

        if ego_id not in pred_by_id:
            # If ego pred missing, all relative errors in this frame are invalid.
            for _ in cav_ids[1:]:
                errors.append((None, None))
            continue

        # Predicted world transforms.
        ego_T_world_pred = pose_to_tfm(np.asarray([pred_by_id[ego_id]], dtype=np.float64))[0]

        # GT clean world transforms (from dataset base_data).
        ego_pose_gt = np.asarray(base[ego_id]["params"]["lidar_pose"], dtype=np.float64).reshape(6)
        ego_T_world_gt = pose_to_tfm(np.asarray([ego_pose_gt], dtype=np.float64))[0]

        for cav_id in cav_ids[1:]:
            if cav_id not in pred_by_id:
                errors.append((None, None))
                continue

            cav_pose_gt = np.asarray(base[cav_id]["params"]["lidar_pose"], dtype=np.float64).reshape(6)
            cav_T_world_gt = pose_to_tfm(np.asarray([cav_pose_gt], dtype=np.float64))[0]

            cav_T_world_pred = pose_to_tfm(np.asarray([pred_by_id[cav_id]], dtype=np.float64))[0]

            rel_gt = np.linalg.inv(ego_T_world_gt) @ cav_T_world_gt
            rel_pred = np.linalg.inv(ego_T_world_pred) @ cav_T_world_pred
            err = np.linalg.inv(rel_gt) @ rel_pred
            te = float(np.linalg.norm(err[:2, 3]))
            re = _angle_abs_deg(_yaw_from_T(err))
            errors.append((te, re))

        if (idx + 1) % 200 == 0:
            print(f"[eval-pgc] {idx+1}/{int(max_samples)}")

    metrics = _compute_rel_pose_metrics(errors)
    report = {
        "split": str(args.split),
        "max_samples": int(max_samples),
        "pgc_json": str(pgc_path),
        "rel_pose": asdict(metrics),
    }
    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval-pgc] saved: {out_path}")


if __name__ == "__main__":
    main()

