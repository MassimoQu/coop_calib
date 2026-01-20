#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _as_int(frame_id: Any) -> Optional[int]:
    if frame_id is None:
        return None
    try:
        return int(str(frame_id))
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj


def _build_sync_map(sync_stage1: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[int, int]]:
    veh_to_infra: Dict[str, str] = {}
    infra_frames = []
    for entry in sync_stage1.values():
        if not isinstance(entry, dict):
            continue
        veh = entry.get("veh_frame_id")
        infra = entry.get("infra_frame_id")
        if veh is None or infra is None:
            continue
        veh_str = str(veh)
        infra_str = str(infra)
        veh_to_infra.setdefault(veh_str, infra_str)
        infra_int = _as_int(infra)
        if infra_int is not None:
            infra_frames.append(infra_int)
    infra_frames_sorted = sorted(set(infra_frames))
    infra_index = {frame: idx for idx, frame in enumerate(infra_frames_sorted)}
    return veh_to_infra, infra_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay_records", type=str, required=True, help="Path to compare_freealign_* output json")
    ap.add_argument("--sync_stage1", type=str, required=True, help="Stage1 cache for 0ms (sync) data")
    ap.add_argument("--method", type=str, default="freealign_paper", choices=["freealign_paper", "freealign_repo"])
    args = ap.parse_args()

    delay_obj = _load_json(Path(args.delay_records))
    records = delay_obj.get("records")
    if not isinstance(records, list):
        raise ValueError("delay_records must contain a 'records' list")

    sync_stage1 = _load_json(Path(args.sync_stage1))
    veh_to_infra_sync, infra_index = _build_sync_map(sync_stage1)

    total = 0
    matched = 0
    exact = 0
    within1 = 0
    missing_meta = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        veh = rec.get("veh_frame_id")
        infra_delay = rec.get("infra_frame_id")
        if veh is None or infra_delay is None:
            continue
        infra_sync = veh_to_infra_sync.get(str(veh))
        if infra_sync is None:
            continue
        infra_sync_int = _as_int(infra_sync)
        infra_delay_int = _as_int(infra_delay)
        if infra_sync_int is None or infra_delay_int is None:
            continue
        if infra_sync_int not in infra_index or infra_delay_int not in infra_index:
            continue
        true_offset = infra_index[infra_sync_int] - infra_index[infra_delay_int]
        meta = (rec.get(args.method) or {}).get("meta") or {}
        pred_offset = meta.get("time_offset_steps")
        if pred_offset is None:
            missing_meta += 1
            continue
        total += 1
        if int(true_offset) == int(pred_offset):
            exact += 1
        if abs(int(true_offset) - int(pred_offset)) <= 1:
            within1 += 1
        matched += 1

    if total <= 0:
        raise SystemExit("No valid samples for temporal alignment evaluation.")

    out = {
        "total": int(total),
        "exact_acc": float(exact / total),
        "within1_acc": float(within1 / total),
        "missing_meta": int(missing_meta),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
