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


def _build_sync_map(sync_stage1: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict[int, int]]]:
    veh_to_infra: Dict[str, str] = {}
    infra_frames = []
    seq_frames: Dict[str, list[int]] = {}
    for entry in sync_stage1.values():
        if not isinstance(entry, dict):
            continue
        veh = entry.get("veh_frame_id")
        infra = entry.get("infra_frame_id")
        if veh is None or infra is None:
            frame_id = entry.get("frame_id")
            if frame_id is None:
                continue
            frame_int = _as_int(frame_id)
            if frame_int is None:
                continue
            seq_id = entry.get("sequence_id")
            seq_key = str(seq_id) if seq_id is not None else "default"
            seq_frames.setdefault(seq_key, []).append(int(frame_int))
            continue
        veh_str = str(veh)
        infra_str = str(infra)
        veh_to_infra.setdefault(veh_str, infra_str)
        infra_int = _as_int(infra)
        if infra_int is not None:
            infra_frames.append(infra_int)
    infra_frames_sorted = sorted(set(infra_frames))
    infra_index = {frame: idx for idx, frame in enumerate(infra_frames_sorted)}
    seq_index: Dict[str, Dict[int, int]] = {}
    for seq_key, frames in seq_frames.items():
        frames_sorted = sorted(set(frames))
        seq_index[str(seq_key)] = {frame: idx for idx, frame in enumerate(frames_sorted)}
    return veh_to_infra, {"infra": infra_index, "seq": seq_index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay_records", type=str, required=True, help="Path to compare_freealign_* output json")
    ap.add_argument("--sync_stage1", type=str, required=True, help="Stage1 cache for 0ms (sync) data")
    ap.add_argument("--method", type=str, default="freealign_paper", choices=["freealign_paper", "freealign_repo"])
    ap.add_argument(
        "--sample_interval_ms",
        type=float,
        default=100.0,
        help="Frame interval in milliseconds for converting offset steps to time.",
    )
    args = ap.parse_args()

    delay_obj = _load_json(Path(args.delay_records))
    records = delay_obj.get("records")
    if not isinstance(records, list):
        raise ValueError("delay_records must contain a 'records' list")

    sync_stage1 = _load_json(Path(args.sync_stage1))
    veh_to_infra_sync, sync_index = _build_sync_map(sync_stage1)
    infra_index = sync_index.get("infra", {})
    seq_index = sync_index.get("seq", {})

    total = 0
    matched = 0
    exact = 0
    within1 = 0
    missing_meta = 0
    abs_offsets = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        veh = rec.get("veh_frame_id")
        infra_delay = rec.get("infra_frame_id")
        true_offset = None
        if veh is not None and infra_delay is not None and veh_to_infra_sync:
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
        else:
            seq_key = str(rec.get("sequence_id")) if rec.get("sequence_id") is not None else "default"
            seq_map = seq_index.get(seq_key) or seq_index.get("default")
            if not seq_map:
                continue
            sync_frame_int = _as_int(rec.get("sync_frame_id"))
            delay_frame_int = _as_int(rec.get("delay_frame_id"))
            if sync_frame_int is not None and delay_frame_int is not None:
                if sync_frame_int not in seq_map or delay_frame_int not in seq_map:
                    continue
                true_offset = seq_map[sync_frame_int] - seq_map[delay_frame_int]
            else:
                frame_int = _as_int(rec.get("frame_id"))
                if frame_int is None:
                    frame_int = delay_frame_int
                if frame_int is None:
                    continue
                if frame_int not in seq_map:
                    continue
                true_offset = 0
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
        abs_offsets.append(abs(int(true_offset) - int(pred_offset)))
        matched += 1

    if total <= 0:
        raise SystemExit("No valid samples for temporal alignment evaluation.")

    avg_abs_steps = float(sum(abs_offsets) / total) if total else 0.0
    out = {
        "total": int(total),
        "exact_acc": float(exact / total),
        "within1_acc": float(within1 / total),
        "missing_meta": int(missing_meta),
        "avg_abs_offset_steps": avg_abs_steps,
        "avg_abs_offset_ms": float(avg_abs_steps * float(args.sample_interval_ms)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
