#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-invasive dataset split helpers for HEAL/OpenCOOD.

Why this exists:
- V2V4Real (OPV2V-format) can contain accidental scenario duplication across
  train/test folders (data leakage).
- DAIR-V2X-C official train/val splits are *frame-level* and may mix the same
  short driving clip ("batch") across splits, which is problematic for
  localization-style evaluation/training.
- OPV2V splits are folder-based but it is convenient to export explicit scenario
  lists for reproducibility and filtering.

All outputs are written under a user-provided out_dir (no modification to the
dataset on disk).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def _list_subdirs(path: Path) -> List[str]:
    if not path.is_dir():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate non-invasive split files for HEAL/OpenCOOD datasets.")
    # NOTE: `required=True` for subparsers is only available in newer Python.
    sub = p.add_subparsers(dest="dataset")

    p_v2v4 = sub.add_parser("v2v4real", help="Export scenario lists for V2V4Real and optionally de-duplicate train/test.")
    p_v2v4.add_argument("--data_dir", type=str, required=True, help="Path to V2V4Real root containing train/validate/test.")
    p_v2v4.add_argument("--out_dir", type=str, required=True, help="Output folder (repo-local or absolute).")
    p_v2v4.add_argument("--no_dedup", action="store_true", help="Do not remove overlapping scenario names across splits.")
    p_v2v4.add_argument(
        "--dedup_keep",
        type=str,
        default="test",
        choices=["test", "train"],
        help="When a scenario name appears in both train and test, keep it in this split and drop from the other.",
    )

    p_opv2v = sub.add_parser("opv2v", help="Export scenario lists for OPV2V.")
    p_opv2v.add_argument("--data_dir", type=str, required=True, help="Path to OPV2V root containing train/validate/test.")
    p_opv2v.add_argument("--out_dir", type=str, required=True, help="Output folder (repo-local or absolute).")

    p_dair = sub.add_parser("dair", help="Generate traversal-safe splits for DAIR-V2X-C based on intersection + batch_id.")
    p_dair.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to DAIR cooperative-vehicle-infrastructure root (contains vehicle-side/, infrastructure-side/, cooperative/).",
    )
    p_dair.add_argument("--out_dir", type=str, required=True, help="Output folder (repo-local or absolute).")
    p_dair.add_argument("--seed", type=int, default=303)
    p_dair.add_argument("--val_ratio", type=float, default=0.2)
    p_dair.add_argument("--test_ratio", type=float, default=0.0, help="Optional test split ratio (default: 0, no test).")
    p_dair.add_argument(
        "--group_by",
        type=str,
        default="intersection_batch",
        choices=["intersection_batch", "intersection", "batch"],
        help="Grouping granularity. Use intersection_batch for traversal-safe localization-style splits.",
    )
    p_dair.add_argument(
        "--min_group_size",
        type=int,
        default=1,
        help="Drop groups with fewer than this many cooperative pairs.",
    )
    p_dair.add_argument(
        "--keep_unknown_intersection",
        action="store_true",
        help="Keep samples with empty/unknown intersection_loc (otherwise dropped).",
    )

    args = p.parse_args()
    if not getattr(args, "dataset", None):
        p.print_help()
        raise SystemExit(2)
    return args


def _write_scenario_lists(
    *,
    dataset_name: str,
    data_dir: Path,
    out_dir: Path,
    dedup: bool = False,
    dedup_keep: str = "test",
) -> None:
    split_dirs = {
        "train": data_dir / "train",
        "val": data_dir / "validate",
        "test": data_dir / "test",
    }
    scenarios = {k: _list_subdirs(v) for k, v in split_dirs.items()}

    report = {
        "dataset": dataset_name,
        "data_dir": str(data_dir),
        "counts": {k: len(v) for k, v in scenarios.items()},
        "overlap_train_test": [],
        "dedup": bool(dedup),
        "dedup_keep": str(dedup_keep),
    }

    if dedup:
        train_set = set(scenarios["train"])
        test_set = set(scenarios["test"])
        overlap = sorted(train_set & test_set)
        report["overlap_train_test"] = overlap

        if overlap:
            if dedup_keep == "test":
                scenarios["train"] = [s for s in scenarios["train"] if s not in overlap]
            else:
                scenarios["test"] = [s for s in scenarios["test"] if s not in overlap]

    _dump_json(out_dir / "train_scenarios.json", scenarios["train"])
    _dump_json(out_dir / "val_scenarios.json", scenarios["val"])
    _dump_json(out_dir / "test_scenarios.json", scenarios["test"])
    _dump_json(out_dir / "report.json", report)

    print(f"[{dataset_name}] wrote scenario lists to: {out_dir}")
    if report["overlap_train_test"]:
        print(f"[{dataset_name}] train/test overlap scenarios: {len(report['overlap_train_test'])}")


def _id_from_path(p: str) -> Optional[str]:
    m = re.search(r"/(\d+)\.(?:pcd|jpg)$", str(p))
    return m.group(1) if m else None


def _build_dair_maps(data_dir: Path) -> Tuple[Dict[str, dict], Dict[str, dict], List[dict]]:
    veh_meta = json.loads((data_dir / "vehicle-side" / "data_info.json").read_text(encoding="utf-8"))
    inf_meta = json.loads((data_dir / "infrastructure-side" / "data_info.json").read_text(encoding="utf-8"))
    coop_meta = json.loads((data_dir / "cooperative" / "data_info.json").read_text(encoding="utf-8"))

    veh_map = {}
    for d in veh_meta:
        vid = _id_from_path(d.get("pointcloud_path", ""))
        if vid is not None:
            veh_map[str(vid)] = d

    inf_map = {}
    for d in inf_meta:
        iid = _id_from_path(d.get("pointcloud_path", ""))
        if iid is not None:
            inf_map[str(iid)] = d

    return veh_map, inf_map, coop_meta


def _group_key_for_dair(
    *,
    group_by: str,
    intersection_loc: str,
    batch_id: Optional[int],
) -> Tuple:
    if group_by == "intersection":
        return (intersection_loc,)
    if group_by == "batch":
        return (int(batch_id) if batch_id is not None else -1,)
    # intersection_batch
    return (intersection_loc, int(batch_id) if batch_id is not None else -1)


def _make_dair_splits(
    *,
    data_dir: Path,
    out_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    group_by: str,
    min_group_size: int,
    keep_unknown_intersection: bool,
) -> None:
    random.seed(int(seed))

    veh_map, inf_map, coop_meta = _build_dair_maps(data_dir)

    records = []
    dropped_unknown = 0
    dropped_missing = 0
    for frame_info in coop_meta:
        veh_frame_id = str(Path(frame_info["vehicle_image_path"]).stem)
        inf_frame_id = str(Path(frame_info["infrastructure_image_path"]).stem)
        veh_meta = veh_map.get(veh_frame_id)
        inf_meta = inf_map.get(inf_frame_id)
        if inf_meta is None:
            dropped_missing += 1
            continue

        intersection_loc = str(inf_meta.get("intersection_loc") or "").strip()
        if not intersection_loc:
            intersection_loc = "?"
        if intersection_loc == "?" and not keep_unknown_intersection:
            dropped_unknown += 1
            continue

        batch_id = None
        ts = None
        if veh_meta is not None:
            try:
                batch_id = int(veh_meta.get("batch_id"))
            except Exception:
                batch_id = None
            try:
                ts = int(veh_meta.get("pointcloud_timestamp"))
            except Exception:
                ts = None

        records.append(
            {
                "veh_frame_id": veh_frame_id,
                "inf_frame_id": inf_frame_id,
                "intersection_loc": intersection_loc,
                "batch_id": batch_id,
                "timestamp": ts,
            }
        )

    # Group records
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        gk = _group_key_for_dair(
            group_by=str(group_by),
            intersection_loc=str(r["intersection_loc"]),
            batch_id=r.get("batch_id"),
        )
        groups[gk].append(r)

    # Drop tiny groups
    groups = {k: v for k, v in groups.items() if len(v) >= int(min_group_size)}

    # Assign groups to splits.
    split_of_group: Dict[Tuple, str] = {}

    if group_by == "intersection_batch":
        # For each intersection, split its batch_id groups to train/val/test.
        per_loc_batches: Dict[str, List[int]] = defaultdict(list)
        for (loc, bid) in groups.keys():
            per_loc_batches[str(loc)].append(int(bid))
        for loc, bids in per_loc_batches.items():
            uniq = sorted(set(bids))
            random.shuffle(uniq)
            n = len(uniq)
            n_test = int(round(float(n) * float(test_ratio)))
            n_val = int(round(float(n) * float(val_ratio)))
            n_test = max(0, min(n, n_test))
            n_val = max(0, min(n - n_test, n_val))
            test_b = set(uniq[:n_test])
            val_b = set(uniq[n_test : n_test + n_val])
            for bid in uniq:
                if bid in test_b:
                    split_of_group[(loc, bid)] = "test"
                elif bid in val_b:
                    split_of_group[(loc, bid)] = "val"
                else:
                    split_of_group[(loc, bid)] = "train"
    elif group_by == "intersection":
        locs = sorted({k[0] for k in groups.keys()})
        random.shuffle(locs)
        n = len(locs)
        n_test = int(round(float(n) * float(test_ratio)))
        n_val = int(round(float(n) * float(val_ratio)))
        n_test = max(0, min(n, n_test))
        n_val = max(0, min(n - n_test, n_val))
        test_l = set(locs[:n_test])
        val_l = set(locs[n_test : n_test + n_val])
        for loc in locs:
            if loc in test_l:
                split_of_group[(loc,)] = "test"
            elif loc in val_l:
                split_of_group[(loc,)] = "val"
            else:
                split_of_group[(loc,)] = "train"
    else:  # batch
        bids = sorted({k[0] for k in groups.keys()})
        random.shuffle(bids)
        n = len(bids)
        n_test = int(round(float(n) * float(test_ratio)))
        n_val = int(round(float(n) * float(val_ratio)))
        n_test = max(0, min(n, n_test))
        n_val = max(0, min(n - n_test, n_val))
        test_b = set(bids[:n_test])
        val_b = set(bids[n_test : n_test + n_val])
        for bid in bids:
            if bid in test_b:
                split_of_group[(bid,)] = "test"
            elif bid in val_b:
                split_of_group[(bid,)] = "val"
            else:
                split_of_group[(bid,)] = "train"

    split_records: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for gk, rs in groups.items():
        split = split_of_group.get(gk)
        if split is None:
            # Should not happen, but keep safe.
            split = "train"
        split_records[split].extend(rs)

    # Sort deterministically for reproducible indexing (important for PGC outputs keyed by idx).
    def _sort_key(r: dict) -> Tuple:
        loc = str(r.get("intersection_loc") or "?")
        bid = int(r.get("batch_id") if r.get("batch_id") is not None else -1)
        ts = int(r.get("timestamp") if r.get("timestamp") is not None else -1)
        return (loc, bid, ts, str(r.get("veh_frame_id")), str(r.get("inf_frame_id")))

    for k in split_records.keys():
        split_records[k] = sorted(split_records[k], key=_sort_key)

    # Only dump veh/inf ids + small debug metadata; DAIR loader ignores extra keys.
    train_out = split_records["train"]
    val_out = split_records["val"]
    test_out = split_records["test"] if test_ratio > 0 else []

    _dump_json(out_dir / "train.json", train_out)
    _dump_json(out_dir / "val.json", val_out)
    if test_ratio > 0:
        _dump_json(out_dir / "test.json", test_out)

    # Write a short report.
    per_loc = defaultdict(lambda: defaultdict(int))
    for split, rs in split_records.items():
        for r in rs:
            per_loc[str(r.get("intersection_loc") or "?")][split] += 1

    report = {
        "dataset": "dair",
        "data_dir": str(data_dir),
        "seed": int(seed),
        "group_by": str(group_by),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "min_group_size": int(min_group_size),
        "dropped_unknown_intersection": int(dropped_unknown),
        "dropped_missing_infra_meta": int(dropped_missing),
        "num_records_raw": int(len(records)),
        "num_records_kept": int(sum(len(v) for v in groups.values())),
        "num_groups": int(len(groups)),
        "counts": {k: int(len(v)) for k, v in split_records.items()},
        "per_intersection_counts": {k: dict(v) for k, v in per_loc.items()},
    }
    _dump_json(out_dir / "report.json", report)

    print(f"[dair] wrote splits to: {out_dir}")
    print(f"[dair] counts: train={len(train_out)} val={len(val_out)} test={len(test_out)}")


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.dataset == "v2v4real":
        _write_scenario_lists(
            dataset_name="v2v4real",
            data_dir=data_dir,
            out_dir=out_dir,
            dedup=not bool(args.no_dedup),
            dedup_keep=str(args.dedup_keep),
        )
        return

    if args.dataset == "opv2v":
        _write_scenario_lists(
            dataset_name="opv2v",
            data_dir=data_dir,
            out_dir=out_dir,
            dedup=False,
        )
        return

    if args.dataset == "dair":
        _make_dair_splits(
            data_dir=data_dir,
            out_dir=out_dir,
            seed=int(args.seed),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            group_by=str(args.group_by),
            min_group_size=int(args.min_group_size),
            keep_unknown_intersection=bool(args.keep_unknown_intersection),
        )
        return

    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
