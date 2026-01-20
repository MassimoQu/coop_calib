#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot pose-noise sweep curves produced by `opencood/tools/inference_w_noise.py`.

Expected YAML schema (see inference_w_noise.py):
  - pos_std_list / rot_std_list: list[float]
  - ap30 / ap50 / ap70: list[float]
  - pose_correction: str
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple


def _load_yaml(path: str) -> Dict[str, Any]:
    # Keep dependency minimal: PyYAML is already a transitive dep of OpenCOOD.
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML content (not a dict): {path}")
    return data


def _as_float_list(obj: Any) -> List[float]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out: List[float] = []
        for v in obj:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def _label_for(path: str, data: Dict[str, Any]) -> str:
    label = str(data.get("pose_correction") or "").strip()
    if label:
        return label
    return os.path.splitext(os.path.basename(path))[0]


def _validate_lengths(path: str, x: List[float], y: List[float], key: str) -> None:
    if not x or not y:
        raise ValueError(f"Missing data in {path}: x={len(x)} y({key})={len(y)}")
    if len(x) != len(y):
        raise ValueError(f"Length mismatch in {path}: len(x)={len(x)} len({key})={len(y)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yamls",
        nargs="+",
        required=True,
        help="One or more AP030507_*.yaml files (output of inference_w_noise.py).",
    )
    parser.add_argument("--out", type=str, required=True, help="Output image path (png/pdf).")
    parser.add_argument("--title", type=str, default="", help="Optional figure title.")
    parser.add_argument(
        "--metric",
        choices=["ap30", "ap50", "ap70", "all"],
        default="all",
        help="Which metric(s) to plot.",
    )
    parser.add_argument(
        "--scale",
        choices=["raw", "percent"],
        default="raw",
        help="raw: keep AP in [0,1]; percent: multiply by 100.",
    )
    parser.add_argument("--xlabel", type=str, default="pos_std (m) / rot_std (deg)")
    parser.add_argument("--ylabel", type=str, default="AP")
    args = parser.parse_args()

    series: List[Tuple[str, List[float], Dict[str, List[float]]]] = []
    x_ref: List[float] = []

    for p in args.yamls:
        data = _load_yaml(p)
        x = _as_float_list(data.get("pos_std_list"))
        ap30 = _as_float_list(data.get("ap30"))
        ap50 = _as_float_list(data.get("ap50"))
        ap70 = _as_float_list(data.get("ap70"))
        for k, y in [("ap30", ap30), ("ap50", ap50), ("ap70", ap70)]:
            if y:
                _validate_lengths(p, x, y, k)
        if not x_ref:
            x_ref = x
        elif x != x_ref:
            raise ValueError(
                "All YAMLs must share the same pos_std_list for a fair plot.\n"
                f"ref={x_ref}\n"
                f"{p}={x}"
            )
        label = _label_for(p, data)
        series.append((label, x, {"ap30": ap30, "ap50": ap50, "ap70": ap70}))

    import matplotlib.pyplot as plt

    metrics = ["ap30", "ap50", "ap70"] if args.metric == "all" else [args.metric]
    scale = 100.0 if args.scale == "percent" else 1.0

    if len(metrics) == 1:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for label, x, y_map in series:
            y = [v * scale for v in y_map[metrics[0]]]
            ax.plot(x, y, marker="o", linewidth=2.0, label=label)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel + (" (%)" if args.scale == "percent" else ""))
        ax.grid(True, linestyle="--", alpha=0.4)
        if args.title:
            ax.set_title(args.title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    for ax, metric in zip(axes, metrics):
        for label, x, y_map in series:
            y = [v * scale for v in y_map[metric]]
            ax.plot(x, y, marker="o", linewidth=2.0, label=label)
        ax.set_title(metric.upper())
        ax.set_xlabel(args.xlabel)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel(args.ylabel + (" (%)" if args.scale == "percent" else ""))
    axes[-1].legend()
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()

