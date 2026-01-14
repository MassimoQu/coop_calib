#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

_HEAL_ROOT = Path(__file__).resolve().parents[2]
if str(_HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_HEAL_ROOT))

from opencood.extrinsics.late_fusion import CBMEstimator, VIPSEstimator, V2XRegPPEstimator
from opencood.extrinsics.path_utils import ensure_v2xreg_root_on_path, resolve_repo_path
from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


def _parse_floats(raw: str) -> List[float]:
    values: List[float] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _normalize_cfg_paths(cfg) -> None:
    cfg.data.data_info_path = str(resolve_repo_path(cfg.data.data_info_path))
    cfg.data.data_root = str(resolve_repo_path(cfg.data.data_root))
    if cfg.data.detection_cache:
        cfg.data.detection_cache = str(resolve_repo_path(cfg.data.detection_cache))
    if getattr(cfg.data, "feature_cache", None):
        cfg.data.feature_cache = str(resolve_repo_path(cfg.data.feature_cache))
    cfg.output.root_dir = str(resolve_repo_path(cfg.output.root_dir))


def add_transform_noise(
    T: np.ndarray,
    trans_std: float,
    rot_std_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if trans_std <= 0 and rot_std_deg <= 0:
        return T.copy()
    delta_t = rng.normal(scale=float(trans_std), size=3)
    rot_std_rad = math.radians(float(rot_std_deg))
    delta_euler = rng.normal(scale=rot_std_rad, size=3)
    delta_R = R.from_euler("xyz", delta_euler).as_matrix()
    noise_T = np.eye(4, dtype=np.float64)
    noise_T[:3, :3] = delta_R
    noise_T[:3, 3] = delta_t
    return noise_T @ T


def build_init(
    *,
    mode: str,
    T_true: np.ndarray,
    trans_std: float,
    rot_std_deg: float,
    rng: np.random.Generator,
) -> Optional[ExtrinsicInit]:
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode == "identity":
        return ExtrinsicInit(T_init=np.eye(4, dtype=np.float64), source="identity")
    if mode == "gt":
        return ExtrinsicInit(T_init=np.array(T_true, dtype=np.float64), source="gt")
    if mode == "noisy_gt":
        return ExtrinsicInit(
            T_init=add_transform_noise(T_true, trans_std, rot_std_deg, rng),
            source="noisy_gt",
        )
    raise ValueError(f"Unknown init mode: {mode}")


def select_boxes(sample, *, use_detection: bool):
    if not use_detection:
        return sample.infra_boxes, sample.veh_boxes, "groundtruth"
    infra = sample.detections_infra if sample.detections_infra is not None else []
    veh = sample.detections_vehicle if sample.detections_vehicle is not None else []
    return infra, veh, "detection"


def run_once(
    *,
    method: str,
    config_path: str,
    init_mode: str,
    init_trans_std: float,
    init_rot_std_deg: float,
    seed: int,
    max_samples: Optional[int],
    use_detection: bool,
    v2xregpp_use_prior: bool,
    data_root_override: Optional[str],
    data_info_override: Optional[str],
    detection_cache_override: Optional[str],
    out_dir: Path,
) -> dict:
    ensure_v2xreg_root_on_path()
    from calib.config import load_config  # type: ignore
    from calib.data.dataset_manager import DatasetManager  # type: ignore
    from calib.evaluation.metrics import FrameMetrics, aggregate_metrics  # type: ignore
    from calib.filters.pipeline import FilterPipeline  # type: ignore
    from v2x_calib.utils import (  # type: ignore
        convert_T_to_6DOF,
        get_RE_TE_by_compare_T_6DOF_result_true,
    )

    cfg = load_config(str(resolve_repo_path(config_path)))
    _normalize_cfg_paths(cfg)
    if data_root_override:
        cfg.data.data_root = str(resolve_repo_path(data_root_override))
    if data_info_override:
        cfg.data.data_info_path = str(resolve_repo_path(data_info_override))
    if detection_cache_override is not None:
        cfg.data.detection_cache = (
            None if detection_cache_override == "" else str(resolve_repo_path(detection_cache_override))
        )
    if max_samples is not None:
        cfg.data.max_samples = int(max_samples)
    if use_detection:
        cfg.data.use_detection = True

    thresholds = list(cfg.evaluation.success_thresholds or [])
    mgr = DatasetManager(cfg.data)
    filters = FilterPipeline(cfg.filters) if getattr(cfg, "filters", None) is not None else None

    method = method.lower()
    if method == "v2xregpp":
        matching_overrides = None
        if v2xregpp_use_prior and init_mode.lower() != "none":
            matching_overrides = {"filter_strategy": "trueRetained"}
        estimator = V2XRegPPEstimator(config_path=config_path, matching_overrides=matching_overrides)
    elif method == "cbm":
        estimator = CBMEstimator()
    elif method == "vips":
        estimator = VIPSEstimator()
    else:
        raise ValueError(f"Unknown method: {method}")

    out_dir.mkdir(parents=True, exist_ok=True)
    matches_path = out_dir / "matches.jsonl"

    rng = np.random.default_rng(int(seed))
    records: List[FrameMetrics] = []
    init_RE_values: List[float] = []
    init_TE_values: List[float] = []
    processed = 0

    with matches_path.open("w", encoding="utf-8") as f:
        for sample in mgr.samples():
            infra_boxes, veh_boxes, source = select_boxes(sample, use_detection=cfg.data.use_detection)
            if filters is not None and method in {"vips", "cbm"}:
                infra_boxes, veh_boxes = filters.apply(infra_boxes or [], veh_boxes or [])
            init = build_init(
                mode=init_mode,
                T_true=sample.T_true,
                trans_std=init_trans_std,
                rot_std_deg=init_rot_std_deg,
                rng=rng,
            )
            init_RE = init_TE = None
            if init is not None:
                init_RE, init_TE = get_RE_TE_by_compare_T_6DOF_result_true(
                    convert_T_to_6DOF(init.T_init), convert_T_to_6DOF(sample.T_true)
                )
                if init_RE is not None:
                    init_RE_values.append(float(init_RE))
                if init_TE is not None:
                    init_TE_values.append(float(init_TE))
                init = ExtrinsicInit(
                    T_init=init.T_init,
                    source=init.source,
                    init_RE=float(init_RE),
                    init_TE=float(init_TE),
                )

            ctx = MethodContext(T_true=sample.T_true, meta={"infra_id": sample.infra_id, "veh_id": sample.veh_id})
            start = perf_counter()
            try:
                res = estimator.estimate(infra_boxes, veh_boxes, init=init, ctx=ctx)
            except Exception as exc:
                res = ExtrinsicEstimate(
                    T=None,
                    success=False,
                    method=method,
                    time_sec=float(perf_counter() - start),
                    extra={"reason": "exception", "error": str(exc)},
                )
            elapsed = perf_counter() - start

            RE = float("inf") if res.RE is None else float(res.RE)
            TE = float("inf") if res.TE is None else float(res.TE)
            matches_count = int(len(res.matches or []))
            records.append(
                FrameMetrics(
                    infra_id=str(sample.infra_id),
                    veh_id=str(sample.veh_id),
                    RE=RE,
                    TE=TE,
                    stability=float(res.stability or 0.0),
                    time_cost=float(res.time_sec if res.time_sec is not None else elapsed),
                    matches_count=matches_count,
                )
            )
            f.write(
                json.dumps(
                    {
                        "index": int(sample.index),
                        "infra_id": str(sample.infra_id),
                        "veh_id": str(sample.veh_id),
                        "bbox_source": source,
                        "init_mode": init_mode,
                        "init_trans_std": float(init_trans_std),
                        "init_rot_std_deg": float(init_rot_std_deg),
                        "init_RE": None if init_RE is None else float(init_RE),
                        "init_TE": None if init_TE is None else float(init_TE),
                        "success": bool(res.success),
                        "RE": None if res.RE is None else float(res.RE),
                        "TE": None if res.TE is None else float(res.TE),
                        "stability": float(res.stability or 0.0),
                        "num_matches": matches_count,
                        "time_sec": float(res.time_sec if res.time_sec is not None else elapsed),
                        "extra": res.extra,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            processed += 1

    summary = aggregate_metrics(records, thresholds)

    def _stats(values: List[float]) -> Optional[dict]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    init_stats = None
    if init_mode.lower() != "none":
        init_stats = {
            "RE_deg": _stats(init_RE_values),
            "TE_m": _stats(init_TE_values),
        }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": method,
                "config": str(config_path),
                "use_detection": bool(cfg.data.use_detection),
                "init_mode": init_mode,
                "init_trans_std": float(init_trans_std),
                "init_rot_std_deg": float(init_rot_std_deg),
                "init_stats": init_stats,
                "v2xregpp_use_prior": bool(v2xregpp_use_prior) if method == "v2xregpp" else None,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate extrinsic estimators on DAIR-V2X pairs.")
    parser.add_argument("--method", choices=["v2xregpp", "cbm", "vips"], required=True)
    parser.add_argument("--config", default="configs/pipeline_hkust_representative.yaml")
    parser.add_argument("--use-detection", action="store_true", help="Use detection cache boxes (if available).")
    parser.add_argument("--max-samples", type=int, default=None, help="Override cfg.data.max_samples; omit to use cfg.")
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--init-mode", choices=["none", "identity", "gt", "noisy_gt"], default="none")
    parser.add_argument("--init-trans-std", type=float, default=2.0)
    parser.add_argument("--init-rot-std-deg", type=float, default=10.0)

    parser.add_argument(
        "--sweep-init-trans-std",
        default="",
        help="Comma-separated translation stds (m) for init sweep; empty disables sweep.",
    )
    parser.add_argument(
        "--sweep-init-rot-std-deg",
        default="",
        help="Comma-separated rotation stds (deg) for init sweep; empty disables sweep.",
    )
    parser.add_argument(
        "--v2xregpp-use-prior",
        action="store_true",
        help="For V2X-Reg++, use init transform as correspondence gate (filter_strategy=trueRetained).",
    )
    parser.add_argument("--output-dir", default="outputs/heal_extrinsic_benchmark")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--data-root", default=None, help="Override cfg.data.data_root (e.g., data/DAIR-V2X)")
    parser.add_argument("--data-info-path", default=None, help="Override cfg.data.data_info_path")
    parser.add_argument(
        "--detection-cache",
        default=None,
        help="Override cfg.data.detection_cache (empty string to disable).",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    out_root = resolve_repo_path(args.output_dir)
    tag = args.tag
    if not tag:
        tag = f"{args.method}_init{args.init_mode}_det{int(args.use_detection)}"
    base_dir = out_root / tag

    trans_list = _parse_floats(args.sweep_init_trans_std)
    rot_list = _parse_floats(args.sweep_init_rot_std_deg)
    if trans_list and rot_list:
        base_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for t_std in trans_list:
            for r_std in rot_list:
                run_tag = f"{tag}_t{t_std:.2f}_r{r_std:.1f}".replace("-", "n").replace(".", "p")
                out_dir = out_root / run_tag
                summary = run_once(
                    method=args.method,
                    config_path=args.config,
                    init_mode=args.init_mode,
                    init_trans_std=float(t_std),
                    init_rot_std_deg=float(r_std),
                    seed=args.seed,
                    max_samples=args.max_samples,
                    use_detection=args.use_detection,
                    v2xregpp_use_prior=args.v2xregpp_use_prior,
                    data_root_override=args.data_root,
                    data_info_override=args.data_info_path,
                    detection_cache_override=args.detection_cache,
                    out_dir=out_dir,
                )
                results.append(
                    {
                        "init_trans_std": float(t_std),
                        "init_rot_std_deg": float(r_std),
                        "output_dir": str(out_dir),
                        "summary": summary,
                    }
                )
        with (base_dir / "sweep_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": args.method,
                    "config": args.config,
                    "use_detection": bool(args.use_detection),
                    "init_mode": args.init_mode,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        run_once(
            method=args.method,
            config_path=args.config,
            init_mode=args.init_mode,
            init_trans_std=args.init_trans_std,
            init_rot_std_deg=args.init_rot_std_deg,
            seed=args.seed,
            max_samples=args.max_samples,
            use_detection=args.use_detection,
            v2xregpp_use_prior=args.v2xregpp_use_prior,
            data_root_override=args.data_root,
            data_info_override=args.data_info_path,
            detection_cache_override=args.detection_cache,
            out_dir=base_dir,
        )


if __name__ == "__main__":
    main()
