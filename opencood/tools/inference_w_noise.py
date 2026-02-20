# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import random
import os
import time
import warnings
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.extrinsics.pose_correction import build_pose_corrector, run_pose_solver
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import pose_to_tfm
from opencood.visualization import vis_utils, my_vis, simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')
# This script can emit extremely noisy warnings (e.g., torch deprecated sigmoid)
# at every iteration, which slows evaluation drastically and floods stdout.
warnings.filterwarnings("ignore")

def _parse_float_list(raw: str):
    values = []
    for token in (raw or "").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _parse_hw(raw: str):
    raw = str(raw or "").strip()
    if not raw:
        return None
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return None
    try:
        side = int(raw)
    except Exception:
        return None
    return side, side


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument(
        "--pos-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated translation noise std list (meters).",
    )
    parser.add_argument(
        "--rot-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated yaw noise std list (degrees).",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["paired", "grid"],
        default="paired",
        help="paired: zip(pos,rot); grid: Cartesian product.",
    )
    parser.add_argument(
        "--noise-target",
        choices=["all", "ego", "non-ego"],
        default="non-ego",
        help="Which agent(s) receive synthetic pose noise. "
             "'all' matches legacy behavior; 'non-ego' makes relative error scale match the requested std.",
    )
    parser.add_argument(
        "--pose-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of simulating localization dropout per sample. "
             "When triggered, poses reuse the last noisy state (if available).",
    )
    parser.add_argument(
        "--pose-correction",
        choices=[
            "none",
            "v2xregpp_initfree",
            "v2xregpp_stable",
            "freealign_paper",
            "freealign_paper_stable",
            "freealign_repo",
            "freealign_repo_stable",
            "vips_initfree",
            "vips_stable",
            "cbm_initfree",
            "cbm_stable",
            "image_match_initfree",
            "image_match_stable",
            "lidar_reg_initfree",
            "lidar_reg_stable",
            # V2VLoc-style pose override/correction from cached pose JSON.
            "v2vloc_pgc_initfree",
            "v2vloc_pgc_stable",
            # Oracle pose override from clean poses (GT).
            "v2vloc_oracle_initfree",
            "v2vloc_oracle_stable",
            # GT pose override from dataset clean poses.
            "oracle_gt",
        ],
        default="none",
        help="Optional extrinsic correction inside the dataset, before building pairwise transforms.",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="auto",
        help="Device for pose correction (v2xregpp/freealign/vips/cbm). "
             "Use 'auto' to pick cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="",
        choices=["", "single_only", "fusion_only", "register_only", "register_and_fuse"],
        help="Optional unified runtime mode override for pose_provider."
             " Empty keeps legacy behavior.",
    )
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="offline_map",
        choices=["offline_map", "online_box", "online_box_feat_refine"],
        help="Pose solver backend. offline_map keeps legacy pre-pass; online_* runs solver inside pose_provider.",
    )
    parser.add_argument(
        "--online-gpu-stage1-solver",
        action="store_true",
        help="Enable experimental GPU stage1 solver path when using online backends.",
    )
    parser.add_argument(
        "--online-skip-pairwise-rebuild",
        action="store_true",
        help=(
            "When using online backends, reuse dataset pairwise instead of runtime rebuild. "
            "Useful for strict oracle parity checks."
        ),
    )
    parser.add_argument(
        "--deterministic-strict",
        action="store_true",
        help=(
            "Enable stricter determinism controls for parity gates. "
            "This may reduce throughput and can raise errors if nondeterministic ops are used."
        ),
    )
    parser.add_argument(
        "--pose-source",
        type=str,
        default="noisy_input",
        choices=["noisy_input", "gt", "identity"],
        help="Pose source for fusion_only runtime mode.",
    )
    parser.add_argument(
        "--stage1-result",
        type=str,
        default="",
        help="Cache JSON used by the pose corrector (required when --pose-correction != none). "
             "For v2xregpp_* / freealign_* this is a stage1_boxes.json. For v2vloc_pgc_* it is a PGC pose json; "
             "for v2vloc_oracle_* it can reuse stage1_boxes.json (lidar_pose_clean_np). "
             "oracle_gt ignores this path.",
    )
    parser.add_argument(
        "--freealign-max-boxes",
        type=int,
        default=60,
        help="FreeAlign: max number of boxes per agent used for matching.",
    )
    parser.add_argument(
        "--freealign-min-nodes",
        type=int,
        default=5,
        help="FreeAlign(paper): minimum number of nodes/boxes required to attempt matching.",
    )
    parser.add_argument(
        "--freealign-sim-threshold",
        type=float,
        default=0.6,
        help="FreeAlign(paper): similarity threshold used for anchor selection/matching.",
    )
    parser.add_argument(
        "--freealign-affine-method",
        type=str,
        default="lmeds",
        choices=["lmeds", "ransac"],
        help="FreeAlign(paper): OpenCV estimateAffinePartial2D method.",
    )
    parser.add_argument(
        "--freealign-ransac-reproj-threshold",
        type=float,
        default=1.0,
        help="FreeAlign(paper): RANSAC reprojection threshold (meters) in estimateAffinePartial2D.",
    )
    parser.add_argument(
        "--freealign-min-anchors",
        type=int,
        default=3,
        help="FreeAlign(repo): minimum anchors for initial subgraph match.",
    )
    parser.add_argument(
        "--freealign-anchor-error",
        type=float,
        default=0.3,
        help="FreeAlign(repo): anchor error threshold.",
    )
    parser.add_argument(
        "--freealign-box-error",
        type=float,
        default=0.5,
        help="FreeAlign(repo): box error threshold.",
    )
    parser.add_argument(
        "--v2xregpp-config",
        type=str,
        default="configs/dair/midfusion/pipeline_midfusion_detection_occ.yaml",
        help="V2X-Reg++ pipeline config used by the pose corrector.",
    )
    parser.add_argument(
        "--v2xregpp-stage1-field",
        type=str,
        default="pred_corner3d_np_list",
        help="Stage1 result field name that contains per-agent 3D boxes (e.g., pred_corner3d_np_list, feature_corner3d_np_list).",
    )
    parser.add_argument(
        "--v2xregpp-bbox-type",
        type=str,
        default="detected",
        help="BBox type tag used inside V2X-Reg++ matching thresholds (e.g., detected, feature).",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-hint",
        action="store_true",
        help="Enable occ-hint seed when stage1_result contains occ_map_level0/paths.",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-pose",
        action="store_true",
        help="Treat occ-hint as a first-class pose candidate (and allow occ-only pose updates when boxes are sparse).",
    )
    parser.add_argument(
        "--v2xregpp-force-occ-pose",
        action="store_true",
        help="Always use occ-based pose (optionally refined), making the output independent of injected pose noise.",
    )
    parser.add_argument(
        "--v2xregpp-occ-from-lidar",
        action="store_true",
        help="Build BEV occupancy maps from raw lidar points (ignores stage1 occ_map fields).",
    )
    parser.add_argument(
        "--v2xregpp-occ-grid",
        type=str,
        default="256",
        help="Occupancy grid size for --v2xregpp-occ-from-lidar (int or 'H,W').",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-xy",
        type=float,
        default=20.0,
        help="Reject occ-hint if it differs from current pose by more than this translation (meters).",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-yaw",
        type=float,
        default=45.0,
        help="Reject occ-hint if it differs from current pose by more than this yaw (degrees).",
    )
    parser.add_argument(
        "--v2xregpp-icp-refine",
        action="store_true",
        help="Run Open3D ICP refinement on raw lidar points using the estimated transform as init.",
    )
    parser.add_argument("--v2xregpp-icp-voxel", type=float, default=1.0, help="ICP voxel downsample size (meters).")
    parser.add_argument("--v2xregpp-icp-max-corr", type=float, default=2.0, help="ICP max correspondence distance (meters).")
    parser.add_argument("--v2xregpp-icp-max-iter", type=int, default=30, help="ICP max iterations.")
    parser.add_argument("--v2xregpp-min-matches", type=int, default=3)
    parser.add_argument("--v2xregpp-min-stability", type=float, default=0.0)
    parser.add_argument(
        "--v2xregpp-min-precision",
        type=float,
        default=0.0,
        help="Absolute precision threshold (CorrespondingDetector) required to apply an estimated pose. 0 disables.",
    )
    parser.add_argument("--v2xregpp-ema-alpha", type=float, default=0.5)
    parser.add_argument("--v2xregpp-max-step-xy", type=float, default=3.0)
    parser.add_argument("--v2xregpp-max-step-yaw", type=float, default=10.0)
    parser.add_argument("--pose-compare-current", action="store_true", help="Compare estimated pose with current pose and keep the better one.")
    parser.add_argument("--pose-compare-distance-threshold", type=float, default=3.0, help="Distance threshold for pose comparison (meters).")
    parser.add_argument("--pose-current-precision-threshold", type=float, default=None, help="Skip update if current precision exceeds this (set <0 to disable).")
    parser.add_argument("--pose-min-precision-improvement", type=float, default=None, help="Minimum precision improvement over current pose.")
    parser.add_argument("--pose-min-matched-improvement", type=int, default=None, help="Minimum matched-count improvement over current pose.")
    parser.add_argument("--pose-min-precision", type=float, default=None, help="Absolute minimum precision required to apply estimated pose.")
    parser.add_argument("--vips-use-prior", action="store_true", help="VIPS: use current pose as initialization.")
    parser.add_argument("--vips-match-threshold", type=float, default=0.5, help="VIPS: matching threshold.")
    parser.add_argument("--vips-match-distance", type=float, default=8.0, help="VIPS: match distance threshold (meters).")
    parser.add_argument("--cbm-use-prior", action="store_true", help="CBM: use current pose as initialization.")
    parser.add_argument("--cbm-sigma1-deg", type=float, default=10.0, help="CBM: sigma1 (deg).")
    parser.add_argument("--cbm-sigma2-m", type=float, default=3.0, help="CBM: sigma2 (meters).")
    parser.add_argument("--cbm-sigma3-m", type=float, default=1.0, help="CBM: sigma3 (meters).")
    parser.add_argument("--cbm-absolute-dis-lim", type=float, default=20.0, help="CBM: absolute distance limit (meters).")
    parser.add_argument(
        "--image-match-matcher",
        type=str,
        default="orb",
        choices=["orb", "sift", "loftr", "disk", "lightglue"],
    )
    parser.add_argument("--image-match-max-features", type=int, default=4000)
    parser.add_argument("--image-match-ratio-test", type=float, default=0.75)
    parser.add_argument("--image-match-cross-check", action="store_true")
    parser.add_argument("--image-match-ransac-thresh", type=float, default=1.0)
    parser.add_argument("--image-match-ransac-confidence", type=float, default=0.999)
    parser.add_argument("--image-match-ransac-max-iters", type=int, default=2000)
    parser.add_argument("--image-match-min-matches", type=int, default=20)
    parser.add_argument("--image-match-min-inliers", type=int, default=15)
    parser.add_argument("--image-match-resize-max-dim", type=int, default=1024)
    parser.add_argument("--image-match-allow-no-intrinsics", action="store_true")
    parser.add_argument("--image-match-t-scale", type=float, default=None)
    parser.add_argument("--image-match-device", type=str, default="cpu")
    parser.add_argument("--image-match-camera-index", type=int, default=0)
    parser.add_argument("--image-match-camera-indices", type=str, default="")
    parser.add_argument("--image-match-try-all-cameras", action="store_true")
    parser.add_argument("--image-match-init-source", type=str, default="current", choices=["current", "clean", "none"])
    parser.add_argument("--image-match-min-stability", type=float, default=0.0)
    parser.add_argument("--lidar-reg-voxel-size", type=float, default=1.0)
    parser.add_argument("--lidar-reg-max-corr", type=float, default=2.0)
    parser.add_argument("--lidar-reg-ransac-n", type=int, default=4)
    parser.add_argument("--lidar-reg-ransac-max-iter", type=int, default=50000)
    parser.add_argument("--lidar-reg-ransac-confidence", type=float, default=0.999)
    parser.add_argument("--lidar-reg-use-fgr", action="store_true")
    parser.add_argument(
        "--lidar-reg-icp-method",
        type=str,
        default="point_to_plane",
        choices=["point_to_plane", "point_to_point", "gicp"],
    )
    parser.add_argument("--lidar-reg-icp-max-iter", type=int, default=50)
    parser.add_argument("--lidar-reg-min-points", type=int, default=200)
    parser.add_argument("--lidar-reg-max-points", type=int, default=60000)
    parser.add_argument("--lidar-reg-min-fitness", type=float, default=0.0)
    parser.add_argument("--lidar-reg-max-inlier-rmse", type=float, default=0.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. For v2xregpp_stable, keep this at 0 to preserve temporal state.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples per noise setting (debug only).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N samples (set 1 to print every sample).",
    )
    parser.add_argument(
        "--comm-range-override",
        type=float,
        default=None,
        help="Optional override for hypes['comm_range'] (useful for pose-correction stress tests).",
    )
    parser.add_argument(
        "--pose-timing",
        action="store_true",
        help="Collect pose correction timing per sample.",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_range_override is not None:
        hypes["comm_range"] = float(opt.comm_range_override)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or 'V2XSET' in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    pos_std_list = _parse_float_list(opt.pos_std_list)
    rot_std_list = _parse_float_list(opt.rot_std_list)
    if not pos_std_list:
        pos_std_list = [0.0]
    if not rot_std_list:
        rot_std_list = [0.0]
    sweep_mode = opt.sweep_mode
    if sweep_mode == "paired":
        if len(pos_std_list) == 1 and len(rot_std_list) > 1:
            pos_std_list = pos_std_list * len(rot_std_list)
        if len(rot_std_list) == 1 and len(pos_std_list) > 1:
            rot_std_list = rot_std_list * len(pos_std_list)
        if len(pos_std_list) != len(rot_std_list):
            raise ValueError("paired sweep requires pos-std-list and rot-std-list to have equal length (or one of them length=1).")
        noise_pairs = list(zip(pos_std_list, rot_std_list))
    else:
        noise_pairs = [(p, r) for p in pos_std_list for r in rot_std_list]

    # Optional pose correction config injection.
    pose_correction = opt.pose_correction
    compare_current = bool(opt.pose_compare_current)
    compare_distance_threshold = float(opt.pose_compare_distance_threshold)
    apply_if_current_precision_below = opt.pose_current_precision_threshold
    min_precision_improvement = opt.pose_min_precision_improvement
    min_matched_improvement = opt.pose_min_matched_improvement
    min_precision = opt.pose_min_precision
    if compare_current:
        if min_precision_improvement is None:
            min_precision_improvement = 0.0
        if min_matched_improvement is None:
            min_matched_improvement = 0
        if apply_if_current_precision_below is None:
            apply_if_current_precision_below = -1.0
    pose_compare_args = {}
    if compare_current:
        pose_compare_args["compare_with_current"] = True
    if compare_distance_threshold is not None:
        pose_compare_args["compare_distance_threshold_m"] = float(compare_distance_threshold)
    if apply_if_current_precision_below is not None:
        pose_compare_args["apply_if_current_precision_below"] = float(apply_if_current_precision_below)
    if min_precision_improvement is not None:
        pose_compare_args["min_precision_improvement"] = float(min_precision_improvement)
    if min_matched_improvement is not None:
        pose_compare_args["min_matched_improvement"] = int(min_matched_improvement)
    if min_precision is not None:
        pose_compare_args["min_precision"] = float(min_precision)
    pose_solver_spec = None
    pose_solver_stage1 = None
    pose_solver_pose = None
    simple_override_cfg = None
    runtime_mode_opt = str(opt.runtime_mode or "").lower().strip()
    solver_backend_opt = str(opt.solver_backend or "offline_map").lower().strip()
    pose_source_opt = str(opt.pose_source or "noisy_input").lower().strip()
    pose_device = str(opt.pose_device or "auto").lower()
    if pose_device == "auto":
        pose_device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 303
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        # These flags reduce nondeterminism in cuDNN. Keep them on by default because
        # this script is used for evidence-grade benchmark runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    if bool(getattr(opt, "deterministic_strict", False)):
        # Parity gates can require tighter numeric stability than typical benchmark runs.
        # Prefer correctness over speed when explicitly requested.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            # `warn_only=False` enforces deterministic kernels; some ops may throw.
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fall back to best-effort determinism rather than crashing.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    if pose_correction != "none":
        if (
            not opt.stage1_result
            and not pose_correction.startswith(("image_match", "lidar_reg"))
            and pose_correction != "oracle_gt"
        ):
            raise ValueError("--stage1-result is required when --pose-correction != none")
        # Ensure we don't accidentally combine multiple alignment blocks.
        hypes.pop("box_align", None)
        hypes.pop("v2xregpp_align", None)
        hypes.pop("freealign_align", None)
        hypes.pop("vips_align", None)
        hypes.pop("cbm_align", None)
        hypes.pop("pgc_pose", None)
        hypes.pop("image_match_align", None)
        raw_pose_override_cfg = hypes.get("pose_override")
        if isinstance(raw_pose_override_cfg, dict) and raw_pose_override_cfg.get("mode"):
            simple_override_cfg = dict(raw_pose_override_cfg)
        if pose_correction.startswith("v2xregpp"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "v2xregpp"
            args = {
                "config_path": opt.v2xregpp_config,
                "device": pose_device,
                "mode": mode,
                "use_occ_hint": bool(opt.v2xregpp_use_occ_hint),
                "use_occ_pose": bool(opt.v2xregpp_use_occ_pose),
                "force_occ_pose": bool(opt.v2xregpp_force_occ_pose),
                "occ_from_lidar": bool(opt.v2xregpp_occ_from_lidar),
                "occ_grid_hw": _parse_hw(opt.v2xregpp_occ_grid) or (256, 256),
                "occ_max_delta_xy_m": float(opt.v2xregpp_occ_max_delta_xy),
                "occ_max_delta_yaw_deg": float(opt.v2xregpp_occ_max_delta_yaw),
                "icp_refine": bool(opt.v2xregpp_icp_refine),
                "icp_voxel_size_m": float(opt.v2xregpp_icp_voxel),
                "icp_max_corr_dist_m": float(opt.v2xregpp_icp_max_corr),
                "icp_max_iterations": int(opt.v2xregpp_icp_max_iter),
                "min_matches": int(opt.v2xregpp_min_matches),
                "min_stability": float(opt.v2xregpp_min_stability),
                "min_precision": float(opt.v2xregpp_min_precision),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
            }
            v2xregpp_compare_args = {}
            if apply_if_current_precision_below is not None:
                v2xregpp_compare_args["apply_if_current_precision_below"] = float(apply_if_current_precision_below)
            if min_precision_improvement is not None:
                v2xregpp_compare_args["min_precision_improvement"] = float(min_precision_improvement)
            if min_matched_improvement is not None:
                v2xregpp_compare_args["min_matched_improvement"] = int(min_matched_improvement)
            if v2xregpp_compare_args:
                args.update(v2xregpp_compare_args)
        elif pose_correction.startswith("freealign"):
            backend = "repo" if "repo" in pose_correction else "paper"
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "freealign"
            args = {
                "backend": backend,
                "mode": mode,
                "device": str(pose_device),
                # Reuse V2X-Reg++ stable hyperparams so "stable" means the same across methods.
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "max_boxes": int(opt.freealign_max_boxes),
                # Paper configs (repo backend will ignore unknown fields).
                "min_nodes": int(opt.freealign_min_nodes),
                "sim_threshold": float(opt.freealign_sim_threshold),
                "affine_method": str(opt.freealign_affine_method),
                "ransac_reproj_threshold": float(opt.freealign_ransac_reproj_threshold),
                # Repo configs (paper backend will ignore unknown fields).
                "min_anchors": int(opt.freealign_min_anchors),
                "anchor_error": float(opt.freealign_anchor_error),
                "box_error": float(opt.freealign_box_error),
            }
            if pose_compare_args:
                args.update(pose_compare_args)
        elif pose_correction.startswith("vips"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "vips"
            args = {
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
                "mode": mode,
                "use_prior": bool(opt.vips_use_prior),
                "device": str(pose_device),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "match_threshold": float(opt.vips_match_threshold),
                "match_distance_thr_m": float(opt.vips_match_distance),
            }
            if pose_compare_args:
                args.update(pose_compare_args)
        elif pose_correction.startswith("cbm"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "cbm"
            args = {
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
                "mode": mode,
                "use_prior": bool(opt.cbm_use_prior),
                "device": str(pose_device),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "sigma1_deg": float(opt.cbm_sigma1_deg),
                "sigma2_m": float(opt.cbm_sigma2_m),
                "sigma3_m": float(opt.cbm_sigma3_m),
                "absolute_dis_lim_m": float(opt.cbm_absolute_dis_lim),
            }
            if pose_compare_args:
                args.update(pose_compare_args)
        elif pose_correction.startswith("image_match"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "image_match"
            args = {
                "mode": mode,
                "camera_index": int(opt.image_match_camera_index),
                "camera_indices": str(opt.image_match_camera_indices or ""),
                "try_all_cameras": bool(opt.image_match_try_all_cameras),
                "init_source": str(opt.image_match_init_source),
                "min_stability": float(opt.image_match_min_stability),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "matcher": str(opt.image_match_matcher),
                "max_features": int(opt.image_match_max_features),
                "ratio_test": float(opt.image_match_ratio_test),
                "cross_check": bool(opt.image_match_cross_check),
                "ransac_thresh_px": float(opt.image_match_ransac_thresh),
                "ransac_confidence": float(opt.image_match_ransac_confidence),
                "ransac_max_iters": int(opt.image_match_ransac_max_iters),
                "min_matches": int(opt.image_match_min_matches),
                "min_inliers": int(opt.image_match_min_inliers),
                "resize_max_dim": int(opt.image_match_resize_max_dim),
                "allow_no_intrinsics": bool(opt.image_match_allow_no_intrinsics),
                "t_scale": None if opt.image_match_t_scale is None else float(opt.image_match_t_scale),
                "device": str(opt.image_match_device),
            }
            if pose_compare_args:
                image_compare_args = {}
                for key in ("compare_with_current", "compare_distance_threshold_m"):
                    if key in pose_compare_args:
                        image_compare_args[key] = pose_compare_args[key]
                if image_compare_args:
                    args.update(image_compare_args)
        elif pose_correction.startswith("lidar_reg"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "lidar_reg"
            args = {
                "mode": mode,
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "min_fitness": float(opt.lidar_reg_min_fitness),
                "max_inlier_rmse": float(opt.lidar_reg_max_inlier_rmse),
                "voxel_size_m": float(opt.lidar_reg_voxel_size),
                "max_corr_dist_m": float(opt.lidar_reg_max_corr),
                "ransac_n": int(opt.lidar_reg_ransac_n),
                "ransac_max_iter": int(opt.lidar_reg_ransac_max_iter),
                "ransac_confidence": float(opt.lidar_reg_ransac_confidence),
                "use_fgr": bool(opt.lidar_reg_use_fgr),
                "icp_method": str(opt.lidar_reg_icp_method),
                "icp_max_iter": int(opt.lidar_reg_icp_max_iter),
                "min_points": int(opt.lidar_reg_min_points),
                "max_points": int(opt.lidar_reg_max_points),
            }
        elif pose_correction.startswith("v2vloc_"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            # We intentionally *don't* write pose_confidence here, so that pose confidence
            # is computed consistently via `attach_pose_confidence` from (pose - pose_clean)
            # for all methods, isolating the effect of pose alignment.
            if "oracle" in pose_correction:
                pose_field = "lidar_pose_clean_np"
            else:
                pose_field = "lidar_pose_pred_np"
            method = "pgc"
            args = {
                "pose_field": str(pose_field),
                "confidence_field": "",
                "min_confidence": 0.0,
                "mode": str(mode),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "freeze_ego": True,
            }
        elif pose_correction == "oracle_gt":
            method = "gt"
            args = {
                "freeze_ego": True,
            }
        else:
            raise ValueError(f"Unsupported --pose-correction: {pose_correction}")

        stage1_path = None
        if opt.stage1_result and method not in {"image_match", "lidar_reg"}:
            stage1_path = str(resolve_repo_path(opt.stage1_result))
            if method == "pgc":
                pose_solver_pose = read_json(stage1_path)
            else:
                pose_solver_stage1 = read_json(stage1_path)
        pose_solver_spec = {"method": method, "args": args}

        pose_provider_cfg = dict(hypes.get("pose_provider") or {})
        if solver_backend_opt in {"online_box", "online_box_feat_refine"}:
            pose_provider_cfg["enabled"] = True
            pose_provider_cfg["runtime_mode"] = runtime_mode_opt or "register_and_fuse"
            pose_provider_cfg["solver_backend"] = solver_backend_opt
            pose_provider_cfg["pose_source"] = pose_source_opt
            pose_provider_cfg["online_method"] = method
            online_args = dict(args)
            if bool(getattr(opt, "online_gpu_stage1_solver", False)):
                online_args["gpu_stage1_solver"] = True
            if bool(getattr(opt, "online_skip_pairwise_rebuild", False)):
                online_args["skip_pairwise_rebuild"] = True
            pose_provider_cfg["online_args"] = online_args
            pose_provider_cfg["recompute_pairwise"] = True
            if stage1_path:
                if method == "pgc":
                    pose_provider_cfg["pose_result"] = stage1_path
                else:
                    pose_provider_cfg["stage1_result"] = stage1_path
            hypes["pose_provider"] = pose_provider_cfg
            # Keep comm-range gating consistent with offline-map correction path:
            # use clean pose for dataset-side pruning when runtime correction is active.
            hypes["comm_range_use_clean_pose"] = True
            # Online backend should not rely on dataset-side override maps.
            hypes["pose_override"] = {"enabled": False}
        else:
            pose_override_cfg = dict(hypes.get("pose_override") or {})
            for key in ("path", "pose_path", "pose_result", "pose_map"):
                pose_override_cfg.pop(key, None)
            pose_override_cfg["enabled"] = True
            pose_override_cfg.setdefault("pose_field", "lidar_pose_pred_np")
            pose_override_cfg.setdefault("confidence_field", "pose_confidence_np")
            hypes["pose_override"] = pose_override_cfg

    if pose_correction == "none" and runtime_mode_opt:
        pose_provider_cfg = dict(hypes.get("pose_provider") or {})
        pose_provider_cfg["enabled"] = True
        pose_provider_cfg["runtime_mode"] = runtime_mode_opt
        pose_provider_cfg["solver_backend"] = solver_backend_opt
        pose_provider_cfg["pose_source"] = pose_source_opt
        pose_provider_cfg["recompute_pairwise"] = True
        hypes["pose_provider"] = pose_provider_cfg

    if opt.pose_timing:
        hypes["pose_timing"] = True
    
    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        timing_stats_all = []
        rel_error_stats_all = []
        mean_pos = 0.0
        mean_rot = 0.0
        # Build the dataset once and only mutate the noise_setting per sweep entry.
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_setting['add_noise'] = True
        noise_setting['args'] = {
            'pos_std': 0.0,
            'rot_std': 0.0,
            'pos_mean': mean_pos,
            'rot_mean': mean_rot,
            'target': opt.noise_target,
        }
        if float(opt.pose_dropout_prob or 0.0) > 0.0:
            noise_setting['args']['dropout_prob'] = float(opt.pose_dropout_prob)
        if use_laplace:
            noise_setting['args']['laplace'] = True
        hypes.update({"noise_setting": noise_setting})
        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        num_workers = opt.num_workers
        if num_workers is None:
            # When pose correction loads a large stage1 cache JSON, multi-worker
            # DataLoader will fork/copy the dict into each worker and can easily
            # OOM. Default to 0 workers for correction modes unless explicitly
            # overridden.
            num_workers = 0 if pose_correction != "none" else 4
            if pose_correction.endswith("stable"):
                num_workers = 0

        for pos_std, rot_std in noise_pairs:
            # setting noise
            noise_setting = OrderedDict()
            noise_args = {
                'pos_std': pos_std,
                'rot_std': rot_std,
                'pos_mean': mean_pos,
                'rot_mean': mean_rot,
                'target': opt.noise_target,
            }
            if float(opt.pose_dropout_prob or 0.0) > 0.0:
                noise_args['dropout_prob'] = float(opt.pose_dropout_prob)

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"

            pose_solver_metrics = None
            if pose_solver_spec is not None and solver_backend_opt == "offline_map":
                corrector = build_pose_corrector(pose_solver_spec["method"], args=pose_solver_spec["args"])
                solver_result = run_pose_solver(
                    opencood_dataset,
                    corrector=corrector,
                    stage1_result=pose_solver_stage1,
                    pose_result=pose_solver_pose,
                    noise_setting=noise_setting,
                    max_samples=opt.max_eval_samples,
                    seed=303,
                    simple_override_cfg=simple_override_cfg,
                )
                pose_solver_metrics = solver_result.metrics
                if hasattr(opencood_dataset, "set_pose_override_map"):
                    opencood_dataset.set_pose_override_map(solver_result.overrides)
                else:
                    opencood_dataset.pose_override_map = solver_result.overrides
                    opencood_dataset.pose_override_enabled = True
                no_noise_setting = OrderedDict()
                no_noise_setting["add_noise"] = False
                no_noise_setting["args"] = {
                    "pos_std": 0.0,
                    "rot_std": 0.0,
                    "pos_mean": mean_pos,
                    "rot_mean": mean_rot,
                    "target": opt.noise_target,
                }
                opencood_dataset.params['noise_setting'] = no_noise_setting
            elif (
                pose_solver_spec is not None
                and solver_backend_opt in {"online_box", "online_box_feat_refine"}
                and str(pose_solver_spec.get("method") or "").lower().strip() == "gt"
            ):
                # For oracle GT runtime, keep eval geometry identical to offline-map path.
                no_noise_setting = OrderedDict()
                no_noise_setting["add_noise"] = False
                no_noise_setting["args"] = {
                    "pos_std": 0.0,
                    "rot_std": 0.0,
                    "pos_mean": mean_pos,
                    "rot_mean": mean_rot,
                    "target": opt.noise_target,
                }
                opencood_dataset.params['noise_setting'] = no_noise_setting
            else:
                opencood_dataset.params['noise_setting'] = noise_setting
            # If pre-processing already produces CUDA tensors (e.g., GPU voxelization),
            # DataLoader pin_memory will fail and worker forking will crash. Disable
            # pin_memory and force num_workers=0 in that case.
            pin_memory = torch.cuda.is_available()
            force_workers_zero = False
            if pin_memory:
                try:
                    pre_processors = []
                    if hasattr(opencood_dataset, "pre_processor"):
                        pre_processors.append(getattr(opencood_dataset, "pre_processor"))
                    for attr in dir(opencood_dataset):
                        if attr.startswith("pre_processor_"):
                            pre_processors.append(getattr(opencood_dataset, attr))
                    if any(getattr(pp, "use_gpu", False) for pp in pre_processors if pp is not None):
                        pin_memory = False
                        force_workers_zero = True
                except Exception:
                    # Fall back to default pin_memory behavior.
                    pass
            if force_workers_zero and int(num_workers) > 0:
                print("[WARN] GPU preprocessor detected; forcing num_workers=0 to avoid CUDA fork issues.")
                num_workers = 0
            data_loader = DataLoader(
                opencood_dataset,
                batch_size=1,
                num_workers=int(num_workers),
                collate_fn=opencood_dataset.collate_batch_test,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
            )
            print(f"Noise Added: {pos_std}/{rot_std}/{mean_pos}/{mean_rot}.")
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            
            noise_level = f"{pos_std}_{rot_std}_{mean_pos}_{mean_rot}_" + opt.fusion_method + suffix + opt.note
            if pose_correction != "none":
                noise_level = f"{noise_level}_{pose_correction}"

            rel_trans_errors = []
            rel_yaw_errors = []
            pose_timing = {}
            infer_start = time.perf_counter()
            sample_count = 0


            for i, batch_data in enumerate(data_loader):
                if int(opt.log_interval or 0) > 0 and (i % int(opt.log_interval) == 0):
                    print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                sample_count += 1
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data = train_utils.maybe_apply_pose_provider(batch_data, hypes)
                    timing_payload = batch_data.get('ego', {}).get('pose_timing')
                    if timing_payload:
                        def _ingest(payload):
                            if isinstance(payload, list):
                                for item in payload:
                                    _ingest(item)
                                return
                            if not isinstance(payload, dict):
                                return
                            for key, val in payload.items():
                                if isinstance(val, bool) or not isinstance(val, (int, float)):
                                    continue
                                key_str = str(key)
                                if not (key_str.endswith("_sec") or key_str.endswith("_count")):
                                    continue
                                pose_timing.setdefault(key_str, []).append(float(val))
                        _ingest(timing_payload)
                    
                    if opt.fusion_method == 'late':
                        infer_result = inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        infer_result = inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'single':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset,
                                                                        single_gt=True)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                'fusion is supported.')

                    pred_box_tensor = infer_result['pred_box_tensor']
                    gt_box_tensor = infer_result['gt_box_tensor']
                    pred_score = infer_result['pred_score']

                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)

                    try:
                        poses = batch_data['ego']['lidar_pose'].detach().cpu().numpy().reshape(-1, 6)
                        poses_clean = batch_data['ego']['lidar_pose_clean'].detach().cpu().numpy().reshape(-1, 6)
                        T_world = pose_to_tfm(poses)
                        T_world_clean = pose_to_tfm(poses_clean)
                        ego_T_world = T_world[0]
                        ego_T_world_clean = T_world_clean[0]
                        for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
                            rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
                            rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
                            err = np.linalg.inv(rel_clean) @ rel
                            rel_trans_errors.append(float(np.linalg.norm(err[:2, 3])))
                            yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
                            rel_yaw_errors.append(abs(yaw))
                    except Exception:
                        pass


                    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None) and (use_laplace is False):
                        vis_save_path_root = os.path.join(opt.model_dir, f'vis_{noise_level}')
                        if not os.path.exists(vis_save_path_root):
                            os.makedirs(vis_save_path_root)

                        """ If you want to 3d vis, uncomment lines below """
                        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                        # simple_vis.visualize(infer_result,
                        #                     batch_data['ego'][
                        #                         'origin_lidar'][0],
                        #                     hypes['postprocess']['gt_range'],
                        #                     vis_save_path,
                        #                     method='3d',
                        #                     left_hand=left_hand)
                        
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand)

                torch.cuda.empty_cache()
                if opt.max_eval_samples is not None and (i + 1) >= int(opt.max_eval_samples):
                    break

            infer_elapsed = float(time.perf_counter() - infer_start)
            timing_summary = {}
            if pose_timing:
                timing_summary = {
                    str(k): float(np.mean(v)) for k, v in pose_timing.items() if v
                }
            pose_total_sec = 0.0
            if timing_summary:
                pose_total_sec = float(sum(val for key, val in timing_summary.items() if str(key).endswith("_sec")))
            timing_stats = {
                "samples": int(sample_count),
                "infer_sec": float(infer_elapsed),
                "infer_fps": float(sample_count / infer_elapsed) if infer_elapsed > 0.0 and sample_count > 0 else None,
                "pose_sec": float(pose_total_sec) if pose_total_sec > 0.0 else None,
                "pose_fps": float(1.0 / pose_total_sec) if pose_total_sec > 0.0 else None,
                "pose_timing": timing_summary if timing_summary else None,
            }
            if pose_solver_metrics is not None:
                timing_stats["pose_solver"] = pose_solver_metrics

            ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                        opt.model_dir, noise_level)
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)
            timing_stats_all.append(timing_stats)

            rel_stats = {}
            if rel_trans_errors:
                arr = np.asarray(rel_trans_errors, dtype=np.float64)
                rel_stats['rel_trans_m'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
                thresholds_m = [1.0, 2.0, 3.0, 5.0, 10.0]
                rel_stats["rel_success_at_m"] = {
                    (str(int(thr)) if float(thr).is_integer() else str(thr)): float(np.mean(arr < float(thr)))
                    for thr in thresholds_m
                }
            if rel_yaw_errors:
                arr = np.asarray(rel_yaw_errors, dtype=np.float64)
                rel_stats['rel_yaw_deg'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
            rel_error_stats_all.append(
                {
                    'pos_std': float(pos_std),
                    'rot_std': float(rot_std),
                    **rel_stats,
                }
            )

            dump_dict = {
                'pos_std_list': [float(p) for p, _ in noise_pairs],
                'rot_std_list': [float(r) for _, r in noise_pairs],
                'noise_target': str(opt.noise_target or "all"),
                'ap30': AP30,
                'ap50': AP50,
                'ap70': AP70,
                'pose_correction': pose_correction,
                'timing_stats': timing_stats_all,
                'rel_error_stats': rel_error_stats_all,
            }
            tag_safe = str(opt.note or "").replace("/", "_").replace(" ", "")
            corr_safe = str(pose_correction or "none")
            yaml_utils.save_yaml(
                dump_dict,
                os.path.join(opt.model_dir, f'AP030507_{corr_safe}{suffix}{tag_safe}.yaml'),
            )


if __name__ == '__main__':
    main()
