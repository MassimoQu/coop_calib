# HEAL Pose Alignment Benchmark Plan

## Goal
Build a single benchmark that positions all methods between two lower bounds
(single-agent perception and initpose-only coop) and one upper bound
(GT extrinsics + coop). The same detection range and evaluation protocol are
used across modalities.

## Benchmark Definition (Merged Track)
- Lower bound A: single-agent perception (single-ckpt), same detection range.
- Lower bound B: initpose-only coop (pose_correction=none, noisy poses, no optimization).
- Upper bound: oracle GT extrinsics + coop (pose_correction=oracle_gt).
- Methods under test: v2xregpp, freealign, vips, cbm with their best strategy
  (best-of vs stable selected per method).

## Fixed Settings (Must Match Across All Methods)
- Dataset: DAIR-V2X cooperative-vehicle-infrastructure test split.
- Detection range: keep cav_lidar_range/lidar_range from the coop configs.
- comm_range_override: 100.
- Noise schedule:
  - pos_std_list: 1..10 m (paired sweep).
  - rot_std_list: camera 1..10 deg, lidar 0 deg.
  - noise_target: non-ego.
- Dropout sweep (optional benchmark axis): pose_dropout_prob=0.2.
- Seeds: numpy seed=303; lock torch/cudnn seeds for full determinism.

## Inputs (Existing Artifacts)
- Coop camera model: HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38
- Coop lidar model: HEAL/opencood/logs/freealign_repro_dair_baseline
- Stage1 caches (current):
  - Camera: data/DAIR-V2X/detected/camera_v2xvit_stage1/stage1_boxes.json
  - LiDAR: HEAL/opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json
- Single-ckpt baselines (same range):
  - Camera: HEAL/opencood/logs/Pyramid_DAIR_m2_lsseff_single_2025_11_24_23_58_05
  - LiDAR: HEAL/opencood/logs/Pyramid_DAIR_m1_pointpillars_single_2025_11_24_18_47_23
- Existing sweep outputs:
  - outputs/pose_sweep_1to10_results.jsonl
  - outputs/pose_dropout_1to10_results.jsonl
  - outputs/pose_sweep_1to10_summary.md
  - outputs/pose_dropout_1to10_summary.md

## Decisions (Locked)
- Stage1 cache source for pose methods: use current coop caches.
- Final benchmark uses full test set (no max_eval_samples cap).

## Implementation Plan
1) Freeze a benchmark manifest:
   - Dataset split, model dirs, stage1 cache paths, commit hash, env hash.
2) Decide stage1 cache source for pose methods:
   - Use current coop caches (as locked above).
3) Run pose sweeps:
   - Noise sweep (1..10 m) and optional dropout sweep with inference_w_noise.py
     via outputs/run_noise_sweep_1to10.py and outputs/run_dropout_sweep_1to10.py.
   - For full test, remove --max-eval-samples and use a new tag (e.g. _full)
     to avoid overwriting prior debug YAMLs and JSONLs.
4) Add lower/upper bounds into the benchmark:
   - Single-agent baseline: fusion_method=single, pos_std=0, rot_std=0.
   - Initpose baseline: pose_correction=none from the sweep.
   - Oracle: pose_correction=oracle_gt from the sweep.
5) Build scorecard and plots:
   - Tables: mean AP50, success@2/5, rel_trans, rel_yaw, fps, pose_time.
   - Curves: AP50 vs noise with lower/upper bound lines.
6) Validate fairness:
   - No GT leakage except oracle; identical comm_range and detection range.

## Output Artifacts
- Benchmark scorecard markdown (single table per modality).
- Benchmark plots (noise sweep curves with bounds).
- JSONL summary (machine-readable).

## Notes
- Expect long runtime for full test sweeps (10 noise levels per method).
