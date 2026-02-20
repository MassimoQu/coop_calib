# HEAL Pose Alignment + Coop Perception Sweep (DAIR-V2X V2XViT / FreeAlign Repro)

## Purpose
- Compare v2xreg++, FreeAlign, VIPS, and CBM under best-of vs stable strategies.
- Evaluate robustness under 1–10 m extrinsic noise and 0.2 dropout (paired sweep).
- Report success rate, accuracy, and timing in a fair, fixed-eval setting.

## Experiment Date
- 2026-02-02 20:52:37 +0800

## Experiment Setup
- Repo: /home/qqxluca/projects/v2xreg_private
- Python: /home/qqxluca/.micromamba/envs/heal/bin/python
- max_eval_samples: 100 (fast sweep)
- sweep_mode: paired
- noise_target: non-ego
- comm_range_override: 100
- pose_timing: enabled
- num_workers: 0
- save_vis_interval: 100000000

### Camera Setup (V2XViT)
- Model dir: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38
- Stage1 boxes: /home/qqxluca/projects/v2xreg_private/data/DAIR-V2X/detected/camera_v2xvit_stage1/stage1_boxes.json
- rot_std_list: 1..10 deg

### LiDAR Setup (FreeAlign Repro)
- Model dir: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline
- Stage1 boxes: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json
- rot_std_list: all zeros

### Shared Sweep Settings
- pos_std_list: 1..10 m
- pos_mean=0, rot_mean=0
- Dropout sweep: pose_dropout_prob=0.2
- Best-of logic: --pose-compare-current with method-specific distance threshold

## Method Hyperparameters (Best per Strategy)
### Camera best-of
- v2xregpp_initfree: min_matches=3, min_precision=0.05
- freealign_paper: compare_distance_threshold=3.0
- vips_initfree: compare_distance_threshold=5.0
- cbm_initfree: compare_distance_threshold=1.0

### Camera stable
- v2xregpp_stable: ema_alpha=0.2, max_step_xy=3.0, max_step_yaw=10.0
- freealign_paper_stable: ema_alpha=0.2, max_step_xy=3.0, max_step_yaw=10.0
- vips_stable: ema_alpha=0.2, max_step_xy=6.0, max_step_yaw=20.0
- cbm_stable: ema_alpha=0.2, max_step_xy=6.0, max_step_yaw=10.0

### LiDAR best-of
- v2xregpp_initfree: min_matches=3, min_precision=0.0
- freealign_repo: compare_distance_threshold=1.0
- vips_initfree: compare_distance_threshold=3.0
- cbm_initfree: compare_distance_threshold=1.0

### LiDAR stable
- v2xregpp_stable: ema_alpha=0.8, max_step_xy=6.0, max_step_yaw=10.0
- freealign_repo_stable: ema_alpha=0.8, max_step_xy=6.0, max_step_yaw=10.0
- vips_stable: ema_alpha=0.2, max_step_xy=3.0, max_step_yaw=10.0
- cbm_stable: ema_alpha=0.2, max_step_xy=3.0, max_step_yaw=10.0

## Noise Sweep Results (AP50, 1–10 m)
### Plots
- camera best: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_plots/pose_sweep_1to10_camera_best_ap50.png
- camera stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_plots/pose_sweep_1to10_camera_stable_ap50.png
- lidar best: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_plots/pose_sweep_1to10_lidar_best_ap50.png
- lidar stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_plots/pose_sweep_1to10_lidar_stable_ap50.png
- Plots include baseline (pose_correction=none) and oracle (pose_correction=oracle_gt) curves.

### Summary (mean over 1–10)
| modality / strategy | method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera / best | v2xregpp | 0.151 | 0.227 | 0.501 | 6.250 | 4.501 | 33.1 | 2.05 |
| camera / best | freealign | 0.133 | 0.196 | 0.452 | 12.587 | 15.398 | 24.7 | 2.04 |
| camera / best | vips | 0.135 | 0.185 | 0.414 | 19.202 | 23.990 | 29.3 | 2.08 |
| camera / best | cbm | 0.093 | 0.054 | 0.110 | 58.487 | 66.758 | 85.0 | 2.12 |
| camera / stable | v2xregpp | 0.115 | 0.015 | 0.062 | 20.277 | 15.243 | 33.1 | 2.03 |
| camera / stable | freealign | 0.079 | 0.010 | 0.034 | 56.079 | 20.801 | 11.8 | 2.13 |
| camera / stable | vips | 0.053 | 0.009 | 0.029 | 140.871 | 159.783 | 19.5 | 2.08 |
| camera / stable | cbm | 0.048 | 0.008 | 0.028 | 118.470 | 114.501 | 62.4 | 2.16 |
| lidar / best | v2xregpp | 0.356 | 0.523 | 0.728 | 4.802 | 2.428 | 33.4 | 2.64 |
| lidar / best | freealign | 0.354 | 0.449 | 0.704 | 4.634 | 2.204 | 64.2 | 2.62 |
| lidar / best | vips | 0.257 | 0.167 | 0.368 | 26.868 | 32.216 | 23.7 | 2.53 |
| lidar / best | cbm | 0.288 | 0.038 | 0.072 | 70.108 | 80.281 | 65.8 | 2.56 |
| lidar / stable | v2xregpp | 0.273 | 0.261 | 0.401 | 7.929 | 0.925 | 34.7 | 2.53 |
| lidar / stable | freealign | 0.269 | 0.259 | 0.403 | 7.947 | 1.039 | 52.6 | 2.51 |
| lidar / stable | vips | 0.223 | 0.004 | 0.015 | 52.433 | 59.350 | 14.0 | 2.58 |
| lidar / stable | cbm | 0.378 | 0.000 | 0.002 | 122.720 | 89.068 | 46.7 | 3.03 |

### Bounds (baseline / oracle)
| modality | method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera | baseline | 0.143 | 0.185 | 0.473 | 6.762 | 4.409 | n/a | 2.10 |
| camera | oracle | 0.264 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 2.10 |
| lidar | baseline | 0.253 | 0.185 | 0.473 | 6.762 | 0.000 | n/a | 2.56 |
| lidar | oracle | 0.382 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 2.63 |

## Dropout Sweep Results (AP50, p=0.2)
### Plots
- camera best: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_plots/pose_dropout_1to10_camera_best_ap50.png
- camera stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_plots/pose_dropout_1to10_camera_stable_ap50.png
- lidar best: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_plots/pose_dropout_1to10_lidar_best_ap50.png
- lidar stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_plots/pose_dropout_1to10_lidar_stable_ap50.png
- Plots include baseline (pose_correction=none) and oracle (pose_correction=oracle_gt) curves.

### Summary (mean over 1–10)
| modality / strategy | method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera / best | v2xregpp | 0.148 | 0.142 | 0.398 | 275.382 | 15.202 | 32.2 | 2.08 |
| camera / best | freealign | 0.124 | 0.133 | 0.366 | 281.353 | 27.604 | 83.7 | 0.65 |
| camera / best | vips | 0.134 | 0.114 | 0.323 | 286.014 | 34.362 | 143.9 | 0.67 |
| camera / best | cbm | 0.093 | 0.027 | 0.092 | 319.319 | 68.716 | 232.0 | 0.68 |
| camera / stable | v2xregpp | 0.148 | 0.023 | 0.096 | 278.927 | 15.818 | 116.8 | 0.65 |
| camera / stable | freealign | 0.043 | 0.013 | 0.039 | 383.427 | 122.555 | 47.9 | 0.67 |
| camera / stable | vips | 0.058 | 0.012 | 0.036 | 383.339 | 140.570 | 131.4 | 0.69 |
| camera / stable | cbm | 0.039 | 0.011 | 0.034 | 365.605 | 116.253 | 187.6 | 0.67 |
| lidar / best | v2xregpp | 0.357 | 0.378 | 0.571 | 277.576 | 15.862 | 113.9 | 0.78 |
| lidar / best | freealign | 0.356 | 0.323 | 0.553 | 276.862 | 14.011 | 217.0 | 0.82 |
| lidar / best | vips | 0.276 | 0.113 | 0.311 | 287.640 | 35.815 | 101.7 | 0.85 |
| lidar / best | cbm | 0.289 | 0.021 | 0.059 | 328.815 | 77.422 | 192.4 | 0.94 |
| lidar / stable | v2xregpp | 0.276 | 0.167 | 0.301 | 277.954 | 12.396 | 107.7 | 0.90 |
| lidar / stable | freealign | 0.286 | 0.204 | 0.340 | 277.070 | 11.723 | 173.1 | 0.98 |
| lidar / stable | vips | 0.245 | 0.011 | 0.050 | 312.441 | 63.438 | 78.6 | 0.91 |
| lidar / stable | cbm | 0.353 | 0.002 | 0.007 | 364.183 | 86.139 | 145.1 | 0.98 |

### Bounds (baseline / oracle)
| modality | method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera | baseline | 0.108 | 0.149 | 0.359 | 22.545 | 21.222 | n/a | 1.31 |
| camera | oracle | 0.264 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 2.12 |
| lidar | baseline | 0.182 | 0.149 | 0.359 | 22.545 | 17.583 | n/a | 1.47 |
| lidar | oracle | 0.382 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 2.66 |

## Single-Agent Baseline (comm_range_override=0)
- Purpose: compare coop vs single-agent at the same detection range (range settings unchanged; only CAV selection is restricted).
- Settings: fusion_method=intermediate, pose_correction=none, comm_range_override=0, pos_std_list=0, rot_std_list=0, max_eval_samples=100.
- Noise target is non-ego; with only ego retained this is effectively a clean single-agent run.

| modality | ap30 | ap50 | ap70 | infer_fps |
| --- | --- | --- | --- | --- |
| camera | 0.1769 | 0.0618 | 0.0056 | 3.00 |
| lidar | 0.4993 | 0.4851 | 0.4206 | 3.96 |

## Single-Agent (single-ckpt, same detection range)
- Purpose: single-ckpt detector outputs under the same cav_lidar_range/lidar_range as coop.
- Settings: fusion_method=single, pose_correction=none, comm_range_override=100, pos_std_list=0, rot_std_list=0, max_eval_samples=100.
- Camera model: Pyramid_DAIR_m2_lsseff_single_2025_11_24_23_58_05 (bestval epoch 29).
- LiDAR model: Pyramid_DAIR_m1_pointpillars_single_2025_11_24_18_47_23 (bestval epoch 39).

| modality | ap30 | ap50 | ap70 | infer_fps |
| --- | --- | --- | --- | --- |
| camera | 0.2311 | 0.1017 | 0.0278 | 2.38 |
| lidar | 0.8258 | 0.8004 | 0.6656 | 3.79 |

## Evaluation Strategy
- Fairness: same max_eval_samples=100, same noise schedule, same noise target, same stage1 boxes, same comm range, same fusion method.
- Success rate: rel_success_at_m aggregated over 1/2/3/5/10 m from rel_error_stats.
- Accuracy: mean rel_trans_m and rel_yaw_deg from rel_error_stats.
- Efficiency: pose_time_ms from pose timing, infer_fps from end-to-end inference timing.
- Best-of: uses pose-compare-current to keep an update only if it improves the compare metric.
- Stable: uses EMA + step limits to smooth pose updates and prevent large jumps.
- Bounds: baseline uses pose_correction=none; oracle uses oracle_gt (dataset clean poses via lidar_pose_clean).
- No GT leakage: v2xregpp/freealign/vips/cbm consume only noisy poses + cached detections; GT extrinsics are used only in oracle_gt.

## Why Stable Often Underperforms Here
- EMA smoothing is applied across dataset order, which is not temporally continuous; updates from unrelated frames can bias the next sample.
- Step limits clip large corrections (1–10 m noise), so stable cannot fully recover when noise is large.
- Dropout reuses last noisy pose with confidence=0, which can propagate stale errors through EMA.
- Best-of explicitly rejects detrimental updates, while stable always applies some update.

## Comparison to 2026-01-29 Report
- This report uses 1–10 m sweeps (vs 0–5 m) and includes LiDAR (previous report was camera-only).
- Hyperparameters are re-tuned per method/strategy (min_matches/min_precision, thresholds, EMA/step caps).
- Dropout sweep is 1–10 m (vs 1–5 m).
- These differences explain why curves and means diverge from the 2026-01-29 report.

## Conclusions
- Camera noise sweep: v2xregpp best-of is strongest on mean AP50 and success rates; VIPS/FreeAlign are mid-tier; CBM trails.
- LiDAR noise sweep: v2xregpp/freealign best-of are strongest; VIPS and CBM trail in accuracy and success.
- Dropout sweep: v2xregpp best-of remains most reliable for camera; LiDAR best-of favors v2xregpp/freealign.
- Stable variants generally underperform best-of, especially on camera; tuning EMA/step caps helps but does not close the gap.

## Files Produced
- Noise sweep results: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_results.jsonl
- Noise sweep summary: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_summary.md
- Noise sweep plots: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_plots/*.png
- Dropout sweep results: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_results.jsonl
- Dropout sweep summary: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_summary.md
- Dropout sweep plots: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_plots/*.png
- Dropout sweep logs: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_logs/*.log
