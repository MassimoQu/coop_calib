# HEAL Pose Alignment + Coop Benchmark (Full Sweep)

## Purpose
- Provide a fair, full-test benchmark for pose solvers in HEAL (v2xregpp/freealign/vips/cbm).
- Compare best-of vs stable strategies, plus lower/upper bounds and single-agent baselines.
- Report pose success/accuracy/latency and downstream cooperative perception.

## Experiment Date
- 2026-02-05 16:16:37 +0800

## Evaluation Strategy (Detailed)
- Two sweeps on full test set: noise sweep (pos_std 1..10m) and dropout sweep (same noise + pose_dropout_prob=0.2).
- Noise is injected into non-ego extrinsics (paired sweep) with rot_std list: camera 1..10 deg, lidar 0 deg.
- Metrics are averaged across 10 noise points: detection AP (AP30/AP50/AP70), pose success@{1,2,3,5,10}m, rel_trans, rel_yaw, pose solve time, infer FPS.
- Lower bounds: (1) single-agent perception with same range, (2) cooperative init pose only (pose_correction=none).
- Upper bound: cooperative perception with oracle_gt extrinsics (GT only for oracle runs).
- No GT extrinsics are used for any method other than the oracle_gt upper bound.

## Experiment Setup
- Repo: /home/qqxluca/projects/v2xreg_private
- Python: /home/qqxluca/.micromamba/envs/heal/bin/python
- full test set (no max_eval_samples)
- sweep_mode: paired; noise_target: non-ego; comm_range_override: 100
- pose_timing enabled; num_workers: 0; save_vis_interval: 100000000

### Camera Setup (V2XViT)
- Model dir: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38
- Stage1 boxes: /home/qqxluca/projects/v2xreg_private/data/DAIR-V2X/detected/camera_v2xvit_stage1/stage1_boxes.json
- rot_std_list: 1..10 deg

### LiDAR Setup (FreeAlign Repro)
- Model dir: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline
- Stage1 boxes: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json
- rot_std_list: 0 deg

### Single-Agent Baselines (Aligned, Comm=0)
- camera: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38 (fusion_method=intermediate, comm_range_override=0)
- lidar: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline (fusion_method=intermediate, comm_range_override=0)

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

## Noise Sweep Results (AP50, 1-10 m)
### Plots
- camera best: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_camera_best_ap50.png
- camera stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_camera_stable_ap50.png
- lidar best: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_lidar_best_ap50.png
- lidar stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_lidar_stable_ap50.png
- camera all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_camera_all_styled_ap50.png
- lidar all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_lidar_all_styled_ap50.png
- camera + lidar all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_sweep_1to10_full_plots/pose_sweep_1to10_full_all_modalities_styled_ap50.png
- plots include baseline (pose_correction=none) and oracle (pose_correction=oracle_gt).
- styled plots: same color = same method; line style = best/stable/bounds/single; marker = modality.


### camera / best
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.124 | 0.186 | 0.464 | 6.848 | 4.326 | 140.5 | 0.53 |
| freealign | 0.114 | 0.171 | 0.429 | 11.425 | 10.657 | 82.4 | 0.54 |
| vips | 0.114 | 0.149 | 0.374 | 19.642 | 20.699 | 161.6 | 0.53 |
| cbm | 0.098 | 0.038 | 0.091 | 55.764 | 69.432 | 254.0 | 0.54 |

### camera / stable
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.106 | 0.014 | 0.051 | 20.361 | 14.608 | 144.3 | 0.52 |
| freealign | 0.068 | 0.011 | 0.029 | 58.395 | 20.610 | 60.0 | 0.56 |
| vips | 0.046 | 0.011 | 0.028 | 137.707 | 162.297 | 147.7 | 0.57 |
| cbm | 0.043 | 0.011 | 0.029 | 118.166 | 114.788 | 195.8 | 0.56 |

### camera / bounds
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.123 | 0.182 | 0.462 | 6.857 | 4.385 | n/a | 0.60 |
| oracle | 0.223 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 0.83 |

### lidar / best
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.394 | 0.477 | 0.688 | 5.170 | 1.761 | 144.3 | 0.63 |
| freealign | 0.382 | 0.413 | 0.633 | 5.335 | 1.105 | 193.8 | 0.63 |
| vips | 0.298 | 0.154 | 0.387 | 19.249 | 16.698 | 155.6 | 0.58 |
| cbm | 0.304 | 0.027 | 0.062 | 64.274 | 74.710 | 242.2 | 0.59 |

### lidar / stable
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.313 | 0.229 | 0.388 | 8.645 | 0.937 | 177.7 | 0.60 |
| freealign | 0.311 | 0.214 | 0.389 | 8.483 | 0.889 | 179.0 | 0.63 |
| vips | 0.250 | 0.005 | 0.016 | 67.668 | 87.883 | 128.2 | 0.61 |
| cbm | 0.379 | 0.004 | 0.009 | 118.354 | 88.892 | 195.8 | 0.67 |

### lidar / bounds
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.300 | 0.182 | 0.462 | 6.857 | 0.000 | n/a | 0.59 |
| oracle | 0.431 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 0.62 |


### Best Strategy by mean AP50 (Noise Sweep)
| modality | method | best strategy | mean_ap50 | success@5m | rel_trans_m | rel_yaw_deg |
| --- | --- | --- | --- | --- | --- | --- |
| camera | cbm | best | 0.098 | 0.091 | 55.764 | 69.432 |
| camera | freealign | best | 0.114 | 0.429 | 11.425 | 10.657 |
| camera | v2xregpp | best | 0.124 | 0.464 | 6.848 | 4.326 |
| camera | vips | best | 0.114 | 0.374 | 19.642 | 20.699 |
| lidar | cbm | stable | 0.379 | 0.009 | 118.354 | 88.892 |
| lidar | freealign | best | 0.382 | 0.633 | 5.335 | 1.105 |
| lidar | v2xregpp | best | 0.394 | 0.688 | 5.170 | 1.761 |
| lidar | vips | best | 0.298 | 0.387 | 19.249 | 16.698 |

### Best Strategy by success@5m (Noise Sweep)
| modality | method | best strategy | mean_ap50 | success@5m | rel_trans_m | rel_yaw_deg |
| --- | --- | --- | --- | --- | --- | --- |
| camera | cbm | best | 0.098 | 0.091 | 55.764 | 69.432 |
| camera | freealign | best | 0.114 | 0.429 | 11.425 | 10.657 |
| camera | v2xregpp | best | 0.124 | 0.464 | 6.848 | 4.326 |
| camera | vips | best | 0.114 | 0.374 | 19.642 | 20.699 |
| lidar | cbm | best | 0.304 | 0.062 | 64.274 | 74.710 |
| lidar | freealign | best | 0.382 | 0.633 | 5.335 | 1.105 |
| lidar | v2xregpp | best | 0.394 | 0.688 | 5.170 | 1.761 |
| lidar | vips | best | 0.298 | 0.387 | 19.249 | 16.698 |

## Dropout Sweep Results (AP50, 1-10 m, pose_dropout_prob=0.2)
### Plots
- camera best: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_camera_best_ap50.png
- camera stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_camera_stable_ap50.png
- lidar best: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_lidar_best_ap50.png
- lidar stable: /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_lidar_stable_ap50.png
- camera all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_camera_all_styled_ap50.png
- lidar all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_lidar_all_styled_ap50.png
- camera + lidar all (styled): /home/qqxluca/projects/v2xreg_private/outputs/pose_dropout_1to10_full_plots/pose_dropout_1to10_full_all_modalities_styled_ap50.png
- plots include baseline (pose_correction=none) and oracle (pose_correction=oracle_gt).
- styled plots: same color = same method; line style = best/stable/bounds/single; marker = modality.


### camera / best
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.117 | 0.145 | 0.380 | 300.895 | 17.428 | 132.9 | 0.53 |
| freealign | 0.109 | 0.133 | 0.350 | 304.788 | 23.351 | 79.9 | 0.54 |
| vips | 0.109 | 0.118 | 0.308 | 310.432 | 31.101 | 159.7 | 0.54 |
| cbm | 0.095 | 0.027 | 0.069 | 339.855 | 72.577 | 263.3 | 0.55 |

### camera / stable
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.119 | 0.017 | 0.074 | 305.014 | 18.196 | 139.4 | 0.53 |
| freealign | 0.036 | 0.009 | 0.024 | 400.833 | 123.899 | 60.1 | 0.57 |
| vips | 0.051 | 0.009 | 0.024 | 402.571 | 141.227 | 149.4 | 0.57 |
| cbm | 0.038 | 0.009 | 0.024 | 384.203 | 114.536 | 184.7 | 0.55 |

### camera / bounds
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.098 | 0.143 | 0.369 | 21.276 | 21.263 | n/a | 0.62 |
| oracle | 0.223 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 0.83 |

### lidar / best
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.391 | 0.385 | 0.561 | 300.265 | 17.620 | 142.2 | 0.64 |
| freealign | 0.381 | 0.323 | 0.509 | 299.796 | 16.565 | 202.0 | 0.64 |
| vips | 0.314 | 0.122 | 0.319 | 310.871 | 27.531 | 149.1 | 0.60 |
| cbm | 0.313 | 0.019 | 0.047 | 346.977 | 77.124 | 244.2 | 0.60 |

### lidar / stable
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2xregpp | 0.326 | 0.188 | 0.330 | 302.058 | 14.691 | 170.8 | 0.61 |
| freealign | 0.324 | 0.166 | 0.307 | 302.584 | 14.537 | 179.5 | 0.63 |
| vips | 0.265 | 0.005 | 0.024 | 349.237 | 88.240 | 119.8 | 0.63 |
| cbm | 0.342 | 0.002 | 0.005 | 383.788 | 89.312 | 191.9 | 0.67 |

### lidar / bounds
| method | mean_ap50 | success@2m | success@5m | rel_trans_m | rel_yaw_deg | pose_time_ms | infer_fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.229 | 0.143 | 0.369 | 21.276 | 17.770 | n/a | 0.60 |
| oracle | 0.431 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0 | 0.64 |


### Best Strategy by mean AP50 (Dropout Sweep)
| modality | method | best strategy | mean_ap50 | success@5m | rel_trans_m | rel_yaw_deg |
| --- | --- | --- | --- | --- | --- | --- |
| camera | cbm | best | 0.095 | 0.069 | 339.855 | 72.577 |
| camera | freealign | best | 0.109 | 0.350 | 304.788 | 23.351 |
| camera | v2xregpp | stable | 0.119 | 0.074 | 305.014 | 18.196 |
| camera | vips | best | 0.109 | 0.308 | 310.432 | 31.101 |
| lidar | cbm | stable | 0.342 | 0.005 | 383.788 | 89.312 |
| lidar | freealign | best | 0.381 | 0.509 | 299.796 | 16.565 |
| lidar | v2xregpp | best | 0.391 | 0.561 | 300.265 | 17.620 |
| lidar | vips | best | 0.314 | 0.319 | 310.871 | 27.531 |

### Best Strategy by success@5m (Dropout Sweep)
| modality | method | best strategy | mean_ap50 | success@5m | rel_trans_m | rel_yaw_deg |
| --- | --- | --- | --- | --- | --- | --- |
| camera | cbm | best | 0.095 | 0.069 | 339.855 | 72.577 |
| camera | freealign | best | 0.109 | 0.350 | 304.788 | 23.351 |
| camera | v2xregpp | best | 0.117 | 0.380 | 300.895 | 17.428 |
| camera | vips | best | 0.109 | 0.308 | 310.432 | 31.101 |
| lidar | cbm | best | 0.313 | 0.047 | 346.977 | 77.124 |
| lidar | freealign | best | 0.381 | 0.509 | 299.796 | 16.565 |
| lidar | v2xregpp | best | 0.391 | 0.561 | 300.265 | 17.620 |
| lidar | vips | best | 0.314 | 0.319 | 310.871 | 27.531 |

## Single-Agent Baselines (Aligned, Comm=0)
| modality | noise_mean_ap50 | dropout_mean_ap50 | path |
| --- | --- | --- | --- |
| camera single_comm0 | 0.052 | 0.040 | /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38/AP030507_none_sweep10m_camera_single_comm0_full.yaml |
| lidar single_comm0 | 0.500 | 0.388 | /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline/AP030507_none_sweep10m_lidar_single_comm0_full.yaml |

## Benchmark Master Table
- Best-per-method selection (by mean AP50) is summarized in: /home/qqxluca/projects/v2xreg_private/HEAL/docs/pose_alignment_benchmark_master.md
- Fairness review (3-pass) is summarized in: /home/qqxluca/projects/v2xreg_private/HEAL/docs/pose_alignment_fairness_review.md

## Notes & Analysis
- Camera (noise sweep): best-of v2xregpp slightly edges baseline (0.124 vs 0.123 AP50); gains are small because stage1 box noise dominates and pose solvers struggle with sparse camera cues.
- LiDAR (noise sweep): v2xregpp best-of is top (0.394 AP50, success@5m=0.688). Freealign best-of is close. VIPS/CBM lag behind.
- Stable strategies generally reduce success@{2m,5m} because EMA/max-step caps under-correct large noise; they trade stability for accuracy.
- CBM stable on LiDAR yields high AP50 but near-zero pose success; treat with caution (pose accuracy collapses despite detection score).
- Oracle gap: camera improves from 0.123->0.223 AP50, lidar from 0.300->0.431 AP50, indicating remaining upside from perfect extrinsics.
- Dropout sweep: pose error metrics explode (rel_trans ~300m) across methods, but detection AP50 degrades moderately; best-of v2xregpp/freealign remain strongest.
- Single_comm0 LiDAR (0.500 AP50) still exceeds some cooperative methods but is below oracle, narrowing the gap vs the previous unaligned single baseline.

## Alignment Check (Model/Stage1/Fusion)
- Coop LiDAR baseline config: `HEAL/opencood/logs/freealign_repro_dair_baseline/config.yaml:1` uses `fusion.core_method=intermediateheter` and `model.core_method=heter_model_baseline`, with `input_source=[lidar]`.
- Coop camera baseline config: `HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38/config.yaml:1` uses heter baseline V2XViT, with `fusion.core_method=intermediateheter`.
- Single_comm0 runs reuse the *same coop checkpoints* and *same fusion_method (intermediate)* with `comm_range_override=0`, so architecture and detector are fully aligned.
- Remaining differences are only the necessary ones: communication disabled (no partners), and no pose correction (pose_correction=none).
- Stage1 caches are still used for pose solvers in coop runs; single_comm0 runs are direct inference without pose correction, which is expected and fair for the lower bound.
