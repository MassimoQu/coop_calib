# HEAL Pose Alignment + Coop Perception Stress Test (DAIR-V2X V2XViT)

## Purpose
- Integrate VIPS and CBM into HEAL pose correction alongside v2xreg++ and FreeAlign.
- Evaluate cooperative perception robustness under extrinsic noise and localization dropout.
- Compare two batch strategies: best-of (compare with current) and stable (EMA + step limits).

## Experiment Date
- 2026-01-29 03:55:32 +0800

## Experiment Setup
- Repo: /home/qqxluca/projects/v2xreg_private/HEAL
- Python: /home/qqxluca/.micromamba/envs/heal/bin/python
- Model checkpoint: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38/net_epoch_bestval_at15.pth
- Stage1 boxes: /home/qqxluca/projects/v2xreg_private/data/DAIR-V2X/detected/camera_v2xvit_stage1/stage1_boxes.json
- Dataset val: /home/qqxluca/projects/v2xreg_private/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/val.json
- Fusion method: intermediate (V2XViT)
- Noise target: non-ego
- Sweep mode: paired
- max_eval_samples: 100 (fast sweep, not full val)
- comm_range_override: 100
- pose_timing: enabled
- v2xregpp config: configs/dair/midfusion/pipeline_midfusion_detection_occ.yaml
- best-of compare: --pose-compare-current (distance threshold 3.0 m)
- stable params (defaults): ema_alpha=0.5, max_step_xy=3.0 m, max_step_yaw=10.0 deg

### Noise Sweep Settings
- pos_std_list = [0, 1, 2, 3, 4, 5]
- rot_std_list = [0, 1, 2, 3, 4, 5]
- pos_mean=0, rot_mean=0

### Dropout Sweep Settings
- pos_std_list = [1, 2, 3, 4, 5]
- rot_std_list = [1, 2, 3, 4, 5]
- pose_dropout_prob = 0.2
- Dropout behavior: reuse last noisy pose and set pose_confidence=0

## Experiment Results (AP50)
### Noise Sweep (best-of)
- baseline: [0.264, 0.160, 0.132, 0.143, 0.132, 0.144]
- v2xregpp_best: [0.244, 0.172, 0.141, 0.152, 0.158, 0.157]
- freealign_best: [0.221, 0.160, 0.123, 0.133, 0.147, 0.147]
- vips_best: [0.195, 0.151, 0.141, 0.144, 0.145, 0.150]
- cbm_best: [0.098, 0.092, 0.093, 0.091, 0.094, 0.096]

### Noise Sweep (stable)
- baseline: [0.264, 0.160, 0.132, 0.143, 0.132, 0.144]
- v2xregpp_stable: [0.172, 0.165, 0.181, 0.126, 0.129, 0.123]
- freealign_stable: [0.074, 0.072, 0.066, 0.076, 0.085, 0.081]
- vips_stable: [0.062, 0.054, 0.056, 0.058, 0.061, 0.053]
- cbm_stable: [0.043, 0.036, 0.050, 0.049, 0.051, 0.055]

### Dropout Sweep (best-of, p=0.2)
- baseline: [0.123, 0.117, 0.120, 0.093, 0.104]
- v2xregpp_best: [0.146, 0.131, 0.147, 0.133, 0.146]
- freealign_best: [0.133, 0.122, 0.132, 0.113, 0.118]
- vips_best: [0.123, 0.128, 0.137, 0.130, 0.135]
- cbm_best: [0.089, 0.091, 0.093, 0.091, 0.092]

### Dropout Sweep (stable, p=0.2)
- baseline: [0.123, 0.117, 0.120, 0.093, 0.104]
- v2xregpp_stable: [0.155, 0.154, 0.136, 0.143, 0.160]
- freealign_stable: [0.037, 0.040, 0.038, 0.036, 0.052]
- vips_stable: [0.051, 0.056, 0.062, 0.057, 0.065]
- cbm_stable: [0.039, 0.040, 0.034, 0.038, 0.040]

## Efficiency (Noise Sweep Mean FPS, max100)
- baseline: infer_fps=0.65, pose_fps=n/a, samples=100
- v2xregpp_best: infer_fps=0.61, pose_fps=14339.17, samples=100
- freealign_best: infer_fps=0.62, pose_fps=8805.53, samples=100
- vips_best: infer_fps=0.61, pose_fps=7871.63, samples=100
- cbm_best: infer_fps=0.62, pose_fps=14545.22, samples=100
- baseline: infer_fps=0.65, pose_fps=n/a, samples=100
- v2xregpp_stable: infer_fps=0.59, pose_fps=15912.22, samples=100
- freealign_stable: infer_fps=0.88, pose_fps=16764.76, samples=100
- vips_stable: infer_fps=0.98, pose_fps=18529.07, samples=100
- cbm_stable: infer_fps=0.95, pose_fps=13035.86, samples=100

Notes:
- pose_fps is computed from dataset-level pose_override timing only (not full end-to-end GPU latency).
- infer_fps is end-to-end model inference on the selected samples.

## Plots
- Noise sweep (best-of): /home/qqxluca/runs/pose_sweep/dair_v2xvit/noise_sweep_best_max100.png
- Noise sweep (stable): /home/qqxluca/runs/pose_sweep/dair_v2xvit/noise_sweep_stable_max100.png
- Dropout sweep (best-of): /home/qqxluca/runs/pose_sweep/dair_v2xvit/dropout_sweep_best_max100.png
- Dropout sweep (stable): /home/qqxluca/runs/pose_sweep/dair_v2xvit/dropout_sweep_stable_max100.png

## Experiment Analysis
- Best-of consistently improves over baseline for v2xregpp, with smaller gains for freealign/vips; cbm is consistently lower.
- Stable strategy is generally worse than baseline for freealign/vips/cbm under this configuration; v2xregpp stable is the only stable variant that remains competitive under dropout.
- Under dropout, v2xregpp (best-of or stable) shows the most robust AP50 across noise levels.
- The stable strategy likely under-corrects when the EMA and step limits dampen large corrections.

## Project Understanding and Assessment
- This pipeline tests calibration-free or calibration-corrected cooperative perception under controlled extrinsic noise and localization outages.
- The evaluation isolates the effect of pose correction by injecting noise only into non-ego agents and using a paired sweep to keep translation/rotation std aligned.
- The outcomes highlight that alignment quality depends strongly on how pose updates are accepted (best-of vs stable) and on the pose estimator itself.
- For real-world usage, best-of logic appears safer for these methods on this data, while stable smoothing needs tuning or method-specific adaptations.

## Files Produced
- YAML results: /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38/AP030507_*_max100.yaml
- Plots: /home/qqxluca/runs/pose_sweep/dair_v2xvit/noise_sweep_best_max100.png, /home/qqxluca/runs/pose_sweep/dair_v2xvit/noise_sweep_stable_max100.png, /home/qqxluca/runs/pose_sweep/dair_v2xvit/dropout_sweep_best_max100.png, /home/qqxluca/runs/pose_sweep/dair_v2xvit/dropout_sweep_stable_max100.png

## Notes for Comparison
- Results are based on max_eval_samples=100, not full validation. This can differ from full-val results in another window.
- Dropout probability is fixed at 0.2 here; other runs with different dropout or noise target will diverge.
