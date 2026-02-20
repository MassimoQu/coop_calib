# FreeAlign reproduction handoff

Repository
- /home/qqxluca/projects/v2xreg_private/HEAL
- Reference FreeAlign repo used for comparison: /home/qqxluca/projects/_tmp_freealign_repo (GitHub: MediaBrain-SJTU/FreeAlign)

What differs from the upstream FreeAlign repo
- Upstream uses FreeAlign inside the dataloader (match_v7_with_detection) and filters with GT pose checks. That can inflate results.
  /home/qqxluca/projects/_tmp_freealign_repo/opencood/data_utils/datasets/intermediate_fusion_dataset.py:13
  /home/qqxluca/projects/_tmp_freealign_repo/freealign/match/match_v7_with_detection.py:20
- Upstream GNN is GraphSAGE with edge features (GNNModel), not EdgeGAT.
  /home/qqxluca/projects/_tmp_freealign_repo/freealign/models/graph/graphlearningmatching.py:106
- Here, FreeAlign paper version is re-implemented under opencood/pose and removed from dataloader; no GT filtering.
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/pose/freealign_paper.py:148
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/extrinsics/pose_correction/stage1_freealign.py:203
- Temporal alignment (time buffer search) is implemented and evaluated explicitly.
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/tools/compare_freealign_v2xregpp_pose.py:192
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/tools/eval_freealign_temporal_alignment.py:28
- OPV2V stage1 cache exports include sequence_id and frame_id for time indexing.
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/tools/pose_graph_pre_calc.py:492

FreeAlign variants implemented in HEAL
- NoGNN: paper MASS + LMEDS, no learned edge features.
- GNN-Lite: EdgeGATLite (lightweight edge feature learner).
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/pose/freealign_paper.py:148
- EdgeGAT: full edge attention (multi-head).
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/pose/freealign_paper.py:199
- EdgeGAT+Time: EdgeGAT + time buffer search.
- Repo baseline: match_v7_with_detection from upstream (for reference).

V2XReg++ version used
- Mode: initfree (no stable smoothing), no ICP, no occ-hint.
- DAIR config: configs/dair/midfusion/pipeline_midfusion_detection_occ.yaml
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_v2xregpp/config.yaml:29
- OPV2V config: configs/dair/detection/pipeline_detection_pp_ft.yaml
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_opv2v_v2xregpp/config.yaml:19

Key datasets and cache artifacts
- DAIR sync stage1 cache: opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json
- DAIR delay400 stage1 cache: opencood/logs/freealign_repro_dair_stage1_delay400/merged_stage1_val.json
- Delay400 uses data_info_path: cooperative/data_info_delay_400ms.json
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline_delay400/config.yaml:22
- OPV2V stage1 cache: opencood/logs/freealign_repro_opv2v_stage1/ (train/val dirs exist, stage1_boxes.json not written yet)

GNN checkpoints
- EdgeGAT (DAIR sync): opencood/logs/freealign_edgegat_dair_r0/edgegat_dair.pth
- EdgeGAT (DAIR delay400): opencood/logs/freealign_edgegat_dair_delay400/edgegat_dair_delay400.pth
- EdgeGATLite (OPV2V): opencood/logs/freealign_edgegatlite_dair_r0/edgegatlite_opv2v_epoch1.pth (+ epoch2/3)

Experiments run (outputs in opencood/logs)
- freealign_repro_summary/curve_summary.yaml (noise robustness curves).
- DAIR sync pose compare: freealign_repro_dair_pose_freealign_*.json
- DAIR delay400 detection: freealign_repro_dair_*_delay400/eval_intermediate_102.4_102.4_epoch27.yaml
- DAIR delay400 pose compare: freealign_repro_dair_pose_compare_delay400_paper.json

Metrics summary

1) Noise robustness (AP50, sigma 0 -> 8)
- DAIR: baseline 0.431 -> 0.274, FreeAlign GNN-Lite 0.431 -> 0.276, V2XReg++ 0.445 -> 0.385
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_summary/curve_summary.yaml:1
- OPV2V: baseline 0.959 -> 0.410, FreeAlign GNN-Lite 0.930 -> 0.882, V2XReg++ 0.958 -> 0.921
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_summary/curve_summary.yaml:1

2) DAIR sync pose compare (stage1 merged val)
- NoGNN: succ 0.086, TE_med 1.236m, RE_med 0.642
- GNN-Lite: succ 0.390, TE_med 1.232m, RE_med 0.700
- GNN-Lite(paper cfg): succ 0.473, TE_med 1.723m, RE_med 1.577
- EdgeGAT: succ 0.423, TE_med 1.553m, RE_med 1.273
- EdgeGAT+Time (sync data): succ 0.534, TE_med 10.977m, RE_med 4.994 (temporal search hurts on sync)
- FreeAlign-Repo (match_v7): succ 0.488, TE_med 1.317m, RE_med 0.857
- V2XReg++: succ 0.771, TE_med 2.181m, RE_med 1.877
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_freealign_egat.json:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_freealign_gnn.json:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_freealign_gnn_papercfg.json:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_freealign_nognn.json:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_freealign_egat_time.json:1

3) DAIR delay400 detection (AP30/AP50/AP70)
- baseline: 0.4688 / 0.3453 / 0.2094
- V2XReg++: 0.4830 / 0.3738 / 0.2282
- FreeAlign EdgeGAT+Time: 0.4570 / 0.3254 / 0.1903
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_baseline_delay400/eval_intermediate_102.4_102.4_epoch27.yaml:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_v2xregpp_delay400/eval_intermediate_102.4_102.4_epoch27.yaml:1
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_freealign_paper_egat_delay400/eval_intermediate_102.4_102.4_epoch27.yaml:1

4) DAIR delay400 pose compare
- FreeAlign paper: succ 0.717, TE_med 2.835m, RE_med 2.184
- FreeAlign repo: succ 0.544, TE_med 1.626m, RE_med 1.359
- V2XReg++: succ 0.922, TE_med 3.130m, RE_med 2.174
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_compare_delay400_paper.json:1

5) DAIR delay400 temporal alignment accuracy (paper config)
- exact_acc 0.314, within1_acc 0.699, avg_abs_offset 1.523 frames (~152.3 ms)
  /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_pose_compare_delay400_paper.json:1

High-level conclusions
- OPV2V noise: FreeAlign GNN-Lite and V2XReg++ stay strong, baseline collapses.
- DAIR noise: V2XReg++ improves over FreeAlign GNN-Lite; FreeAlign GNN-Lite ~ baseline.
- Temporal alignment search should be used only on delayed data; on sync data it hurts TE/RE.
- Paper-level temporal accuracy (22.8ms / 45.6ms) is not reproduced; current delay400 avg offset is ~152ms.
- Upstream FreeAlign code mixes GT filtering in training/inference, which can inflate results.

Pending / not finished
- OPV2V stage1 cache export with sequence_id/frame_id not completed (no stage1_boxes.json yet).
- OPV2V temporal alignment eval (Table IV analog) not run.
- OPV2V latency deviation (Table III analog) not run; would require synthetic frame offsets.

Commands for continuation
- Export OPV2V stage1 cache (val):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/qqxluca/projects/v2xreg_private/HEAL     /home/qqxluca/miniconda3/envs/heal/bin/python -m opencood.tools.pose_graph_pre_calc     --hypes_yaml opencood/logs/freealign_repro_opv2v_baseline/config.yaml     --stage1_checkpoint opencood/logs/freealign_repro_opv2v_baseline/net_epoch_bestval_at27.pth     --output_dir opencood/logs/freealign_repro_opv2v_stage1     --splits val --train_eval_mode

- OPV2V temporal pose compare (FreeAlign paper):
  PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=0 /home/qqxluca/miniconda3/envs/heal/bin/python     -m opencood.tools.compare_freealign_v2xregpp_pose     --stage1_cache opencood/logs/freealign_repro_opv2v_stage1/val/stage1_boxes.json     --freealign_paper_time_buffer 4 --freealign_paper_time_stride 1     --freealign_paper_use_gnn --freealign_paper_gnn_type lite     --freealign_paper_ckpt_path /home/qqxluca/v2xreg_private/opencood/logs/freealign_edgegatlite_dair_r0/edgegatlite_opv2v_epoch1.pth     --freealign_paper_device cuda:0     --out opencood/logs/freealign_repro_opv2v_pose_compare_time.json

- OPV2V temporal alignment accuracy:
  /home/qqxluca/miniconda3/envs/heal/bin/python -m opencood.tools.eval_freealign_temporal_alignment     --delay_records opencood/logs/freealign_repro_opv2v_pose_compare_time.json     --sync_stage1 opencood/logs/freealign_repro_opv2v_stage1/val/stage1_boxes.json     --method freealign_paper --sample_interval_ms 100
