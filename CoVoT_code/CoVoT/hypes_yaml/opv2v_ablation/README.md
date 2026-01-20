OPV2V ablation configs

Configs in this folder are meant to make backbone/fusion comparisons easier.
Update `root_dir` and `validate_dir` for your dataset location before running.

Configs
- opv2v_point_pillar_v2vnet.yaml: point pillar + V2VNet (baseline, weak backbone)
- opv2v_point_pillar_fcooper.yaml: point pillar + F-Cooper (baseline, weak backbone; uses proj_first)
- opv2v_point_pillar_v2xvit.yaml: point pillar + V2X-ViT (baseline, weak backbone)
- opv2v_point_pillar_attfuse.yaml: point pillar + AttFuse (baseline, weak backbone)
- opv2v_point_pillar_where2comm.yaml: point pillar + Where2Comm (baseline, weak backbone)
- opv2v_voxelnext_v2xast_hybrid_strong.yaml: VoxelNeXt + AST (strong backbone baseline)
- opv2v_voxelnext_where2comm_strong.yaml: VoxelNeXt + Where2Comm (strong backbone baseline)
- opv2v_voxelnext_covot_hybrid_weak.yaml: CoVoT with coarse voxel size (weaker backbone variant)

Notes
- `backbone_fix` is set to `true` in the CoVoT/V2XAST VoxelNeXt configs; the Where2Comm VoxelNeXt config uses `false` by default.
  Flip it if you want to freeze or fully train the backbone.
- `opv2v_point_pillar_fcooper.yaml` uses `proj_first: true` since the fusion does not warp features.
- `max_voxel_test` is set for full-res runs. On smaller GPUs, reduce it to avoid OOM.

Example commands
- Train: `python tools/train.py --hypes_yaml hypes_yaml/opv2v_ablation/opv2v_point_pillar_v2vnet.yaml`
- Eval:  `python tools/inference.py --model_dir /path/to/run --fusion_method intermediate`
  (use `--fusion_method hybrid` for the VoxelNeXt hybrid configs)
