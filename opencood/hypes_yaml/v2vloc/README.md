# V2VLoc (PGC + PASTAT) in HEAL

This folder contains a starter config to reproduce the V2VLoc pipeline described in `docs/v2vloc.pdf` inside HEAL/OpenCOOD:

1) Train Pose Generator with Confidence (PGC) on the localization subsets (Town1Loc/Town4Loc).
2) Run PGC on V2VDet to export per-frame predicted poses + confidences.
3) Train collaborative detection with PASTAT using the exported PGC poses.

## Notes

- The paper links `https://huggingface.co/datasets/linwk/V2VLoc`, but at the time of writing that HF repo has no data files. If you already have V2VLoc on disk (or you generated it with OpenCDA/CARLA), point `root_dir/validate_dir/test_dir` to your local paths.
- PASTAT in this repo is implemented as a fusion module over the PointPillars backbone, with CE+FSA before a V2X-ViT style transformer encoder.

## Commands

### 1) Train PGC

Prepare a yaml that can load Town1Loc/Town4Loc in OpenCOOD/OPV2V-style format (same folder layout as OPV2V). Then:

```bash
cd HEAL
python opencood/tools/train_v2vloc_pgc.py -y opencood/hypes_yaml/v2vloc/<your_townloc_yaml>.yaml --out_ckpt outputs/v2vloc/pgc.pth
```

### 2) Export PGC poses for V2VDet

```bash
cd HEAL
python opencood/tools/infer_v2vloc_pgc_pose.py -y opencood/hypes_yaml/v2vloc/lidar_pastat.yaml --pgc_ckpt outputs/v2vloc/pgc.pth --split train --out_json outputs/v2vloc/pgc_pose_train.json
python opencood/tools/infer_v2vloc_pgc_pose.py -y opencood/hypes_yaml/v2vloc/lidar_pastat.yaml --pgc_ckpt outputs/v2vloc/pgc.pth --split val --out_json outputs/v2vloc/pgc_pose_val.json
```

### 3) Train PASTAT

```bash
cd HEAL
python opencood/tools/train.py -y opencood/hypes_yaml/v2vloc/lidar_pastat.yaml
```

