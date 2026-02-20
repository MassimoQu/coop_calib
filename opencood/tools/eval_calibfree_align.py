#!/usr/bin/env python3
"""
Evaluate calibfree alignment quality (estimated ego->cav SE(2) warps) against GT pairwise_t_matrix.

This is meant as a quick sanity/ablation tool:
  - It runs a normal forward pass (so the model builds features as usual),
  - Then reads `model.fusion_net.last_calibfree_affine` populated by V2XViTFusion,
  - And compares it to GT (normalized affine from GT pairwise_t_matrix).

Notes
-----
- This tool requires the model config to use `fusion_method: v2xvit` and enable
  `model.args.v2xvit.calibfree.enabled: true`.
- For meaningful numbers you should load a trained checkpoint.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.transformation_utils import normalize_pairwise_tfm


def _wrap_angle_rad(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_affine(M: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    """
    Recover pixel-space yaw (radians) from the normalized affine used by affine_grid.

    In OpenCOOD normalization:
      a00 = cos(yaw)
      a10 = sin(yaw) * W / H
    => sin(yaw) = a10 * H / W
    """
    cos = M[..., 0, 0]
    sin = M[..., 1, 0] * (float(H) / float(W))
    return torch.atan2(sin, cos)


def _xy_m_from_affine(M: torch.Tensor, *, range_x_m: float, range_y_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalized translation (affine_grid coords) back to meters.
    """
    tx_m = M[..., 0, 2] * (float(range_x_m) / 2.0)
    ty_m = M[..., 1, 2] * (float(range_y_m) / 2.0)
    return tx_m, ty_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypes_yaml", "-y", required=True, type=str)
    ap.add_argument("--model_dir", type=str, default="", help="checkpoint dir (logs/xxx). optional.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_batches", type=int, default=50)
    args = ap.parse_args()

    hypes = yaml_utils.load_yaml(args.hypes_yaml, None)
    ds = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        ds,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=4,
        collate_fn=ds.collate_batch_train,
        shuffle=False,
        pin_memory=(args.device != "cpu"),
        drop_last=False,
        prefetch_factor=2,
    )

    model = train_utils.create_model(hypes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Avoid rare MKLDNN NHWC assertion failures on some conv shapes.
        torch.backends.mkldnn.enabled = False
    model.to(device)
    model.eval()

    if args.model_dir:
        _epoch, model = train_utils.load_saved_model(args.model_dir, model)
        model.to(device)
        model.eval()

    # Use lidar range from config to convert translation back to meters.
    lidar_range = hypes.get("cav_lidar_range") or hypes.get("model", {}).get("args", {}).get("lidar_range")
    if not lidar_range or len(lidar_range) < 6:
        raise ValueError("cannot infer cav_lidar_range from yaml")
    range_x = float(lidar_range[3]) - float(lidar_range[0])
    range_y = float(lidar_range[4]) - float(lidar_range[1])

    stats: Dict[str, float] = {"n_pairs": 0.0, "rte_sum": 0.0, "rre_sum": 0.0}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.num_batches and i >= int(args.num_batches):
                break
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            batch = train_utils.maybe_apply_pose_provider(batch, hypes)
            ego = batch["ego"]
            if "pairwise_t_matrix" not in ego:
                # We can still run the model, but cannot compute GT error.
                continue

            out = model(ego)
            _ = out

            # Grab predicted alignment from V2XViTFusion (debug hook).
            fusion = getattr(model, "fusion_net", None)
            pred = getattr(fusion, "last_calibfree_affine", None) if fusion is not None else None
            if pred is None:
                raise RuntimeError("model.fusion_net.last_calibfree_affine is None; is v2xvit.calibfree enabled?")

            # Feature map resolution used for fusion (from detection head output).
            Hf = int(out["cls_preds"].shape[-2])
            Wf = int(out["cls_preds"].shape[-1])

            # Build GT affine at this feature resolution.
            discrete_ratio = float(range_x) / float(Wf) if Wf > 0 else 1.0
            gt_affine = normalize_pairwise_tfm(ego["pairwise_t_matrix"], Hf, Wf, discrete_ratio)  # (B,L,L,2,3)

            record_len = ego["record_len"]
            B, L = int(gt_affine.shape[0]), int(gt_affine.shape[1])

            # Compare ego->cav (row 0) only.
            for b in range(B):
                N = int(record_len[b].item())
                if N <= 1:
                    continue
                M_gt = gt_affine[b, 0, :N]  # (N,2,3)
                M_pd = pred[b, :N]          # (N,2,3)

                # Skip ego itself at index 0.
                M_gt = M_gt[1:]
                M_pd = M_pd[1:]
                if M_gt.numel() == 0:
                    continue

                yaw_gt = _yaw_from_affine(M_gt, H=Hf, W=Wf)
                yaw_pd = _yaw_from_affine(M_pd, H=Hf, W=Wf)
                dyaw = _wrap_angle_rad(yaw_pd - yaw_gt).abs() * (180.0 / math.pi)

                tx_gt, ty_gt = _xy_m_from_affine(M_gt, range_x_m=range_x, range_y_m=range_y)
                tx_pd, ty_pd = _xy_m_from_affine(M_pd, range_x_m=range_x, range_y_m=range_y)
                dxy = torch.sqrt((tx_pd - tx_gt) ** 2 + (ty_pd - ty_gt) ** 2)

                stats["n_pairs"] += float(M_gt.shape[0])
                stats["rte_sum"] += float(dxy.sum().item())
                stats["rre_sum"] += float(dyaw.sum().item())

    n = max(1.0, stats["n_pairs"])
    print(f"pairs: {int(stats['n_pairs'])} | mean RTE(m): {stats['rte_sum']/n:.3f} | mean RRE(deg): {stats['rre_sum']/n:.3f}")


if __name__ == "__main__":
    main()
