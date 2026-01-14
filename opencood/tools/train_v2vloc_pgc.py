#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.pose.pgc import PGCNet, rsd_downsample
from opencood.utils.transformation_utils import pose_to_tfm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V2VLoc Pose Generator with Confidence (PGC) on OpenCOOD-style datasets.")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hypes_yaml", "-y", type=str, help="HEAL/OpenCOOD yaml for dataset paths.")
    g.add_argument("--model_dir", type=str, help="HEAL checkpoint folder containing config.yaml (dataset).")

    p.add_argument("--out_ckpt", type=str, required=True, help="Output checkpoint path for PGC.")
    p.add_argument("--train_split", action="store_true", help="Use training split (default).")
    p.add_argument("--val_split", action="store_true", help="Use validation split instead of training split.")
    p.add_argument("--max_samples", type=int, default=0, help="Limit number of frames (0=all).")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1200.0)
    p.add_argument("--seed", type=int, default=303)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--num_points", type=int, default=4096)
    p.add_argument("--rsd_voxel_size", type=float, default=0.2)
    p.add_argument("--feat_dim", type=int, default=256)

    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class PGCFrameDataset(Dataset):
    def __init__(self, hypes: dict, *, train: bool, max_samples: int, num_points: int, rsd_voxel_size: float) -> None:
        super().__init__()
        self.dataset = build_dataset(hypes, visualize=False, train=train)
        self.max_samples = int(max_samples) if int(max_samples) > 0 else len(self.dataset)
        self.num_points = int(num_points)
        self.rsd_voxel_size = float(rsd_voxel_size)

    def __len__(self) -> int:
        return int(self.max_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.dataset.retrieve_base_data(int(idx))
        ego = None
        for _, cav in base.items():
            if bool(cav.get("ego", False)):
                ego = cav
                break
        if ego is None:
            ego = list(base.values())[0]

        pts = ego["lidar_np"].astype(np.float32)
        pts = rsd_downsample(pts, num_points=int(self.num_points), voxel_size=float(self.rsd_voxel_size))
        pose = np.asarray(ego["params"]["lidar_pose"], dtype=np.float32).reshape(6)
        return torch.from_numpy(pts), torch.from_numpy(pose)


def _world_coords(points_xyzi: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Args:
        points_xyzi: (B,N,4)
        pose: (B,6) [x,y,z,roll,yaw,pitch] degrees
    Returns:
        world_xyz: (B,N,3)
    """
    pts = points_xyzi[..., :3]
    B, N, _ = pts.shape
    ones = torch.ones((B, N, 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)  # (B,N,4)
    T = pose_to_tfm(pose)  # (B,4,4)
    out = torch.bmm(T, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
    return out


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    if args.model_dir:
        hypes = yaml_utils.load_yaml(str(Path(args.model_dir) / "config.yaml"), None)
    else:
        hypes = yaml_utils.load_yaml(args.hypes_yaml, None)

    train = True
    if args.val_split:
        train = False

    ds = PGCFrameDataset(
        hypes,
        train=train,
        max_samples=int(args.max_samples),
        num_points=int(args.num_points),
        rsd_voxel_size=float(args.rsd_voxel_size),
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
        pin_memory=True,
    )

    device = torch.device(str(args.device))
    model = PGCNet(in_dim=4, feat_dim=int(args.feat_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    model.train()
    for epoch in range(int(args.epochs)):
        running = 0.0
        n = 0
        for step, batch in enumerate(loader):
            points, pose = batch
            points = points.to(device, non_blocking=True).float()
            pose = pose.to(device, non_blocking=True).float()

            y_gt = _world_coords(points, pose)  # (B,N,3)
            y_pred, eps_pred = model(points)

            u = (y_pred - y_gt).abs().mean(dim=(1, 2))  # (B,)
            loss = (u + (u - eps_pred).abs()).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu().item())
            n += 1
            if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
                print(f"[pgc][epoch {epoch+1}/{int(args.epochs)}] step={step+1} avg_loss={running/max(n,1):.6f}")

        print(f"[pgc] epoch {epoch+1}/{int(args.epochs)} done: avg_loss={running/max(n,1):.6f}")

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, str(out_ckpt))
    print(f"[pgc] saved: {out_ckpt}")


if __name__ == "__main__":
    main()

