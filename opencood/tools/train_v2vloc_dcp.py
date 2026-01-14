#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a lightweight learned LiDAR registration module (DCP-style) for V2VLoc-like pose refinement.

Training target:
  - We simulate a noisy initial cav->ego transform: T_init = T_noise @ T_gt
  - The network sees ego points (ego frame) and cav points transformed by T_init (ego frame),
    and predicts a delta transform to undo the noise: T_delta ~= inv(T_noise)
  - Final transform at inference: T_pred = T_delta @ T_init

This aligns with a practical "noisy pose -> LiDAR-based refinement" pipeline and can be used
as `--calib_method v2vloc_dcp` in `opencood/tools/inference_online_calib.py`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.pose.v2vloc_dcp import DCP2D
from opencood.utils.transformation_utils import get_relative_transformation


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v2vloc_dcp (learned point registration) on OPV2V/DAIR via HEAL datasets.")
    p.add_argument("--model_dir", type=str, required=True, help="Any HEAL checkpoint folder with a usable config.yaml for dataset.")
    p.add_argument("--out_ckpt", type=str, required=True, help="Output checkpoint path for v2vloc_dcp.")
    p.add_argument("--train_split", action="store_true", help="Use training split (default).")
    p.add_argument("--val_split", action="store_true", help="Use validation/test split instead of training split.")
    p.add_argument("--max_samples", type=int, default=0, help="Limit number of frames (0 = all).")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=303)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--coord_scale", type=float, default=50.0)
    p.add_argument("--feat_dim", type=int, default=256)

    # Noise simulation (training distribution for the initial guess).
    p.add_argument("--noise_pos_std", type=float, default=8.0)
    p.add_argument("--noise_rot_std", type=float, default=8.0, help="Yaw noise std (deg).")

    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_se2(dx: float, dy: float, yaw_deg: float) -> np.ndarray:
    th = float(np.deg2rad(yaw_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(dx)
    T[1, 3] = float(dy)
    return T


def _apply_T(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (np.asarray(T, dtype=np.float32) @ pts_h.T).T[:, :3]
    return out


def _sample_xy(points_xyz: np.ndarray, num: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((int(num), 2), dtype=np.float32)
    if int(num) <= 0:
        return pts[:, :2]
    if pts.shape[0] >= int(num):
        idx = np.random.choice(pts.shape[0], int(num), replace=False)
    else:
        idx = np.random.choice(pts.shape[0], int(num), replace=True)
    return pts[idx, :2].astype(np.float32)


class V2VLocPairDataset(Dataset):
    def __init__(
        self,
        hypes: dict,
        *,
        train: bool,
        max_samples: int,
        num_points: int,
        coord_scale: float,
        noise_pos_std: float,
        noise_rot_std: float,
    ) -> None:
        super().__init__()
        self.dataset = build_dataset(hypes, visualize=False, train=train)
        self.max_samples = int(max_samples) if int(max_samples) > 0 else len(self.dataset)
        self.num_points = int(num_points)
        self.coord_scale = float(coord_scale) if float(coord_scale) > 0 else 1.0
        self.noise_pos_std = float(noise_pos_std)
        self.noise_rot_std = float(noise_rot_std)
        self.lidar_range = hypes.get("cav_lidar_range") or hypes.get("preprocess", {}).get("cav_lidar_range")

    def __len__(self) -> int:
        return int(self.max_samples)

    def _lookup(self, base: dict, key):
        if key in base:
            return base[key]
        if isinstance(key, str):
            try:
                k2 = int(key)
            except Exception:
                k2 = None
            if k2 is not None and k2 in base:
                return base[k2]
        else:
            k2 = str(key)
            if k2 in base:
                return base[k2]
        raise KeyError(f"cav_id={key} not found in base keys (example) {list(base.keys())[:10]}")

    def _range_filter(self, xyz: np.ndarray) -> np.ndarray:
        if self.lidar_range is None:
            return xyz
        x1, y1, z1, x2, y2, z2 = [float(v) for v in self.lidar_range]
        m = (
            (xyz[:, 0] >= x1)
            & (xyz[:, 0] <= x2)
            & (xyz[:, 1] >= y1)
            & (xyz[:, 1] <= y2)
            & (xyz[:, 2] >= z1)
            & (xyz[:, 2] <= z2)
        )
        return xyz[m]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = int(idx)
        # Robustly skip degenerate samples.
        for _ in range(10):
            item = self.dataset[idx]["ego"]
            cav_num = int(item.get("cav_num", 0) or 0)
            if cav_num < 2:
                idx = (idx + 1) % len(self.dataset)
                continue
            cav_idx = int(np.random.randint(1, cav_num))
            poses_clean = np.asarray(item["lidar_poses_clean"], dtype=np.float64)
            T_all = get_relative_transformation(poses_clean)  # (L,4,4) cav->ego
            T_gt = np.asarray(T_all[cav_idx], dtype=np.float64)

            # Simulate noisy initial estimate.
            dx = float(np.random.normal(0.0, self.noise_pos_std))
            dy = float(np.random.normal(0.0, self.noise_pos_std))
            dyaw = float(np.random.normal(0.0, self.noise_rot_std))
            T_noise = _make_se2(dx, dy, dyaw)
            T_init = T_noise @ T_gt
            T_delta_gt = np.linalg.inv(T_noise)  # target delta, in ego frame

            sample_idx = int(item.get("sample_idx", idx))
            cav_id_list = list(item.get("cav_id_list") or [])
            if len(cav_id_list) != cav_num:
                idx = (idx + 1) % len(self.dataset)
                continue
            base = self.dataset.retrieve_base_data(sample_idx)
            ego_pts = self._lookup(base, cav_id_list[0])["lidar_np"]
            cav_pts = self._lookup(base, cav_id_list[cav_idx])["lidar_np"]
            ego_xyz = np.asarray(ego_pts, dtype=np.float32).reshape(-1, ego_pts.shape[-1])[:, :3]
            cav_xyz = np.asarray(cav_pts, dtype=np.float32).reshape(-1, cav_pts.shape[-1])[:, :3]
            ego_xyz = self._range_filter(ego_xyz)
            cav_xyz = self._range_filter(cav_xyz)
            cav_in_ego_init = _apply_T(cav_xyz, T_init)

            ego_xy = _sample_xy(ego_xyz, self.num_points) / self.coord_scale
            cav_xy = _sample_xy(cav_in_ego_init, self.num_points) / self.coord_scale

            R = T_delta_gt[0:2, 0:2].astype(np.float32)
            t = (T_delta_gt[0:2, 3] / self.coord_scale).astype(np.float32)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))

            return {
                "src": torch.from_numpy(cav_xy),  # (N,2)  src is noisy-aligned cav points in ego frame
                "tgt": torch.from_numpy(ego_xy),  # (N,2)
                "t_gt": torch.from_numpy(t),  # (2,)
                "yaw_gt": torch.tensor(yaw, dtype=torch.float32),
            }
        # Fallback (should be rare)
        return {
            "src": torch.zeros((self.num_points, 2), dtype=torch.float32),
            "tgt": torch.zeros((self.num_points, 2), dtype=torch.float32),
            "t_gt": torch.zeros((2,), dtype=torch.float32),
            "yaw_gt": torch.zeros((), dtype=torch.float32),
        }


def _angle_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    d = pred - gt
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.abs(d).mean()


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    model_dir = Path(args.model_dir)
    hypes = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    if bool(args.val_split):
        hypes["validate_dir"] = hypes["test_dir"]
        train = False
    else:
        train = True

    device = torch.device(str(args.device) if (str(args.device) != "cuda" or torch.cuda.is_available()) else "cpu")
    ds = V2VLocPairDataset(
        hypes,
        train=train,
        max_samples=int(args.max_samples),
        num_points=int(args.num_points),
        coord_scale=float(args.coord_scale),
        noise_pos_std=float(args.noise_pos_std),
        noise_rot_std=float(args.noise_rot_std),
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = DCP2D(feat_dim=int(args.feat_dim)).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for epoch in range(int(args.epochs)):
        running = 0.0
        n = 0
        for step, batch in enumerate(dl):
            src = batch["src"].to(device)  # (B,N,2)
            tgt = batch["tgt"].to(device)
            t_gt = batch["t_gt"].to(device)  # (B,2)
            yaw_gt = batch["yaw_gt"].to(device)  # (B,)

            R, t_pred, _conf = model(src, tgt)
            yaw_pred = torch.atan2(R[:, 1, 0], R[:, 0, 0])

            loss_t = F.smooth_l1_loss(t_pred, t_gt)
            loss_yaw = _angle_l1(yaw_pred, yaw_gt)
            loss = loss_t + loss_yaw

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu().item())
            n += 1
            if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
                print(f"[v2vloc_dcp][epoch {epoch+1}/{int(args.epochs)}] step={step+1} avg_loss={running/max(n,1):.4f}")

        print(f"[v2vloc_dcp] epoch {epoch+1}/{int(args.epochs)} done: avg_loss={running/max(n,1):.4f}")

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_ckpt))
    print(f"[v2vloc_dcp] saved: {out_ckpt}")


if __name__ == "__main__":
    main()

