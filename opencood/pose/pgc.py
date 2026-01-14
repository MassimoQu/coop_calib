from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rsd_downsample(points_xyzi: np.ndarray, *, num_points: int, voxel_size: float = 0.2) -> np.ndarray:
    """
    Redundant Sample Downsampling (RSD) approximation.

    We voxelize XYZ and keep at most 1 point per voxel, then (re-)sample to `num_points`.
    """
    pts = np.asarray(points_xyzi, dtype=np.float32).reshape(-1, 4)
    if pts.shape[0] == 0:
        return np.zeros((int(num_points), 4), dtype=np.float32)
    if int(num_points) <= 0:
        return pts

    vs = float(voxel_size)
    if not (vs > 0):
        vs = 0.2

    keys = np.floor(pts[:, :3] / vs).astype(np.int64)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    uniq = pts[np.sort(uniq_idx)]

    if uniq.shape[0] >= int(num_points):
        sel = np.random.choice(uniq.shape[0], int(num_points), replace=False)
        return uniq[sel].astype(np.float32)
    pad = np.random.choice(uniq.shape[0], int(num_points) - uniq.shape[0], replace=True)
    out = np.concatenate([uniq, uniq[pad]], axis=0)
    return out.astype(np.float32)


class PGCNet(nn.Module):
    """
    Pose Generator with Confidence (PGC).

    It regresses per-point world coordinates and a sample-level error proxy (epsilon).
    """

    def __init__(self, *, in_dim: int = 4, feat_dim: int = 256) -> None:
        super().__init__()
        in_dim = int(in_dim)
        feat_dim = int(feat_dim)
        if in_dim <= 0 or feat_dim <= 0:
            raise ValueError("in_dim/feat_dim must be positive")

        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        # PointNet-style SCR head: concatenate a global scene descriptor back to
        # per-point features so coordinates can vary across frames/places.
        self.coord_head = nn.Sequential(
            nn.Conv1d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim, 3, kernel_size=1, bias=True),
        )
        self.err_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B,N,4)
        Returns:
            y_pred: (B,N,3) predicted world coords
            eps_pred: (B,) non-negative error proxy
        """
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(f"Expected points shape (B,N,4), got {tuple(points.shape)}")
        x = points.transpose(1, 2).contiguous()  # (B,4,N)
        feat = self.encoder(x)  # (B,C,N)
        g = torch.max(feat, dim=2)[0]  # (B,C)
        g_exp = g.unsqueeze(-1).expand(-1, -1, feat.shape[2])
        feat_cat = torch.cat([feat, g_exp], dim=1)  # (B,2C,N)
        y = self.coord_head(feat_cat).transpose(1, 2).contiguous()  # (B,N,3)
        eps = F.softplus(self.err_head(g)).squeeze(-1)  # (B,)
        return y, eps


def pose_confidence_from_epsilon(eps: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + eps * eps)


def _kabsch_se3(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)
    src_mean = src.mean(axis=0, keepdims=True)
    tgt_mean = tgt.mean(axis=0, keepdims=True)
    X = src - src_mean
    Y = tgt - tgt_mean
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = (tgt_mean - (R @ src_mean.T).T).reshape(3)
    return R, t


def ransac_se3(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    num_iter: int = 64,
    inlier_th: float = 1.0,
    min_inliers: int = 16,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    RANSAC SE(3) alignment for correspondences src -> tgt.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] != tgt.shape[0]:
        raise ValueError("src/tgt must have same number of points")
    N = int(src.shape[0])
    if N < 3:
        T = np.eye(4, dtype=np.float64)
        return T, {"inliers": 0, "reason": "too_few_points"}

    best_inliers = np.zeros((N,), dtype=bool)
    best_cnt = 0
    th = float(inlier_th)

    for _ in range(int(num_iter)):
        idx = np.random.choice(N, 3, replace=False)
        R, t = _kabsch_se3(src[idx], tgt[idx])
        pred = (R @ src.T).T + t.reshape(1, 3)
        err = np.linalg.norm(pred - tgt, axis=1)
        inliers = err < th
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers

    if best_cnt < int(min_inliers):
        R, t = _kabsch_se3(src, tgt)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T, {"inliers": best_cnt, "reason": "low_inliers"}

    R, t = _kabsch_se3(src[best_inliers], tgt[best_inliers])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, {"inliers": best_cnt, "reason": "ok"}


@dataclass(frozen=True)
class PGCInferConfig:
    num_points: int = 4096
    rsd_voxel_size: float = 0.2
    ransac_iter: int = 64
    ransac_inlier_th: float = 1.0
    ransac_min_inliers: int = 16


@torch.no_grad()
def infer_pose_and_confidence(
    model: PGCNet,
    points_xyzi: np.ndarray,
    *,
    cfg: Optional[PGCInferConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Run PGC forward + RANSAC to get a pose (SE3) and confidence.
    """
    cfg = cfg or PGCInferConfig()
    pts = rsd_downsample(points_xyzi, num_points=int(cfg.num_points), voxel_size=float(cfg.rsd_voxel_size))
    pts_t = torch.from_numpy(pts).unsqueeze(0)
    if device is not None:
        pts_t = pts_t.to(device)
    y_pred, eps = model(pts_t)
    conf = float(pose_confidence_from_epsilon(eps)[0].detach().cpu().item())

    src = pts[:, :3].astype(np.float64)
    tgt = y_pred[0].detach().cpu().numpy().astype(np.float64)
    T, meta = ransac_se3(
        src,
        tgt,
        num_iter=int(cfg.ransac_iter),
        inlier_th=float(cfg.ransac_inlier_th),
        min_inliers=int(cfg.ransac_min_inliers),
    )
    meta = dict(meta)
    meta.update({"confidence": conf})
    return T, conf, meta
