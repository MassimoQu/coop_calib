from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wrap_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_R2(R: torch.Tensor) -> torch.Tensor:
    # R: (...,2,2)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def _kabsch_2d(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate R,t such that: (R @ src + t) ~= tgt

    Args:
        src: (B,N,2)
        tgt: (B,N,2)
    Returns:
        R: (B,2,2)
        t: (B,2)
    """
    if src.ndim != 3 or tgt.ndim != 3 or src.shape[-1] != 2 or tgt.shape[-1] != 2:
        raise ValueError(f"Expected src/tgt shape (B,N,2), got {tuple(src.shape)} and {tuple(tgt.shape)}")
    src_mean = src.mean(dim=1, keepdim=True)  # (B,1,2)
    tgt_mean = tgt.mean(dim=1, keepdim=True)
    X = src - src_mean
    Y = tgt - tgt_mean
    H = X.transpose(1, 2) @ Y  # (B,2,2)
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)
    R = V @ U.transpose(1, 2)
    det = torch.det(R)
    if (det < 0).any():
        V_fix = V.clone()
        # flip last column for reflection fix
        V_fix[det < 0, :, 1] *= -1.0
        R = V_fix @ U.transpose(1, 2)
    t = (tgt_mean.squeeze(1) - (R @ src_mean.squeeze(1).unsqueeze(-1)).squeeze(-1)).contiguous()
    return R, t


class PointNet2D(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, out_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.conv4 = nn.Conv1d(out_dim * 2, out_dim, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,N,2)
        Returns:
            feat: (B,N,C)
        """
        x = x.transpose(1, 2).contiguous()  # (B,2,N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B,C,N)
        g = torch.max(x, dim=2, keepdim=True)[0]  # (B,C,1)
        x = torch.cat([x, g.expand_as(x)], dim=1)  # (B,2C,N)
        x = F.relu(self.bn4(self.conv4(x)))  # (B,C,N)
        return x.transpose(1, 2).contiguous()  # (B,N,C)


class DCP2D(nn.Module):
    """
    A lightweight Deep-Closest-Point style registration module in 2D.

    It predicts a rigid SE(2) transform by:
      - learning per-point embeddings for src/tgt (PointNet2D),
      - computing soft correspondences via attention,
      - solving a closed-form Kabsch alignment.
    """

    def __init__(self, feat_dim: int = 256) -> None:
        super().__init__()
        self.feat = PointNet2D(out_dim=int(feat_dim))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src: (B,N,2)
            tgt: (B,M,2)
        Returns:
            R: (B,2,2)
            t: (B,2)
            confidence: (B,)
        """
        if src.ndim != 3 or tgt.ndim != 3:
            raise ValueError(f"Expected src/tgt rank-3, got {tuple(src.shape)} {tuple(tgt.shape)}")
        f_src = self.feat(src)  # (B,N,C)
        f_tgt = self.feat(tgt)  # (B,M,C)
        # Attention score matrix (B,N,M)
        scores = torch.matmul(f_src, f_tgt.transpose(1, 2)) / np.sqrt(float(f_src.shape[-1]))
        prob = torch.softmax(scores, dim=-1)
        tgt_hat = torch.matmul(prob, tgt)  # (B,N,2)
        R, t = _kabsch_2d(src, tgt_hat)
        src_aligned = (R @ src.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
        residual = torch.norm(src_aligned - tgt_hat, dim=-1).mean(dim=1)  # (B,)
        confidence = torch.exp(-residual)
        return R, t, confidence


@dataclass(frozen=True)
class V2VLocDCPConfig:
    ckpt_path: str
    device: str = "cpu"
    num_points: int = 2048
    coord_scale: float = 50.0
    feat_dim: int = 256


class V2VLocDCPEstimator:
    def __init__(self, cfg: V2VLocDCPConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.model = DCP2D(feat_dim=int(cfg.feat_dim)).to(self.device)
        state = torch.load(str(cfg.ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @staticmethod
    def _sample_points_xy(points_xyz: np.ndarray, num: int) -> np.ndarray:
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

    @staticmethod
    def _apply_T(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return pts
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
        out = (np.asarray(T, dtype=np.float32) @ pts_h.T).T[:, :3]
        return out

    @torch.no_grad()
    def estimate(
        self,
        cav_points_xyz: np.ndarray,
        ego_points_xyz: np.ndarray,
        T_init: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float, int, Dict[str, Any]]:
        if cav_points_xyz is None or ego_points_xyz is None:
            return None, 0.0, 0, {"reason": "missing_points"}
        if T_init is None:
            T_init = np.eye(4, dtype=np.float32)
        cav_in_ego = self._apply_T(cav_points_xyz, T_init)

        cav_xy = self._sample_points_xy(cav_in_ego, int(self.cfg.num_points))
        ego_xy = self._sample_points_xy(ego_points_xyz, int(self.cfg.num_points))
        scale = float(self.cfg.coord_scale) if float(self.cfg.coord_scale) > 0 else 1.0
        cav_xy = cav_xy / scale
        ego_xy = ego_xy / scale

        src = torch.from_numpy(cav_xy).unsqueeze(0).to(self.device)  # (1,N,2)
        tgt = torch.from_numpy(ego_xy).unsqueeze(0).to(self.device)
        R, t, conf = self.model(src, tgt)
        R = R[0].detach().cpu().numpy().astype(np.float64)
        t = (t[0].detach().cpu().numpy().astype(np.float64) * scale).astype(np.float64)
        conf_val = float(conf[0].detach().cpu().item())

        T_delta = np.eye(4, dtype=np.float64)
        T_delta[0:2, 0:2] = R
        T_delta[0:2, 3] = t
        T_pred = T_delta @ np.asarray(T_init, dtype=np.float64)
        meta = {
            "confidence": conf_val,
            "coord_scale": float(scale),
            "num_points": int(self.cfg.num_points),
        }
        return T_pred, conf_val, int(self.cfg.num_points), meta

