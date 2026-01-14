#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import EdgeGATLite
from opencood.utils.box_utils import project_box3d
from opencood.utils.transformation_utils import x1_to_x2


def _sorted_keys(data: Mapping[str, Any]) -> List[str]:
    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    return sorted(data.keys(), key=_key)


def _centers_xy(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 8, 3)
    if corners.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return corners.mean(axis=1)[:, :2]


def _pairwise_dist(centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def _greedy_match(ego_xy: np.ndarray, cav_xy_in_ego: np.ndarray, thr_m: float) -> List[Tuple[int, int]]:
    if ego_xy.size == 0 or cav_xy_in_ego.size == 0:
        return []
    d = ego_xy[:, None, :] - cav_xy_in_ego[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    pairs = [(float(dist[i, j]), int(i), int(j)) for i in range(dist.shape[0]) for j in range(dist.shape[1])]
    pairs.sort(key=lambda x: x[0])

    used_i = set()
    used_j = set()
    out: List[Tuple[int, int]] = []
    for dij, i, j in pairs:
        if dij > float(thr_m):
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


@dataclass(frozen=True)
class PairSample:
    D_ego: np.ndarray  # (Ne,Ne) float32
    D_cav: np.ndarray  # (Nc,Nc) float32
    match_ego: np.ndarray  # (K,) int64
    match_cav: np.ndarray  # (K,) int64


def _load_stage1(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict stage1 json at {path}")
    return data


def _build_samples(
    *,
    stage1: Mapping[str, Any],
    max_boxes: int,
    match_thr_m: float,
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> List[PairSample]:
    rs = random.Random(int(seed))
    keys = _sorted_keys(stage1)
    samples: List[PairSample] = []

    for key in keys:
        entry = stage1.get(key)
        if entry is None:
            continue
        corners_all = entry.get("pred_corner3d_np_list") or []
        poses_all = entry.get("lidar_pose_clean_np") or []
        if not isinstance(corners_all, list) or not isinstance(poses_all, list):
            continue
        if len(corners_all) < 2 or len(poses_all) < 2:
            continue

        ego_pose = poses_all[0]
        if not isinstance(ego_pose, (list, tuple)) or len(ego_pose) < 6:
            continue
        ego_corners = np.asarray(corners_all[0] or [], dtype=np.float64).reshape(-1, 8, 3)
        if ego_corners.size == 0:
            continue
        ego_corners = ego_corners[: int(max_boxes)]
        ego_xy = _centers_xy(ego_corners)
        if ego_xy.shape[0] < 2:
            continue
        D_ego = _pairwise_dist(ego_xy)

        # Pair ego with each non-ego CAV.
        for cav_idx in range(1, len(corners_all)):
            cav_pose = poses_all[cav_idx]
            if not isinstance(cav_pose, (list, tuple)) or len(cav_pose) < 6:
                continue
            cav_corners = np.asarray(corners_all[cav_idx] or [], dtype=np.float64).reshape(-1, 8, 3)
            if cav_corners.size == 0:
                continue
            cav_corners = cav_corners[: int(max_boxes)]
            cav_xy = _centers_xy(cav_corners)
            if cav_xy.shape[0] < 2:
                continue
            D_cav = _pairwise_dist(cav_xy)

            # Supervision: match nodes using clean pose to project cav boxes into ego frame.
            T_ego_cav = x1_to_x2(cav_pose, ego_pose)  # cav -> ego
            cav_proj = project_box3d(cav_corners.astype(np.float32, copy=False), T_ego_cav)
            cav_xy_ego = _centers_xy(cav_proj)
            if cav_xy_ego.shape[0] < 2:
                continue
            matches = _greedy_match(ego_xy, cav_xy_ego, thr_m=float(match_thr_m))
            if len(matches) < 2:
                continue

            match_ego = np.asarray([i for i, _ in matches], dtype=np.int64)
            match_cav = np.asarray([j for _, j in matches], dtype=np.int64)
            samples.append(PairSample(D_ego=D_ego, D_cav=D_cav, match_ego=match_ego, match_cav=match_cav))

            if max_pairs is not None and len(samples) >= int(max_pairs):
                rs.shuffle(samples)
                return samples[: int(max_pairs)]

    if max_pairs is not None and len(samples) > int(max_pairs):
        rs.shuffle(samples)
        samples = samples[: int(max_pairs)]
    return samples


def _sample_pos_edges(
    match_ego: torch.Tensor,
    match_cav: torch.Tensor,
    *,
    num_edges: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = int(match_ego.numel())
    if k < 2:
        raise ValueError("Need at least 2 matched nodes")
    i = torch.randint(0, k, (num_edges,), device=match_ego.device)
    j = torch.randint(0, k, (num_edges,), device=match_ego.device)
    # Ensure i != j (few resamples; k is small).
    for _ in range(4):
        bad = i == j
        if not bool(bad.any()):
            break
        j = torch.where(
            bad,
            torch.randint(0, k, (num_edges,), device=match_ego.device),
            j,
        )
    bad = i == j
    if bool(bad.any()):
        j = torch.where(bad, (j + 1) % k, j)

    ego_u = match_ego[i]
    ego_v = match_ego[j]
    cav_u = match_cav[i]
    cav_v = match_cav[j]
    return ego_u, ego_v, cav_u, cav_v


@torch.no_grad()
def _clamp_neg_indices(neg: torch.Tensor, avoid: torch.Tensor) -> torch.Tensor:
    # Ensure neg != avoid element-wise by shifting conflicts by +1 (mod M) later.
    # This assumes avoid and neg share leading dims.
    bad = neg == avoid
    if bool(bad.any()):
        neg = neg.clone()
        neg[bad] = neg[bad] + 1
    return neg


def train_one_epoch(
    *,
    net: nn.Module,
    opt: torch.optim.Optimizer,
    samples: Sequence[PairSample],
    device: torch.device,
    shuffle_seed: int,
    pos_edges_per_pair: int,
    neg_per_pos: int,
    margin: float,
) -> float:
    net.train()
    rs = random.Random(int(shuffle_seed))
    indices = list(range(len(samples)))
    rs.shuffle(indices)

    losses: List[float] = []
    for idx in indices:
        s = samples[idx]
        D_ego = torch.from_numpy(np.asarray(s.D_ego, dtype=np.float32)).to(device)
        D_cav = torch.from_numpy(np.asarray(s.D_cav, dtype=np.float32)).to(device)
        match_ego = torch.from_numpy(np.asarray(s.match_ego, dtype=np.int64)).to(device)
        match_cav = torch.from_numpy(np.asarray(s.match_cav, dtype=np.int64)).to(device)

        if int(match_ego.numel()) < 2:
            continue

        W_ego = net(D_ego)  # (Ne,Ne,K)
        W_cav = net(D_cav)  # (Nc,Nc,K)

        P = int(min(int(pos_edges_per_pair), max(int(match_ego.numel() * (match_ego.numel() - 1) // 2), 1)))
        ego_u, ego_v, cav_u, cav_v = _sample_pos_edges(match_ego, match_cav, num_edges=P)
        w_pos_ego = W_ego[ego_u, ego_v]  # (P,K)
        w_pos_cav = W_cav[cav_u, cav_v]  # (P,K)

        pos_dist2 = (w_pos_ego - w_pos_cav).pow(2).sum(dim=-1)  # (P,)
        pos_loss = pos_dist2.mean()

        # Negatives: keep cav_u fixed, corrupt cav_v.
        M = int(D_cav.shape[0])
        if M < 2:
            continue
        neg_v = torch.randint(0, M, (P, int(neg_per_pos)), device=device)
        neg_v = _clamp_neg_indices(neg_v, cav_v[:, None])
        neg_v = torch.remainder(neg_v, M)

        w_neg_cav = W_cav[cav_u[:, None], neg_v]  # (P,neg,K)
        neg_dist2 = (w_pos_ego[:, None, :] - w_neg_cav).pow(2).sum(dim=-1)  # (P,neg)
        neg_loss = torch.relu(float(margin) - neg_dist2).mean()

        loss = pos_loss + neg_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses) if losses else 0.0)


@torch.no_grad()
def eval_loss(
    *,
    net: nn.Module,
    samples: Sequence[PairSample],
    device: torch.device,
    pos_edges_per_pair: int,
    neg_per_pos: int,
    margin: float,
) -> float:
    net.eval()
    losses: List[float] = []
    for s in samples:
        D_ego = torch.from_numpy(np.asarray(s.D_ego, dtype=np.float32)).to(device)
        D_cav = torch.from_numpy(np.asarray(s.D_cav, dtype=np.float32)).to(device)
        match_ego = torch.from_numpy(np.asarray(s.match_ego, dtype=np.int64)).to(device)
        match_cav = torch.from_numpy(np.asarray(s.match_cav, dtype=np.int64)).to(device)
        if int(match_ego.numel()) < 2:
            continue

        W_ego = net(D_ego)
        W_cav = net(D_cav)
        P = int(min(int(pos_edges_per_pair), max(int(match_ego.numel() * (match_ego.numel() - 1) // 2), 1)))
        ego_u, ego_v, cav_u, cav_v = _sample_pos_edges(match_ego, match_cav, num_edges=P)
        w_pos_ego = W_ego[ego_u, ego_v]
        w_pos_cav = W_cav[cav_u, cav_v]
        pos_dist2 = (w_pos_ego - w_pos_cav).pow(2).sum(dim=-1)
        pos_loss = pos_dist2.mean()

        M = int(D_cav.shape[0])
        if M < 2:
            continue
        neg_v = torch.randint(0, M, (P, int(neg_per_pos)), device=device)
        neg_v = _clamp_neg_indices(neg_v, cav_v[:, None])
        neg_v = torch.remainder(neg_v, M)
        w_neg_cav = W_cav[cav_u[:, None], neg_v]
        neg_dist2 = (w_pos_ego[:, None, :] - w_neg_cav).pow(2).sum(dim=-1)
        neg_loss = torch.relu(float(margin) - neg_dist2).mean()
        loss = pos_loss + neg_loss
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses) if losses else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a lightweight EdgeGAT-style edge feature learner on OPV2V stage1 boxes.")
    ap.add_argument("--train_stage1", type=str, required=True, help="Path to OPV2V train stage1_boxes.json")
    ap.add_argument("--val_stage1", type=str, default=None, help="Optional path to OPV2V val stage1_boxes.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints/logs")

    ap.add_argument("--max_boxes", type=int, default=30)
    ap.add_argument("--match_thr_m", type=float, default=2.0, help="Center-distance threshold to treat two boxes as same object (meters).")
    ap.add_argument("--max_train_pairs", type=int, default=None, help="Optional cap on training pairs (for quick runs).")
    ap.add_argument("--max_val_pairs", type=int, default=2000, help="Optional cap on validation pairs.")

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=16)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--pos_edges_per_pair", type=int, default=64)
    ap.add_argument("--neg_per_pos", type=int, default=4)
    ap.add_argument("--margin", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_path = resolve_repo_path(args.train_stage1)
    train_data = _load_stage1(train_path)
    train_samples = _build_samples(
        stage1=train_data,
        max_boxes=int(args.max_boxes),
        match_thr_m=float(args.match_thr_m),
        max_pairs=None if args.max_train_pairs is None else int(args.max_train_pairs),
        seed=int(args.seed),
    )
    if not train_samples:
        raise SystemExit("No training samples built. Check stage1 cache path and thresholds.")

    val_samples: List[PairSample] = []
    if args.val_stage1:
        val_path = resolve_repo_path(args.val_stage1)
        val_data = _load_stage1(val_path)
        val_samples = _build_samples(
            stage1=val_data,
            max_boxes=int(args.max_boxes),
            match_thr_m=float(args.match_thr_m),
            max_pairs=None if args.max_val_pairs is None else int(args.max_val_pairs),
            seed=int(args.seed),
        )

    device = torch.device(str(args.device))
    net = EdgeGATLite(hidden_dim=int(args.hidden_dim), out_dim=int(args.out_dim))
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        json.dumps(
            {
                "train_pairs": int(len(train_samples)),
                "val_pairs": int(len(val_samples)),
                "max_boxes": int(args.max_boxes),
                "match_thr_m": float(args.match_thr_m),
                "device": str(device),
            },
            indent=2,
        )
    )

    best_val = float("inf")
    best_path: Optional[Path] = None

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            net=net,
            opt=opt,
            samples=train_samples,
            device=device,
            shuffle_seed=int(args.seed) + int(epoch),
            pos_edges_per_pair=int(args.pos_edges_per_pair),
            neg_per_pos=int(args.neg_per_pos),
            margin=float(args.margin),
        )
        val_loss = None
        if val_samples:
            val_loss = eval_loss(
                net=net,
                samples=val_samples,
                device=device,
                pos_edges_per_pair=int(args.pos_edges_per_pair),
                neg_per_pos=int(args.neg_per_pos),
                margin=float(args.margin),
            )

        ckpt = out_dir / f"edgegatlite_opv2v_epoch{epoch}.pth"
        torch.save(net.state_dict(), str(ckpt))
        record = {"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": None if val_loss is None else float(val_loss), "ckpt": str(ckpt)}
        with (out_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))

        score = float(val_loss) if val_loss is not None else float(train_loss)
        if score < best_val - 1e-9:
            best_val = score
            best_path = ckpt

    if best_path is not None:
        print(json.dumps({"best_ckpt": str(best_path), "best_score": float(best_val)}))


if __name__ == "__main__":
    main()
