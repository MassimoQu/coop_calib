#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.pose.freealign_paper import EdgeGATLite, FreeAlignPaperConfig
from opencood.utils.common_utils import compute_iou, convert_format


def _as_T_from_rt(rotation, translation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj


def _dair_T_infra_to_vehicle(dair_root: Path, infra_frame_id: str, veh_frame_id: str) -> np.ndarray:
    infra_frame = str(infra_frame_id)
    veh_frame = str(veh_frame_id)
    infra_calib = dair_root / "infrastructure-side" / "calib" / "virtuallidar_to_world" / f"{infra_frame}.json"
    veh_world = dair_root / "vehicle-side" / "calib" / "novatel_to_world" / f"{veh_frame}.json"
    veh_lidar = dair_root / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{veh_frame}.json"

    infra_obj = _read_json(infra_calib)
    veh_world_obj = _read_json(veh_world)
    veh_lidar_obj = _read_json(veh_lidar)

    T_world_infra = _as_T_from_rt(infra_obj["rotation"], infra_obj["translation"])
    T_world_novatel = _as_T_from_rt(veh_world_obj["rotation"], veh_world_obj["translation"])
    # lidar_to_novatel stores parent=novatel, child=lidar.
    T_novatel_lidar = _as_T_from_rt(
        veh_lidar_obj["transform"]["rotation"],
        [x[0] for x in veh_lidar_obj["transform"]["translation"]],
    )
    T_world_veh_lidar = T_world_novatel @ T_novatel_lidar
    return np.linalg.inv(T_world_veh_lidar) @ T_world_infra


def _centers_xy(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 8, 3)
    if corners.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.mean(axis=1)[:, :2].astype(np.float32)


def _pairwise_dist_torch(centers_xy: torch.Tensor) -> torch.Tensor:
    if centers_xy.numel() == 0:
        return centers_xy.new_zeros((0, 0))
    diff = centers_xy[:, None, :] - centers_xy[None, :, :]
    return torch.linalg.norm(diff, dim=-1)


def _topk_indices(scores: Optional[Sequence[float]], k: int, *, n: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if scores is None:
        return np.arange(min(int(k), int(n)), dtype=np.int64)
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size <= k:
        return np.arange(arr.size, dtype=np.int64)
    return np.argsort(-arr)[:k].astype(np.int64)


def _match_nodes_by_hungarian(
    veh_centers: np.ndarray,
    infra_centers_in_veh: np.ndarray,
    *,
    max_dist: float,
) -> List[Tuple[int, int]]:
    if veh_centers.size == 0 or infra_centers_in_veh.size == 0:
        return []
    C = np.linalg.norm(veh_centers[:, None, :] - infra_centers_in_veh[None, :, :], axis=2).astype(np.float64)
    row, col = linear_sum_assignment(C)
    pairs = []
    for r, c in zip(row.tolist(), col.tolist()):
        if float(C[r, c]) <= float(max_dist):
            pairs.append((int(r), int(c)))
    return pairs


def _transform_corners_xy(corners: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 8, 3)
    out = corners.copy()
    out[:, :, :2] = (out[:, :, :2] @ R.T) + t[None, None, :]
    return out


def _iou_matrix_bev(corners0: np.ndarray, corners1: np.ndarray) -> np.ndarray:
    corners0 = np.asarray(corners0, dtype=np.float32).reshape(-1, 8, 3)
    corners1 = np.asarray(corners1, dtype=np.float32).reshape(-1, 8, 3)
    if corners0.shape[0] == 0 or corners1.shape[0] == 0:
        return np.zeros((int(corners0.shape[0]), int(corners1.shape[0])), dtype=np.float32)
    polys0 = convert_format(corners0)
    polys1 = convert_format(corners1)
    out = np.zeros((int(corners0.shape[0]), int(corners1.shape[0])), dtype=np.float32)
    for i in range(int(corners0.shape[0])):
        out[i] = compute_iou(polys0[i], polys1[:])
    return out


def _match_nodes_by_iou_hungarian(
    veh_corners: np.ndarray,
    infra_corners_in_veh: np.ndarray,
    *,
    min_iou: float,
) -> List[Tuple[int, int]]:
    iou = _iou_matrix_bev(veh_corners, infra_corners_in_veh).astype(np.float64)
    if iou.size == 0:
        return []
    cost = 1.0 - iou
    row, col = linear_sum_assignment(cost)
    pairs: List[Tuple[int, int]] = []
    for r, c in zip(row.tolist(), col.tolist()):
        if float(iou[r, c]) >= float(min_iou):
            pairs.append((int(r), int(c)))
    return pairs


def _build_edge_pairs(matches: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    pairs = []
    for a in range(len(matches)):
        vi_a, ii_a = matches[a]
        for b in range(a + 1, len(matches)):
            vi_b, ii_b = matches[b]
            pairs.append((int(vi_a), int(vi_b), int(ii_a), int(ii_b)))
    return pairs


def _sample_negative_edges(
    *,
    num_infra: int,
    exclude: Tuple[int, int],
    rng: random.Random,
) -> Tuple[int, int]:
    a_ex, b_ex = int(exclude[0]), int(exclude[1])
    if num_infra <= 1:
        return 0, 0
    for _ in range(50):
        a = rng.randrange(num_infra)
        b = rng.randrange(num_infra)
        if a == b:
            continue
        if (a == a_ex and b == b_ex) or (a == b_ex and b == a_ex):
            continue
        return int(a), int(b)
    # fall back (may collide in degenerate cases)
    a = 0
    b = 1 if num_infra > 1 else 0
    return int(a), int(b)


def train_one_epoch(
    *,
    net: EdgeGATLite,
    optimizer: torch.optim.Optimizer,
    samples: List[Dict[str, Any]],
    dair_root: Path,
    device: torch.device,
    max_boxes: int,
    max_match_dist: float,
    min_match_iou: float,
    pairing: str,
    min_match_nodes: int,
    margin: float,
    neg_per_pos: int,
    rng: random.Random,
) -> Dict[str, float]:
    net.train()
    loss_sum = 0.0
    pos_sum = 0.0
    neg_sum = 0.0
    n_steps = 0
    n_used = 0

    margin_t = torch.tensor(float(margin), device=device)

    for entry in samples:
        corners_all = entry.get("pred_corner3d_np_list") or []
        scores_all = entry.get("pred_score_np_list") or []
        if not isinstance(corners_all, list) or len(corners_all) < 2:
            continue

        infra_corners = np.asarray(corners_all[0], dtype=np.float32).reshape(-1, 8, 3)
        veh_corners = np.asarray(corners_all[1], dtype=np.float32).reshape(-1, 8, 3)
        infra_scores = scores_all[0] if isinstance(scores_all, list) and len(scores_all) >= 2 else None
        veh_scores = scores_all[1] if isinstance(scores_all, list) and len(scores_all) >= 2 else None

        if infra_corners.shape[0] == 0 or veh_corners.shape[0] == 0:
            continue

        infra_idx = _topk_indices(infra_scores, max_boxes, n=int(infra_corners.shape[0]))
        veh_idx = _topk_indices(veh_scores, max_boxes, n=int(veh_corners.shape[0]))
        infra_corners = infra_corners[infra_idx] if infra_idx.size else infra_corners[:max_boxes]
        veh_corners = veh_corners[veh_idx] if veh_idx.size else veh_corners[:max_boxes]

        infra_cent = _centers_xy(infra_corners)
        veh_cent = _centers_xy(veh_corners)
        if infra_cent.shape[0] < min_match_nodes or veh_cent.shape[0] < min_match_nodes:
            continue

        infra_frame_id = str(entry.get("infra_frame_id"))
        veh_frame_id = str(entry.get("veh_frame_id"))
        if not infra_frame_id or not veh_frame_id:
            continue

        try:
            T = _dair_T_infra_to_vehicle(dair_root, infra_frame_id, veh_frame_id)
        except Exception:
            continue
        R = T[:2, :2].astype(np.float32)
        t = T[:2, 3].astype(np.float32)
        infra_cent_in_veh = (infra_cent @ R.T) + t[None, :]

        pairing_key = str(pairing or "center_hungarian").lower().strip()
        if pairing_key in {"center_hungarian", "center"}:
            node_matches = _match_nodes_by_hungarian(veh_cent, infra_cent_in_veh, max_dist=max_match_dist)
        elif pairing_key in {"iou_hungarian", "iou"}:
            infra_corners_in_veh = _transform_corners_xy(infra_corners, R, t)
            node_matches = _match_nodes_by_iou_hungarian(veh_corners, infra_corners_in_veh, min_iou=min_match_iou)
        else:
            raise ValueError(f"Unsupported pairing: {pairing!r}")
        if len(node_matches) < int(min_match_nodes):
            continue
        edge_pairs = _build_edge_pairs(node_matches)
        if not edge_pairs:
            continue

        veh_cent_t = torch.from_numpy(veh_cent).to(device)
        infra_cent_t = torch.from_numpy(infra_cent).to(device)
        D_veh = _pairwise_dist_torch(veh_cent_t)
        D_infra = _pairwise_dist_torch(infra_cent_t)

        W_veh = net(D_veh)  # (N,N,K)
        W_infra = net(D_infra)  # (M,M,K)

        ep = torch.as_tensor(edge_pairs, dtype=torch.long, device=device)  # (E,4)
        if ep.numel() == 0:
            continue
        vi_a = ep[:, 0]
        vi_b = ep[:, 1]
        ii_a = ep[:, 2]
        ii_b = ep[:, 3]

        a = W_veh[vi_a, vi_b]  # (E,K)
        b = W_infra[ii_a, ii_b]  # (E,K)
        pos_d2 = torch.sum((a - b) ** 2, dim=-1)  # (E,)
        pos_loss = pos_d2.mean()

        if int(neg_per_pos) <= 0 or int(W_infra.shape[0]) <= 1:
            neg_loss = pos_loss.new_tensor(0.0)
        else:
            E = int(a.shape[0])
            neg_total = E * int(neg_per_pos)
            a_rep = a.repeat_interleave(int(neg_per_pos), dim=0)  # (E*neg_per_pos, K)
            ii_a_rep = ii_a.repeat_interleave(int(neg_per_pos))
            ii_b_rep = ii_b.repeat_interleave(int(neg_per_pos))

            M = int(W_infra.shape[0])
            ni_a = torch.randint(0, M, (neg_total,), device=device)
            ni_b = torch.randint(0, M, (neg_total,), device=device)
            for _ in range(5):
                bad = (ni_a == ni_b) | ((ni_a == ii_a_rep) & (ni_b == ii_b_rep)) | ((ni_a == ii_b_rep) & (ni_b == ii_a_rep))
                if not bool(bad.any()):
                    break
                n_bad = int(bad.sum())
                ni_a[bad] = torch.randint(0, M, (n_bad,), device=device)
                ni_b[bad] = torch.randint(0, M, (n_bad,), device=device)

            bn = W_infra[ni_a, ni_b]  # (E*neg_per_pos, K)
            neg_d2 = torch.sum((a_rep - bn) ** 2, dim=-1)
            neg_loss = F.relu(margin_t - neg_d2).mean()

        loss = pos_loss + neg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        pos_sum += float(pos_loss.detach().cpu())
        neg_sum += float(neg_loss.detach().cpu())
        n_steps += 1
        n_used += 1

    return {
        "loss": float(loss_sum / max(n_steps, 1)),
        "loss_pos": float(pos_sum / max(n_steps, 1)),
        "loss_neg": float(neg_sum / max(n_steps, 1)),
        "steps": float(n_steps),
        "used_samples": float(n_used),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_cache", type=str, required=True, help="Dual-agent stage1 cache json (dict of entries).")
    ap.add_argument(
        "--dair_root",
        type=str,
        default="~/datasets/data2/DAIR-V2X-C/cooperative-vehicle-infrastructure",
        help="DAIR-V2X-C root dir used to load calib.",
    )
    ap.add_argument("--out_ckpt", type=str, required=True, help="Where to save trained EdgeGATLite state_dict.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=16)
    ap.add_argument("--max_boxes", type=int, default=60)
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on samples per epoch (for quick runs).")
    ap.add_argument("--max_match_dist", type=float, default=3.0, help="Max center distance (m) to accept a GT node match.")
    ap.add_argument("--pairing", type=str, default="center_hungarian", choices=["center_hungarian", "iou_hungarian"])
    ap.add_argument("--min_match_iou", type=float, default=0.1, help="Min BEV IoU to accept a GT node match (for IoU pairing).")
    ap.add_argument("--min_match_nodes", type=int, default=4, help="Minimum matched nodes to form training edges.")
    ap.add_argument("--margin", type=float, default=1.0, help="Contrastive margin gamma (on squared distance).")
    ap.add_argument("--neg_per_pos", type=int, default=4, help="Number of negative edges per positive edge.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    stage1_path = resolve_repo_path(args.stage1_cache)
    dair_root = Path(str(args.dair_root)).expanduser().resolve()
    out_ckpt = resolve_repo_path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[freealign-train] loading stage1 cache: {stage1_path}")
    data = json.loads(stage1_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("stage1_cache must be a dict JSON")

    keys = sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    samples = [data[k] for k in keys]
    if args.max_samples:
        samples = samples[: int(args.max_samples)]
    rng.shuffle(samples)

    device = torch.device(str(args.device))
    net = EdgeGATLite(hidden_dim=int(args.hidden_dim), out_dim=int(args.out_dim)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    cfg = FreeAlignPaperConfig(
        ckpt_path=str(out_ckpt),
        device=str(args.device),
        hidden_dim=int(args.hidden_dim),
        out_dim=int(args.out_dim),
        use_gnn=True,
        max_boxes=int(args.max_boxes),
    )
    print("[freealign-train] config:", json.dumps(asdict(cfg), indent=2))

    for epoch in range(1, int(args.epochs) + 1):
        metrics = train_one_epoch(
            net=net,
            optimizer=optimizer,
            samples=samples,
            dair_root=dair_root,
            device=device,
            max_boxes=int(args.max_boxes),
            max_match_dist=float(args.max_match_dist),
            min_match_iou=float(args.min_match_iou),
            pairing=str(args.pairing),
            min_match_nodes=int(args.min_match_nodes),
            margin=float(args.margin),
            neg_per_pos=int(args.neg_per_pos),
            rng=rng,
        )
        print(
            f"[freealign-train] epoch {epoch}/{args.epochs} "
            f"loss={metrics['loss']:.6f} pos={metrics['loss_pos']:.6f} neg={metrics['loss_neg']:.6f} "
            f"steps={int(metrics['steps'])}"
        )

        payload = {
            "state_dict": net.state_dict(),
            "cfg": asdict(cfg),
            "epoch": int(epoch),
            "metrics": metrics,
        }
        torch.save(payload, str(out_ckpt))

    print(f"[freealign-train] saved: {out_ckpt}")


if __name__ == "__main__":
    main()
