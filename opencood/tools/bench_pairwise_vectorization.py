import time

import numpy as np
import torch

from opencood.utils.transformation_utils import get_pairwise_transformation_torch, pose_to_tfm


def old_get_pairwise_transformation_torch(lidar_poses, max_cav, record_len, dof):
    def regroup(x, rec_len):
        cum_sum_len = torch.cumsum(rec_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    batch_size = len(record_len)
    lidar_poses_list = regroup(lidar_poses, record_len)

    pairwise_t_matrix = (
        torch.eye(4, device=lidar_poses.device, dtype=lidar_poses.dtype)
        .view(1, 1, 1, 4, 4)
        .repeat(batch_size, max_cav, max_cav, 1, 1)
    )

    for b in range(batch_size):
        lidar_poses_b = lidar_poses_list[b]
        t_list = pose_to_tfm(lidar_poses_b)
        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    pairwise_t_matrix[b, i, j] = torch.linalg.solve(t_list[j], t_list[i])

    return pairwise_t_matrix


def bench_case(batch_size, max_cav, loops, seed=0):
    rng = np.random.default_rng(seed)
    poses = torch.from_numpy(rng.normal(size=(batch_size * max_cav, 6)).astype(np.float32)).cuda()
    record_len = torch.tensor([max_cav] * batch_size, device=poses.device)

    new_out = get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
    old_out = old_get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
    max_diff = torch.max(torch.abs(new_out - old_out)).item()

    for _ in range(30):
        old_get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
        get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(loops):
        old_get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
    torch.cuda.synchronize()
    old_sec = time.time() - t0

    t0 = time.time()
    for _ in range(loops):
        get_pairwise_transformation_torch(poses, max_cav, record_len, dof=6)
    torch.cuda.synchronize()
    new_sec = time.time() - t0

    return {
        "B": int(batch_size),
        "L": int(max_cav),
        "loops": int(loops),
        "max_abs_diff": float(max_diff),
        "old_sec": float(old_sec),
        "new_sec": float(new_sec),
        "speedup": float(old_sec / max(new_sec, 1e-12)),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable; benchmark skipped.")
        return

    cases = [(1, 5, 4000), (8, 5, 2000), (16, 5, 1000)]
    for bsz, cav, loops in cases:
        out = bench_case(bsz, cav, loops)
        print(
            "B={B} L={L} loops={loops} diff={max_abs_diff:.2e} "
            "old={old_sec:.4f}s new={new_sec:.4f}s speedup={speedup:.2f}x".format(**out)
        )


if __name__ == "__main__":
    main()
