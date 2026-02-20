from collections import OrderedDict

try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
except Exception:
    print("SKIP: torch not available, skip pose provider tests")
    raise SystemExit(0)

from opencood.utils.pose_provider_runtime import (
    PoseProviderConfig,
    apply_overrides_to_batch,
    apply_pose_provider,
)
from opencood.utils.pose_utils import add_noise_data_dict, attach_pose_confidence
from opencood.utils.transformation_utils import (
    get_pairwise_transformation,
    get_pairwise_transformation_torch,
)


def _build_base_data(poses):
    base = OrderedDict()
    for i, pose in enumerate(poses):
        base[i] = {"params": {"lidar_pose": pose}}
    return base


def _cpu_pairwise_batch(poses_by_batch, max_cav, proj_first=False):
    if np is None:
        raise RuntimeError("numpy is required for CPU parity test")
    mats = []
    for poses in poses_by_batch:
        base = _build_base_data(poses)
        mats.append(get_pairwise_transformation(base, max_cav, proj_first))
    return np.stack(mats, axis=0)


def test_pairwise_parity():
    if np is None:
        print("SKIP: numpy not available, skip CPU/GPU parity test")
        return
    rng = np.random.default_rng(42)
    poses_by_batch = [
        rng.normal(size=(2, 6)).astype(np.float32),
        rng.normal(size=(3, 6)).astype(np.float32),
    ]
    max_cav = 3
    cpu = _cpu_pairwise_batch(poses_by_batch, max_cav)

    lidar_pose = torch.from_numpy(np.concatenate(poses_by_batch, axis=0))
    record_len = torch.tensor([len(p) for p in poses_by_batch], dtype=torch.int64)
    gpu = get_pairwise_transformation_torch(lidar_pose, max_cav, record_len, dof=6).cpu().numpy()

    diff = np.max(np.abs(cpu - gpu))
    assert diff <= 1e-5, f"pairwise parity failed: max diff {diff}"


def test_override_non_ego():
    override_map = {
        "0": {
            "cav_id_list": [0, 1],
            "lidar_pose_pred_np": [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
            ],
            "pose_confidence_np": [0.1, 0.2],
        }
    }
    batch = {
        "lidar_pose": torch.zeros((2, 6), dtype=torch.float32),
        "pose_confidence": torch.ones((2,), dtype=torch.float32),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
    }
    cfg = PoseProviderConfig(
        enabled=True,
        mode="register_and_fuse",
        apply_to="non-ego",
        freeze_ego=True,
        override_map=override_map,
    )
    applied = apply_overrides_to_batch(batch, cfg)
    assert applied, "override should apply"
    assert torch.allclose(batch["lidar_pose"][0], torch.zeros(6)), "ego should remain"
    assert torch.allclose(
        batch["lidar_pose"][1],
        torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.float32),
    )
    assert abs(float(batch["pose_confidence"][1]) - 0.2) < 1e-6


def test_gt_only_pairwise():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    poses_clean = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 20.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "lidar_pose_clean": poses_clean.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "pairwise_t_matrix": torch.eye(4, dtype=torch.float32)
        .view(1, 1, 1, 4, 4)
        .repeat(1, 2, 2, 1, 1),
    }
    cfg = PoseProviderConfig(enabled=True, mode="gt_only", recompute_pairwise=True)
    apply_pose_provider(batch, cfg)
    expected = get_pairwise_transformation_torch(poses_clean, 2, batch["record_len"], dof=6)
    diff = torch.max(torch.abs(batch["pairwise_t_matrix"] - expected)).item()
    assert diff <= 1e-5, f"gt_only pairwise mismatch: {diff}"


def test_proj_first_identity_and_label_sync():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0, 15.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
        "label_dict": {},
    }
    cfg = PoseProviderConfig(
        enabled=True,
        mode="register_and_fuse",
        recompute_pairwise=True,
        proj_first=True,
        max_cav=2,
    )
    apply_pose_provider(batch, cfg)
    expected = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    diff = torch.max(torch.abs(batch["pairwise_t_matrix"] - expected)).item()
    assert diff <= 1e-6, f"proj_first identity mismatch: {diff}"
    assert "pairwise_t_matrix" in batch["label_dict"], "label_dict pairwise sync missing"
    diff_label = torch.max(
        torch.abs(batch["label_dict"]["pairwise_t_matrix"] - batch["pairwise_t_matrix"])
    ).item()
    assert diff_label <= 1e-6, f"label_dict sync mismatch: {diff_label}"


def test_multibatch_override_with_string_keys():
    override_map = {
        "0": {
            "cav_id_list": [0, 1],
            "lidar_pose_pred_np": [[0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0]],
            "pose_confidence_np": [1.0, 0.5],
        },
        "1": {
            "cav_id_list": [0, 1],
            "lidar_pose_pred_np": [[0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]],
            "pose_confidence_np": [1.0, 0.2],
        },
    }
    batch = {
        "lidar_pose": torch.zeros((4, 6), dtype=torch.float32),
        "pose_confidence": torch.ones((4,), dtype=torch.float32),
        "record_len": torch.tensor([2, 2], dtype=torch.int64),
        "sample_idx": [0, 1],
        "cav_id_list": [[0, 1], [0, 1]],
    }
    cfg = PoseProviderConfig(
        enabled=True,
        mode="register_and_fuse",
        apply_to="non-ego",
        freeze_ego=True,
        override_map=override_map,
    )
    applied = apply_overrides_to_batch(batch, cfg)
    assert applied, "multibatch override should apply"
    assert float(batch["lidar_pose"][0, 0]) == 0.0
    assert float(batch["lidar_pose"][2, 0]) == 0.0
    assert abs(float(batch["lidar_pose"][1, 0]) - 10.0) < 1e-6
    assert abs(float(batch["lidar_pose"][3, 0]) - 20.0) < 1e-6
    assert abs(float(batch["pose_confidence"][1]) - 0.5) < 1e-6
    assert abs(float(batch["pose_confidence"][3]) - 0.2) < 1e-6


def test_legacy_pose_override_config_parse():
    hypes = {
        "pose_provider": {"enabled": True, "mode": "register_and_fuse"},
        "pose_override": {
            "pose_field": "foo_pose",
            "confidence_field": "foo_conf",
            "apply_to": "all",
            "freeze_ego": False,
        },
    }
    cfg = PoseProviderConfig.from_hypes(hypes)
    assert cfg.enabled is True
    assert cfg.mode == "register_and_fuse"
    assert cfg.pose_field == "foo_pose"
    assert cfg.confidence_field == "foo_conf"
    assert cfg.apply_to == "all"
    assert cfg.freeze_ego is False


def test_fusion_and_trainparams_fallback_parse():
    hypes = {
        "pose_provider": {"enabled": True, "mode": "register_and_fuse"},
        "fusion": {"args": {"proj_first": True}},
        "train_params": {"max_cav": 7},
    }
    cfg = PoseProviderConfig.from_hypes(hypes)
    assert cfg.proj_first is True
    assert cfg.max_cav == 7


def test_runtime_mode_and_solver_backend_parse():
    hypes = {
        "pose_provider": {
            "enabled": True,
            "mode": "register_only",
            "runtime_mode": "fusion_only",
            "pose_source": "gt",
            "solver_backend": "online_box",
            "online_method": "v2xregpp",
            "stage1_result": "/tmp/stage1.json",
            "online_args": {"min_matches": 3},
        },
    }
    cfg = PoseProviderConfig.from_hypes(hypes)
    # runtime_mode takes precedence and maps fusion_only+gt to legacy gt_only behavior.
    assert cfg.runtime_mode == "fusion_only"
    assert cfg.mode == "gt_only"
    assert cfg.pose_source == "gt"
    assert cfg.solver_backend == "online_box"
    assert cfg.online_method == "v2xregpp"
    assert cfg.online_stage1_result_path == "/tmp/stage1.json"
    assert int(cfg.online_args.get("min_matches")) == 3


def test_online_oracle_gt_fast_path_avoids_cpu_fallback():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    poses_clean = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 20.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "lidar_pose_clean": poses_clean.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="register_and_fuse",
        solver_backend="online_box",
        online_method="gt",
        online_args={"freeze_ego": True},
        recompute_pairwise=False,
    )
    apply_pose_provider(batch, cfg)
    assert torch.allclose(batch["lidar_pose"][0], poses[0]), "ego should remain unchanged"
    assert torch.allclose(batch["lidar_pose"][1], poses_clean[1]), "non-ego should use clean pose"
    timing = batch.get("pose_timing") or {}
    assert int(timing.get("cpu_fallback_count", -1)) == 0, f"expected cpu_fallback_count=0, got {timing}"


def test_fusion_only_identity_pairwise():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [8.0, 1.0, 0.0, 0.0, 20.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="fusion_only",
        pose_source="identity",
        recompute_pairwise=True,
        max_cav=2,
    )
    apply_pose_provider(batch, cfg)
    expected = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    diff = torch.max(torch.abs(batch["pairwise_t_matrix"] - expected)).item()
    assert diff <= 1e-6, f"fusion_only(identity) should build identity pairwise, got {diff}"


def test_online_backend_gt_corrector_updates_pose():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    poses_clean = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 20.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "lidar_pose_clean": poses_clean.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="register_and_fuse",
        solver_backend="online_box",
        online_method="gt",
        online_args={"freeze_ego": True},
        recompute_pairwise=True,
        max_cav=2,
    )
    apply_pose_provider(batch, cfg)
    assert abs(float(batch["lidar_pose"][0, 0]) - 0.0) < 1e-6
    assert abs(float(batch["lidar_pose"][1, 0]) - 2.0) < 1e-6
    timing = batch.get("pose_timing") or {}
    # online_method=="gt" is an oracle path that should not fall back to CPU solvers.
    assert int(timing.get("cpu_fallback_count", -1)) == 0
    assert str(timing.get("solver_backend")) == "online_box"




def _box3d_from_center(cx, cy, cz=0.0, dx=4.0, dy=2.0, dz=1.5):
    x0 = float(cx) - float(dx) / 2.0
    x1 = float(cx) + float(dx) / 2.0
    y0 = float(cy) - float(dy) / 2.0
    y1 = float(cy) + float(dy) / 2.0
    z0 = float(cz)
    z1 = float(cz) + float(dz)
    return [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]


def _boxes_from_centers(centers):
    return [_box3d_from_center(cx, cy) for cx, cy in centers]


def test_online_backend_gpu_stage1_solver_updates_pose_without_cpu_fallback():
    ego_centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    cav_centers = [(-2.0, 0.0), (8.0, 0.0), (-2.0, 10.0)]
    stage1_entry = {
        "cav_id_list": [0, 1],
        "pred_corner3d_np_list": [_boxes_from_centers(ego_centers), _boxes_from_centers(cav_centers)],
        "pred_score_np_list": [[0.95, 0.90, 0.85], [0.98, 0.93, 0.88]],
    }

    batch = {
        "lidar_pose": torch.zeros((2, 6), dtype=torch.float32),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="register_and_fuse",
        solver_backend="online_box",
        online_method="v2xregpp",
        online_stage1_result={"0": stage1_entry},
        online_args={
            "min_matches": 3,
            "max_match_distance_m": 5.0,
            "max_boxes": 16,
            "use_score_weight": True,
            "gpu_stage1_solver": True,
        },
        recompute_pairwise=True,
        max_cav=2,
    )

    apply_pose_provider(batch, cfg)

    # The synthetic payload encodes cav->ego translation (+2m on x).
    assert abs(float(batch["lidar_pose"][1, 0]) - 2.0) <= 1e-2
    timing = batch.get("pose_timing") or {}
    assert int(timing.get("cpu_fallback_count", -1)) == 0
    assert bool(timing.get("pose_provider_applied", False)) is True
    assert float(batch.get("pose_confidence", torch.ones(2))[1]) > 0.5



def test_online_gt_can_skip_pairwise_rebuild_for_strict_parity():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    poses_clean = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 20.0, 0.0]],
        dtype=torch.float32,
    )
    pairwise_seed = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    batch = {
        "lidar_pose": poses.clone(),
        "lidar_pose_clean": poses_clean.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
        "pairwise_t_matrix": pairwise_seed.clone(),
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="register_and_fuse",
        solver_backend="online_box",
        online_method="gt",
        online_args={"freeze_ego": True, "skip_pairwise_rebuild": True},
        recompute_pairwise=True,
        max_cav=2,
    )
    apply_pose_provider(batch, cfg)
    assert abs(float(batch["lidar_pose"][1, 0]) - 2.0) < 1e-6
    assert torch.max(torch.abs(batch["pairwise_t_matrix"] - pairwise_seed)).item() <= 1e-6
    timing = batch.get("pose_timing") or {}
    assert float(timing.get("pairwise_rebuild_sec", 0.0)) <= 1e-8


def test_single_only_does_not_override_or_rebuild_by_default():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 5.0, 0.0]],
        dtype=torch.float32,
    )
    pairwise_seed = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    batch = {
        "lidar_pose": poses.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
        "pairwise_t_matrix": pairwise_seed.clone(),
    }
    override_map = {
        "0": {
            "cav_id_list": [0, 1],
            "lidar_pose_pred_np": [[0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]],
        }
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="single_only",
        recompute_pairwise=False,
        override_map=override_map,
    )
    apply_pose_provider(batch, cfg)
    assert torch.max(torch.abs(batch["lidar_pose"] - poses)).item() <= 1e-6
    assert torch.max(torch.abs(batch["pairwise_t_matrix"] - pairwise_seed)).item() <= 1e-6
    timing = batch.get("pose_timing") or {}
    assert str(timing.get("runtime_mode")) == "single_only"


def test_register_only_applies_override_and_rebuilds_pairwise():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
    }
    override_map = {
        "0": {
            "cav_id_list": [0, 1],
            "lidar_pose_pred_np": [[0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 15, 0]],
            "pose_confidence_np": [1.0, 0.8],
        }
    }
    cfg = PoseProviderConfig(
        enabled=True,
        runtime_mode="register_only",
        recompute_pairwise=False,
        override_map=override_map,
        max_cav=2,
    )
    apply_pose_provider(batch, cfg)
    assert abs(float(batch["lidar_pose"][1, 0]) - 3.0) < 1e-6
    expected = get_pairwise_transformation_torch(batch["lidar_pose"], 2, batch["record_len"], dof=6)
    diff = torch.max(torch.abs(batch["pairwise_t_matrix"] - expected)).item()
    assert diff <= 1e-5, "register_only should rebuild pairwise from updated pose"


def test_attach_pose_confidence_defaults_and_formula():
    if np is None:
        print("SKIP: numpy not available, skip pose confidence test")
        return
    data = {
        0: {"params": {"lidar_pose": [0, 0, 0, 0, 0, 0], "lidar_pose_clean": [3, 4, 0, 0, 0, 0]}},
        1: {"params": {"lidar_pose": [1, 1, 0, 0, 0, 0]}},
    }
    attach_pose_confidence(data)
    conf0 = float(data[0]["params"]["pose_confidence"])
    conf1 = float(data[1]["params"]["pose_confidence"])
    assert abs(conf0 - (1.0 / 26.0)) < 1e-6, f"unexpected confidence formula output: {conf0}"
    assert abs(conf1 - 1.0) < 1e-6, f"missing-clean confidence should default to 1.0, got {conf1}"


def test_dropout_uniform_and_last_pose_reuse():
    if np is None:
        print("SKIP: numpy not available, skip dropout parity test")
        return
    if hasattr(add_noise_data_dict, "_dropout_state"):
        delattr(add_noise_data_dict, "_dropout_state")

    base_frame = {
        0: {"ego": True, "params": {"lidar_pose": np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)}},
        1: {"ego": False, "params": {"lidar_pose": np.array([5, 0, 0, 0, 0, 0], dtype=np.float32)}},
        2: {"ego": False, "params": {"lidar_pose": np.array([9, 0, 0, 0, 0, 0], dtype=np.float32)}},
    }
    no_dropout = {
        "add_noise": True,
        "args": {
            "pos_std": 0.0,
            "rot_std": 0.0,
            "pos_mean": 0.0,
            "rot_mean": 0.0,
            "target": "all",
            "dropout_prob": 0.0,
        },
    }
    add_noise_data_dict(base_frame, no_dropout)
    expected = {cid: np.asarray(content["params"]["lidar_pose"]).copy() for cid, content in base_frame.items()}

    frame_with_new_pose = {
        0: {"ego": True, "params": {"lidar_pose": np.array([100, 0, 0, 0, 0, 0], dtype=np.float32)}},
        1: {"ego": False, "params": {"lidar_pose": np.array([101, 0, 0, 0, 0, 0], dtype=np.float32)}},
        2: {"ego": False, "params": {"lidar_pose": np.array([102, 0, 0, 0, 0, 0], dtype=np.float32)}},
    }
    force_dropout = {
        "add_noise": True,
        "args": {
            "pos_std": 0.0,
            "rot_std": 0.0,
            "pos_mean": 0.0,
            "rot_mean": 0.0,
            "target": "all",
            "dropout_prob": 1.0,
        },
    }
    add_noise_data_dict(frame_with_new_pose, force_dropout)

    for cid, content in frame_with_new_pose.items():
        pose_now = np.asarray(content["params"]["lidar_pose"], dtype=np.float32)
        assert np.allclose(pose_now, expected[cid]), f"dropout should reuse last pose for cav {cid}"
        assert float(content["params"].get("pose_confidence", 1.0)) == 0.0, "dropout should set confidence=0.0"


def test_pose_provider_timing_payload_present():
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    batch = {
        "lidar_pose": poses.clone(),
        "record_len": torch.tensor([2], dtype=torch.int64),
        "pairwise_t_matrix": torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32),
        "sample_idx": 0,
        "cav_id_list": [0, 1],
    }
    cfg = PoseProviderConfig(
        enabled=True,
        mode="register_and_fuse",
        recompute_pairwise=True,
        max_cav=2,
        proj_first=False,
    )
    apply_pose_provider(batch, cfg)
    timing = batch.get("pose_timing")
    assert isinstance(timing, dict), "pose_timing payload should be a dict"
    for key in ["pose_provider_total_sec", "pose_override_sec", "pairwise_rebuild_sec", "online_solver_sec"]:
        assert key in timing, f"missing timing field: {key}"
        assert float(timing[key]) >= 0.0, f"timing field should be non-negative: {key}={timing[key]}"
    assert "cpu_fallback_count" in timing
    assert int(timing["cpu_fallback_count"]) >= 0
    assert str(timing.get("runtime_mode")) in {"register_and_fuse", "register_only", "fusion_only", "single_only"}


def main():
    test_pairwise_parity()
    test_override_non_ego()
    test_gt_only_pairwise()
    test_proj_first_identity_and_label_sync()
    test_multibatch_override_with_string_keys()
    test_legacy_pose_override_config_parse()
    test_fusion_and_trainparams_fallback_parse()
    test_runtime_mode_and_solver_backend_parse()
    test_fusion_only_identity_pairwise()
    test_online_backend_gt_corrector_updates_pose()
    test_online_backend_gpu_stage1_solver_updates_pose_without_cpu_fallback()
    test_online_gt_can_skip_pairwise_rebuild_for_strict_parity()
    test_single_only_does_not_override_or_rebuild_by_default()
    test_register_only_applies_override_and_rebuilds_pairwise()
    test_attach_pose_confidence_defaults_and_formula()
    test_dropout_uniform_and_last_pose_reuse()
    test_pose_provider_timing_payload_present()
    print("OK")


if __name__ == "__main__":
    main()
