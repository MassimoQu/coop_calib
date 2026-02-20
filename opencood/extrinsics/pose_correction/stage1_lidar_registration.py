from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.late_fusion.lidar_registration import (
    LidarRegistrationConfig,
    LidarRegistrationEstimator,
)
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose


def _wrap_angle_deg(angle: float) -> float:
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _delta_angle_deg(a: float, b: float) -> float:
    return _wrap_angle_deg(float(a) - float(b))


@dataclass
class Stage1LidarRegPoseCorrector:
    """
    Pose correction using LiDAR registration (FPFH + RANSAC/FGR + ICP).
    """

    cfg: LidarRegistrationConfig = field(default_factory=LidarRegistrationConfig)
    mode: str = "initfree"  # initfree | stable
    min_fitness: float = 0.0
    max_inlier_rmse: float = 0.0
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    freeze_ego: bool = True
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = LidarRegistrationEstimator(cfg=self.cfg)
        mode = str(self.mode or "initfree").lower().strip()
        if mode not in {"initfree", "stable"}:
            mode = "initfree"
        self.mode = mode
        self.min_fitness = float(self.min_fitness or 0.0)
        self.max_inlier_rmse = float(self.max_inlier_rmse or 0.0)
        self.ema_alpha = float(np.clip(float(self.ema_alpha or 0.0), 0.0, 1.0))
        self.max_step_xy_m = float(self.max_step_xy_m or 0.0)
        self.max_step_yaw_deg = float(self.max_step_yaw_deg or 0.0)
        self.freeze_ego = bool(self.freeze_ego)

    def _reset_if_new_epoch(self, sample_idx: int) -> None:
        last = self._state.get("last_sample_idx")
        if last is None or int(sample_idx) < int(last):
            self._state.clear()
        self._state["last_sample_idx"] = int(sample_idx)

    def _get_prev_delta(self, key: str) -> Optional[Tuple[float, float, float]]:
        prev = self._state.get("delta_se2", {}).get(key)
        if prev is None:
            return None
        try:
            x, y, yaw = float(prev[0]), float(prev[1]), float(prev[2])
        except Exception:
            return None
        return x, y, yaw

    def _set_prev_delta(self, key: str, delta: Tuple[float, float, float]) -> None:
        store = self._state.setdefault("delta_se2", {})
        store[key] = (float(delta[0]), float(delta[1]), float(delta[2]))

    def _smooth_se2(
        self, prev: Tuple[float, float, float], cur: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        a = self.ema_alpha
        if a <= 0.0:
            return prev
        if a >= 1.0:
            return cur
        x = (1.0 - a) * prev[0] + a * cur[0]
        y = (1.0 - a) * prev[1] + a * cur[1]
        dyaw = _delta_angle_deg(cur[2], prev[2])
        yaw = _wrap_angle_deg(prev[2] + a * dyaw)
        return float(x), float(y), float(yaw)

    def _limit_step_se2(self, prev: Tuple[float, float, float], cur: Tuple[float, float, float]) -> bool:
        if self.max_step_xy_m > 0.0:
            dx = float(cur[0] - prev[0])
            dy = float(cur[1] - prev[1])
            if float(np.hypot(dx, dy)) > self.max_step_xy_m + 1e-6:
                return False
        if self.max_step_yaw_deg > 0.0:
            if abs(_delta_angle_deg(cur[2], prev[2])) > self.max_step_yaw_deg + 1e-6:
                return False
        return True

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
    ) -> bool:
        self._reset_if_new_epoch(int(sample_idx))
        if not cav_id_list:
            return False
        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        ego_entry = base_data_dict[ego_id]
        ego_points = ego_entry.get("lidar_np")
        if ego_points is None:
            return False

        mode = str(self.mode or "initfree").lower().strip()
        stable = mode == "stable"

        ego_pose_current = ego_entry["params"]["lidar_pose"]
        ego_T_world_current = pose_to_tfm(np.asarray([ego_pose_current], dtype=np.float64))[0]
        updated_any = False

        for cav_id in cav_id_list:
            if cav_id == ego_id and self.freeze_ego:
                continue
            cav_entry = base_data_dict.get(cav_id)
            if not isinstance(cav_entry, Mapping):
                continue
            cav_points = cav_entry.get("lidar_np")
            if cav_points is None:
                continue

            estimate = self._estimator.estimate_from_points(cav_points, ego_points)
            if not estimate.success or estimate.T is None:
                continue

            fitness = float(estimate.extra.get("fitness", 0.0))
            rmse = float(estimate.extra.get("inlier_rmse", 0.0))
            if self.min_fitness > 0.0 and fitness < self.min_fitness:
                continue
            if self.max_inlier_rmse > 0.0 and rmse > self.max_inlier_rmse:
                continue

            rel_T_est = np.linalg.inv(np.asarray(estimate.T, dtype=np.float64))
            cav_pose_current = cav_entry["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]
            rel_T_current = np.linalg.inv(ego_T_world_current) @ cav_T_world_current

            if not stable:
                rel_T_corrected = rel_T_est
            else:
                rel_key = str(cav_id)
                prev_delta = self._get_prev_delta(rel_key)
                try:
                    delta_T = rel_T_est @ np.linalg.inv(rel_T_current)
                    delta_pose6 = tfm_to_pose(delta_T)
                    delta_xyyaw = (
                        float(delta_pose6[0]),
                        float(delta_pose6[1]),
                        _wrap_angle_deg(float(delta_pose6[4])),
                    )
                except Exception:
                    delta_xyyaw = None

                if delta_xyyaw is not None:
                    if prev_delta is not None:
                        if not self._limit_step_se2(prev_delta, delta_xyyaw):
                            delta_xyyaw = prev_delta
                        else:
                            delta_xyyaw = self._smooth_se2(prev_delta, delta_xyyaw)
                    self._set_prev_delta(rel_key, delta_xyyaw)
                    prev_delta = delta_xyyaw

                if prev_delta is None:
                    continue
                delta_T_se2 = pose_to_tfm(
                    np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64)
                )[0]
                rel_T_corrected = delta_T_se2 @ rel_T_current

            cav_T_world_new = ego_T_world_current @ rel_T_corrected
            cav_pose_new = tfm_to_pose(cav_T_world_new)
            cav_pose = list(cav_entry["params"]["lidar_pose"])
            cav_pose[0] = float(cav_pose_new[0])
            cav_pose[1] = float(cav_pose_new[1])
            cav_pose[4] = float(cav_pose_new[4])
            cav_entry["params"]["lidar_pose"] = cav_pose
            updated_any = True

        return updated_any


__all__ = ["Stage1LidarRegPoseCorrector"]
