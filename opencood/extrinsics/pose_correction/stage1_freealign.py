from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator
from opencood.pose.freealign_repo import FreeAlignRepoConfig, FreeAlignRepoEstimator
from opencood.utils.transformation_utils import tfm_to_pose, x_to_world


def _normalize_agent_role(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    name = str(raw).lower()
    if "veh" in name or "vehicle" in name:
        return "vehicle"
    if "infra" in name or "rsu" in name or "infrastructure" in name:
        return "infrastructure"
    return None


def _infer_dair_role_from_base(base_data_dict: Mapping[Any, Any], cav_id: Any) -> Optional[str]:
    entry = base_data_dict.get(cav_id)
    if not isinstance(entry, Mapping):
        return None
    params = entry.get("params") or {}
    if not isinstance(params, Mapping):
        return None
    vehicles_all = params.get("vehicles_all")
    if isinstance(vehicles_all, list):
        return "vehicle" if len(vehicles_all) > 0 else "infrastructure"
    vehicles_front = params.get("vehicles_front")
    if isinstance(vehicles_front, list):
        return "vehicle" if len(vehicles_front) > 0 else "infrastructure"
    return None


def _extract_agent_indices(
    all_agent_ids: Sequence[Any],
    *,
    ego_id: Any,
    cav_id: Any,
    base_data_dict: Mapping[Any, Any],
) -> Optional[Tuple[int, int]]:
    try:
        ego_idx = all_agent_ids.index(ego_id)
        cav_idx = all_agent_ids.index(cav_id)
        return int(ego_idx), int(cav_idx)
    except ValueError:
        pass

    ego_str = str(ego_id)
    cav_str = str(cav_id)
    ego_idx = None
    cav_idx = None
    for idx, agent_id in enumerate(all_agent_ids):
        if ego_idx is None and str(agent_id) == ego_str:
            ego_idx = int(idx)
        if cav_idx is None and str(agent_id) == cav_str:
            cav_idx = int(idx)
    if ego_idx is not None and cav_idx is not None:
        return int(ego_idx), int(cav_idx)

    role_to_idx: Dict[str, int] = {}
    for idx, agent_id in enumerate(all_agent_ids):
        role = _normalize_agent_role(agent_id)
        if role and role not in role_to_idx:
            role_to_idx[role] = int(idx)
    if role_to_idx:
        ego_role = _infer_dair_role_from_base(base_data_dict, ego_id)
        cav_role = _infer_dair_role_from_base(base_data_dict, cav_id)
        if ego_role and cav_role and ego_role in role_to_idx and cav_role in role_to_idx:
            return int(role_to_idx[ego_role]), int(role_to_idx[cav_role])
    return None


def _extract_boxes(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
    bbox_type: str,
) -> List[object]:
    corners_all = stage1_content.get("pred_corner3d_np_list")
    if not isinstance(corners_all, list) or agent_idx < 0 or agent_idx >= len(corners_all):
        return []
    corners_list = corners_all[agent_idx]
    if not isinstance(corners_list, list) or not corners_list:
        return []

    scores_all = stage1_content.get("pred_score_np_list")
    scores = None
    if isinstance(scores_all, list) and agent_idx < len(scores_all) and isinstance(scores_all[agent_idx], list):
        scores = [float(x) for x in scores_all[agent_idx]]

    corners_np = np.asarray(corners_list, dtype=np.float32)
    if corners_np.ndim != 3 or corners_np.shape[1:] != (8, 3):
        return []
    if scores is not None and len(scores) != int(corners_np.shape[0]):
        scores = None
    return corners_to_bbox3d_list(corners_np, bbox_type=bbox_type, scores=scores)


@dataclass
class Stage1FreeAlignPoseCorrector:
    """
    Apply FreeAlign pose correction using cached stage-1 boxes.

    This keeps FreeAlign logic out of dataset code.
    """

    cfg: FreeAlignPaperConfig = field(default_factory=FreeAlignPaperConfig)
    bbox_type: str = "detected"

    _estimator: FreeAlignPaperEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = FreeAlignPaperEstimator(self.cfg)

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Mapping[str, Any],
    ) -> bool:
        """
        Update `base_data_dict[cav_id]['params']['lidar_pose']` in-place.

        Returns:
            bool: True if any CAV pose was updated, False otherwise.
        """
        key = str(sample_idx)
        if key not in stage1_result:
            return False
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            return False

        all_agent_ids = stage1_content.get("cav_id_list") or []
        if not isinstance(all_agent_ids, list) or len(all_agent_ids) < 2:
            return False
        if not cav_id_list:
            return False

        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        Tw_ego = x_to_world(ego_pose)

        updated = False
        for cav_id in cav_id_list[1:]:
            if cav_id not in base_data_dict:
                continue
            indices = _extract_agent_indices(all_agent_ids, ego_id=ego_id, cav_id=cav_id, base_data_dict=base_data_dict)
            if indices is None:
                continue
            ego_idx, cav_idx = indices

            ego_boxes = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
            cav_boxes = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
            T_ego_cav, stability, matches, meta = self._estimator.estimate(
                cav_boxes=cav_boxes, ego_boxes=ego_boxes, T_init=None
            )
            if T_ego_cav is None:
                continue

            Tw_cav_new = Tw_ego @ np.asarray(T_ego_cav, dtype=np.float64)
            pose_new = tfm_to_pose(Tw_cav_new)
            pose_old = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            if len(pose_old) < 6:
                continue
            pose_old[0] = float(pose_new[0])
            pose_old[1] = float(pose_new[1])
            pose_old[4] = float(pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = pose_old

            updated = True
            _ = (stability, matches, meta)  # keep for optional future logging
        return updated


@dataclass
class Stage1FreeAlignRepoPoseCorrector:
    """
    Apply FreeAlign (released repo matching: match_v7_with_detection) pose correction using cached stage-1 boxes.

    This keeps FreeAlign logic out of dataset code.
    """

    cfg: FreeAlignRepoConfig = field(default_factory=FreeAlignRepoConfig)
    bbox_type: str = "detected"

    _estimator: FreeAlignRepoEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = FreeAlignRepoEstimator(self.cfg)

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Mapping[str, Any],
    ) -> bool:
        """
        Update `base_data_dict[cav_id]['params']['lidar_pose']` in-place.

        Returns:
            bool: True if any CAV pose was updated, False otherwise.
        """
        key = str(sample_idx)
        if key not in stage1_result:
            return False
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            return False

        all_agent_ids = stage1_content.get("cav_id_list") or []
        if not isinstance(all_agent_ids, list) or len(all_agent_ids) < 2:
            return False
        if not cav_id_list:
            return False

        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        Tw_ego = x_to_world(ego_pose)

        updated = False
        for cav_id in cav_id_list[1:]:
            if cav_id not in base_data_dict:
                continue
            indices = _extract_agent_indices(all_agent_ids, ego_id=ego_id, cav_id=cav_id, base_data_dict=base_data_dict)
            if indices is None:
                continue
            ego_idx, cav_idx = indices

            ego_boxes = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
            cav_boxes = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
            T_ego_cav, stability, matches, meta = self._estimator.estimate(cav_boxes=cav_boxes, ego_boxes=ego_boxes)
            if T_ego_cav is None:
                continue

            Tw_cav_new = Tw_ego @ np.asarray(T_ego_cav, dtype=np.float64)
            pose_new = tfm_to_pose(Tw_cav_new)
            pose_old = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            if len(pose_old) < 6:
                continue
            pose_old[0] = float(pose_new[0])
            pose_old[1] = float(pose_new[1])
            pose_old[4] = float(pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = pose_old

            updated = True
            _ = (stability, matches, meta)  # keep for optional future logging
        return updated


__all__ = ["Stage1FreeAlignPoseCorrector", "Stage1FreeAlignRepoPoseCorrector"]
