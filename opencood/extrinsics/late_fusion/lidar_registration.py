from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


@dataclass(frozen=True)
class LidarRegistrationConfig:
    voxel_size_m: float = 1.0
    max_corr_dist_m: float = 2.0
    ransac_n: int = 4
    ransac_max_iter: int = 50000
    ransac_confidence: float = 0.999
    use_fgr: bool = False
    icp_method: str = "point_to_plane"  # point_to_plane | point_to_point | gicp
    icp_max_iter: int = 50
    min_points: int = 200
    max_points: int = 60000


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def _compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.pipelines.registration.Feature:
    radius_normal = max(1e-3, voxel_size * 2.0)
    radius_feature = max(1e-3, voxel_size * 5.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )


class LidarRegistrationEstimator:
    """
    Coarse-to-fine LiDAR registration using FPFH + RANSAC/FGR + ICP.
    Returns a source->target transform in LiDAR coordinates.
    """

    def __init__(self, *, cfg: Optional[LidarRegistrationConfig] = None) -> None:
        self._cfg = cfg or LidarRegistrationConfig()

    @property
    def config(self) -> LidarRegistrationConfig:
        return self._cfg

    def estimate_from_points(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        ctx = ctx or MethodContext()
        start = perf_counter()
        cfg = self._cfg

        if src_points is None or dst_points is None:
            return ExtrinsicEstimate(T=None, success=False, method="lidar_reg", extra={"reason": "missing_points"})

        src = np.asarray(src_points, dtype=np.float32).reshape(-1, src_points.shape[-1])
        dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, dst_points.shape[-1])
        if src.shape[0] < cfg.min_points or dst.shape[0] < cfg.min_points:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="lidar_reg",
                extra={"reason": "too_few_points", "src_points": int(src.shape[0]), "dst_points": int(dst.shape[0])},
            )

        src = _downsample_points(src, int(cfg.max_points))
        dst = _downsample_points(dst, int(cfg.max_points))

        pcd_src = _to_o3d(src)
        pcd_dst = _to_o3d(dst)

        voxel_size = float(cfg.voxel_size_m)
        if voxel_size > 0:
            pcd_src_down = pcd_src.voxel_down_sample(voxel_size)
            pcd_dst_down = pcd_dst.voxel_down_sample(voxel_size)
        else:
            pcd_src_down = pcd_src
            pcd_dst_down = pcd_dst

        if len(pcd_src_down.points) < cfg.min_points or len(pcd_dst_down.points) < cfg.min_points:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="lidar_reg",
                extra={"reason": "too_few_points_downsampled"},
            )

        fpfh_src = _compute_fpfh(pcd_src_down, max(1e-3, voxel_size))
        fpfh_dst = _compute_fpfh(pcd_dst_down, max(1e-3, voxel_size))

        max_corr = max(1e-3, float(cfg.max_corr_dist_m))
        if cfg.use_fgr:
            coarse = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                pcd_src_down,
                pcd_dst_down,
                fpfh_src,
                fpfh_dst,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=max_corr,
                ),
            )
        else:
            coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_src_down,
                pcd_dst_down,
                fpfh_src,
                fpfh_dst,
                mutual_filter=True,
                max_correspondence_distance=max_corr,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=int(cfg.ransac_n),
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(cfg.ransac_max_iter), float(cfg.ransac_confidence)
                ),
            )

        init_T = coarse.transformation
        if init is not None and init.T_init is not None:
            init_T = np.asarray(init.T_init, dtype=np.float64)

        icp_method = str(cfg.icp_method or "point_to_plane").lower().strip()
        if icp_method == "gicp":
            estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        elif icp_method == "point_to_point":
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        if icp_method != "point_to_point":
            radius = max(1e-3, voxel_size * 2.0)
            pcd_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            pcd_dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

        icp = o3d.pipelines.registration.registration_icp(
            pcd_src,
            pcd_dst,
            max_corr,
            init_T,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg.icp_max_iter)),
        )

        T = np.asarray(icp.transformation, dtype=np.float64)
        time_sec = float(perf_counter() - start)

        stability = float(icp.fitness) if icp is not None else 0.0
        return ExtrinsicEstimate(
            T=T,
            success=True,
            method="lidar_reg",
            stability=stability,
            time_sec=time_sec,
            extra={
                "fitness": float(icp.fitness),
                "inlier_rmse": float(icp.inlier_rmse),
            },
        )


__all__ = ["LidarRegistrationConfig", "LidarRegistrationEstimator"]
