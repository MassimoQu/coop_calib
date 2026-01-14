# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.distributions as dist


def pose_xy_error(pose: np.ndarray, pose_clean: np.ndarray) -> float:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    pose_clean = np.asarray(pose_clean, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 2 or pose_clean.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(pose[:2] - pose_clean[:2], ord=2))


def pose_confidence_from_epsilon(epsilon: float) -> float:
    eps = float(epsilon)
    return float(1.0 / (1.0 + eps * eps))


def attach_pose_confidence(data_dict, *, key: str = "pose_confidence") -> None:
    """
    Attach a scalar pose confidence to each CAV if missing.

    Following V2VLoc Eq.(6): sigma = 1 / (1 + epsilon^2).
    Here epsilon is approximated by XY translation error between lidar_pose and lidar_pose_clean.
    """
    for _, cav_content in data_dict.items():
        params = cav_content.get("params") or {}
        if key in params:
            continue
        pose = params.get("lidar_pose")
        pose_clean = params.get("lidar_pose_clean")
        if pose is None or pose_clean is None:
            params[key] = 1.0
            continue
        eps = pose_xy_error(pose, pose_clean)
        params[key] = pose_confidence_from_epsilon(eps)
        cav_content["params"] = params


def add_noise_data_dict(data_dict, noise_setting):
    """ Update the base data dict. 
        We retrieve lidar_pose and add_noise to it.
        And set a clean pose.
    """
    noise_args = noise_setting.get('args', {}) if isinstance(noise_setting, dict) else {}
    target = noise_args.get('target', 'all') if isinstance(noise_args, dict) else 'all'

    def _should_apply_noise(cav_id, cav_content) -> bool:
        if target is None or target == 'all':
            return True
        if target in {'ego', 'self'}:
            return bool(cav_content.get('ego', False))
        if target in {'non-ego', 'cav', 'other', 'others'}:
            return not bool(cav_content.get('ego', False))
        if isinstance(target, (list, tuple, set)):
            return cav_id in target
        try:
            return int(cav_id) == int(target)
        except Exception:
            # Backward compatible fallback: if unknown target, apply to all.
            return True

    if noise_setting['add_noise']:
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

            if not _should_apply_noise(cav_id, cav_content):
                continue

            if "laplace" in noise_setting['args'].keys() and noise_setting['args']['laplace'] is True:
                cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                        generate_noise_laplace( # we just use the same key name
                                                            noise_setting['args']['pos_std'],
                                                            noise_setting['args']['rot_std'],
                                                            noise_setting['args']['pos_mean'],
                                                            noise_setting['args']['rot_mean']
                                                        )
            else:
                cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                            generate_noise(
                                                                noise_setting['args']['pos_std'],
                                                                noise_setting['args']['rot_std'],
                                                                noise_setting['args']['pos_mean'],
                                                                noise_setting['args']['rot_mean']
                                                            )

    else:
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

            
    return data_dict

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.
    
    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    
    return pose_noise



def generate_noise_laplace(pos_b, rot_b, pos_mu=0, rot_mu=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use laplace distribution to generate noise.
    
    Args:

        pos_b : float 
            parameter b of laplace dist, in meter

        rot_b : float
            parameter b of laplace dist, in degree

        pos_mu : float
            mean of laplace dist, in meter

        rot_mu : float
            mean of laplace dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.laplace(pos_mu, pos_b, size=(2))
    yaw = np.random.laplace(rot_mu, rot_b, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])
    return pose_noise


def generate_noise_torch(pose, pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ only used for v2vnet robust.
        rotation noise is sampled from von_mises distribution
    
    Args:
        pose : Tensor, [N. 6]
            including [x, y, z, roll, yaw, pitch]

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noisy: Tensor, [N, 6]
            noisy pose
    """

    N = pose.shape[0]
    noise = torch.zeros_like(pose, device=pose.device)
    concentration = (180 / (np.pi * rot_std)) ** 2

    noise[:, :2] = torch.normal(pos_mean, pos_std, size=(N, 2), device=pose.device)
    noise[:, 4] = dist.von_mises.VonMises(loc=rot_mean, concentration=concentration).sample((N,)).to(noise.device)


    return noise


def remove_z_axis(T):
    """ remove rotation/translation related to z-axis
    Args:
        T: np.ndarray
            [4, 4]
    Returns:
        T: np.ndarray
            [4, 4]
    """
    T[2,3] = 0 # z-trans
    T[0,2] = 0
    T[1,2] = 0
    T[2,0] = 0
    T[2,1] = 0
    T[2,2] = 1
    
    return T
