from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .io_utils import SamplePair
from .robot_pose_parser import RobotPoseSample


@dataclass
class ValidationStats:
    mean_mm: float
    median_mm: float
    p95_mm: float
    max_mm: float
    count: int


@dataclass
class PoseConsistencyStats:
    position_rmse_mm: float
    orientation_rmse_deg: float
    position_mean_mm: float
    orientation_mean_deg: float
    count: int


def _target_origin_in_base(T_cam2base: np.ndarray, t_target2cam: np.ndarray) -> np.ndarray:
    p_cam = np.asarray(t_target2cam, dtype=np.float64).reshape(3)
    p_base = T_cam2base[:3, :3] @ p_cam + T_cam2base[:3, 3]
    return p_base


def validate_translation_error(
    T_cam2base: np.ndarray,
    robot_samples: list[RobotPoseSample],
    pairs: list[SamplePair],
    target_offset_gripper_m: np.ndarray | None = None,
) -> tuple[ValidationStats, list[float]]:
    """Validate by comparing transformed target origin with robot-side reference.

    By default, assumes calibration target origin coincides with gripper origin.
    If target_offset_gripper_m is provided (3,), it will be transformed by gripper pose.
    """
    errs_mm: list[float] = []

    off = np.zeros(3, dtype=np.float64)
    if target_offset_gripper_m is not None:
        off = np.asarray(target_offset_gripper_m, dtype=np.float64).reshape(3)

    for rs, pair in zip(robot_samples, pairs):
        p_from_cam = _target_origin_in_base(T_cam2base, pair.t_target2cam)
        p_ref = rs.R_gripper2base @ off + rs.t_gripper2base
        err_mm = float(np.linalg.norm(p_from_cam - p_ref) * 1000.0)
        errs_mm.append(err_mm)

    arr = np.asarray(errs_mm, dtype=np.float64)
    stats = ValidationStats(
        mean_mm=float(np.mean(arr)),
        median_mm=float(np.median(arr)),
        p95_mm=float(np.percentile(arr, 95)),
        max_mm=float(np.max(arr)),
        count=len(errs_mm),
    )
    return stats, errs_mm


def validate_target_in_gripper_consistency(
    T_cam2base: np.ndarray,
    robot_samples: list[RobotPoseSample],
    pairs: list[SamplePair],
) -> PoseConsistencyStats:
    """Estimate target->gripper constancy and return residual RMSE metrics.

    For each sample i:
      T_t2g_i = inv(T_g2b_i) @ (T_c2b @ T_t2c_i)
    If calibration and robot poses are consistent, T_t2g_i should be constant.
    """
    if len(robot_samples) != len(pairs):
        raise ValueError("robot_samples and pairs must have same length")
    if not pairs:
        raise ValueError("at least one sample is required")

    t_list: list[np.ndarray] = []
    rot_list: list[Rotation] = []

    R_c2b = np.asarray(T_cam2base[:3, :3], dtype=np.float64)
    t_c2b = np.asarray(T_cam2base[:3, 3], dtype=np.float64).reshape(3)

    for rs, pair in zip(robot_samples, pairs):
        R_g2b = np.asarray(rs.R_gripper2base, dtype=np.float64)
        t_g2b = np.asarray(rs.t_gripper2base, dtype=np.float64).reshape(3)
        R_t2c = np.asarray(pair.R_target2cam, dtype=np.float64)
        t_t2c = np.asarray(pair.t_target2cam, dtype=np.float64).reshape(3)

        R_t2b = R_c2b @ R_t2c
        t_t2b = R_c2b @ t_t2c + t_c2b

        R_b2g = R_g2b.T
        t_b2g = -R_b2g @ t_g2b

        R_t2g = R_b2g @ R_t2b
        t_t2g = R_b2g @ t_t2b + t_b2g

        t_list.append(t_t2g)
        rot_list.append(Rotation.from_matrix(R_t2g))

    t_stack = np.vstack([t.reshape(1, 3) for t in t_list])
    t_mean = np.mean(t_stack, axis=0)
    try:
        rot_mean = Rotation.concatenate(rot_list).mean()
    except Exception:
        quats = np.vstack([r.as_quat() for r in rot_list])
        # Align quaternion hemisphere before averaging to avoid cancellation.
        for i in range(1, quats.shape[0]):
            if np.dot(quats[0], quats[i]) < 0.0:
                quats[i] = -quats[i]
        q_mean = np.mean(quats, axis=0)
        rot_mean = Rotation.from_quat(q_mean / np.linalg.norm(q_mean))

    trans_err_mm = np.linalg.norm(t_stack - t_mean.reshape(1, 3), axis=1) * 1000.0
    rot_err_deg = []
    for r in rot_list:
        rel = rot_mean.inv() * r
        rot_err_deg.append(float(np.degrees(rel.magnitude())))

    trans_arr = np.asarray(trans_err_mm, dtype=np.float64)
    rot_arr = np.asarray(rot_err_deg, dtype=np.float64)

    return PoseConsistencyStats(
        position_rmse_mm=float(np.sqrt(np.mean(np.square(trans_arr)))),
        orientation_rmse_deg=float(np.sqrt(np.mean(np.square(rot_arr)))),
        position_mean_mm=float(np.mean(trans_arr)),
        orientation_mean_deg=float(np.mean(rot_arr)),
        count=len(pairs),
    )
