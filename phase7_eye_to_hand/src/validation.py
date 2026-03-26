from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import SamplePair
from .robot_pose_parser import RobotPoseSample


@dataclass
class ValidationStats:
    mean_mm: float
    median_mm: float
    p95_mm: float
    max_mm: float
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
