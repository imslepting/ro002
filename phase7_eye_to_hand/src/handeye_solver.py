from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .robot_pose_parser import RobotPoseSample, make_base2gripper_lists
from .io_utils import SamplePair


_METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeResult:
    method: str
    R_cam2base: np.ndarray
    t_cam2base: np.ndarray
    T_cam2base: np.ndarray



def solve_eye_to_hand(
    robot_samples: list[RobotPoseSample],
    sample_pairs: list[SamplePair],
    method: str = "tsai",
) -> HandEyeResult:
    if len(robot_samples) != len(sample_pairs):
        raise ValueError("robot_samples and sample_pairs must have same length")
    if len(sample_pairs) < 3:
        raise ValueError("need at least 3 samples for hand-eye calibration")

    R_base2gripper, t_base2gripper = make_base2gripper_lists(robot_samples)

    R_target2cam = [np.asarray(p.R_target2cam, dtype=np.float64) for p in sample_pairs]
    t_target2cam = [np.asarray(p.t_target2cam, dtype=np.float64).reshape(3, 1) for p in sample_pairs]

    cv_method = _METHOD_MAP.get(method.lower())
    if cv_method is None:
        raise ValueError(f"unsupported hand-eye method: {method}")

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper,
        t_base2gripper,
        R_target2cam,
        t_target2cam,
        method=cv_method,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.reshape(3)

    return HandEyeResult(
        method=method.lower(),
        R_cam2base=R_cam2base.astype(np.float64),
        t_cam2base=t_cam2base.reshape(3, 1).astype(np.float64),
        T_cam2base=T,
    )
