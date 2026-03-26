from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RobotPoseSample:
    index: int
    raw_row: list[str]
    t_gripper2base: np.ndarray  # (3,)
    R_gripper2base: np.ndarray  # (3,3)


def _to_float(row: list[str], col: int, name: str) -> float:
    try:
        return float(row[col])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid {name} at col={col}: {row}") from exc


def load_robot_pose_csv(
    csv_path: str,
    delimiter: str = ",",
    has_header: bool = False,
    pose_columns: dict | None = None,
    unit_scale: float = 0.001,
    euler_order: str = "xyz",
) -> list[RobotPoseSample]:
    """Load robot poses from CSV and convert to SE(3).

    Expected per row fields include X,Y,Z,Rx,Ry,Rz, and optional trailing status.
    The pose_columns map can override indices.
    """
    cols = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if pose_columns:
        cols.update(pose_columns)

    samples: list[RobotPoseSample] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)

        for i, row in enumerate(reader):
            if not row:
                continue

            x = _to_float(row, cols["x"], "x") * unit_scale
            y = _to_float(row, cols["y"], "y") * unit_scale
            z = _to_float(row, cols["z"], "z") * unit_scale
            rx = _to_float(row, cols["rx"], "rx")
            ry = _to_float(row, cols["ry"], "ry")
            rz = _to_float(row, cols["rz"], "rz")

            R = Rotation.from_euler(euler_order, [rx, ry, rz], degrees=True).as_matrix()
            t = np.array([x, y, z], dtype=np.float64)

            samples.append(
                RobotPoseSample(
                    index=i,
                    raw_row=row,
                    t_gripper2base=t,
                    R_gripper2base=R.astype(np.float64),
                )
            )

    if not samples:
        raise ValueError(f"no valid rows in {csv_path}")

    return samples


def invert_pose(R_gripper2base: np.ndarray, t_gripper2base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute base->gripper from gripper->base."""
    R_base2gripper = R_gripper2base.T
    t_base2gripper = -R_base2gripper @ t_gripper2base.reshape(3)
    return R_base2gripper, t_base2gripper


def make_base2gripper_lists(samples: list[RobotPoseSample]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    R_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    for s in samples:
        R_b2g, t_b2g = invert_pose(s.R_gripper2base, s.t_gripper2base)
        R_list.append(R_b2g)
        t_list.append(t_b2g.reshape(3, 1))
    return R_list, t_list
