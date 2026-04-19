from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class CaptureRecord:
    """Raw capture: image path + robot state (no Charuco analysis yet)."""
    sample_index: int
    image_path: str
    R_gripper2base: np.ndarray  # (3,3)
    t_gripper2base: np.ndarray  # (3,)
    robot_euler_deg: np.ndarray | None = None  # (3,), raw [rx, ry, rz]
    euler_order_used: str | None = None


@dataclass
class SamplePair:
    sample_index: int
    robot_row_index: int
    image_path: str
    R_target2cam: np.ndarray
    t_target2cam: np.ndarray
    num_corners: int
    # Optional real-time robot state (when captured from robot, not CSV)
    R_gripper2base: np.ndarray | None = None  # (3,3)
    t_gripper2base: np.ndarray | None = None  # (3,)
    robot_euler_deg: np.ndarray | None = None  # (3,), raw [rx, ry, rz]
    euler_order_used: str | None = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_capture_records_jsonl(records: list[CaptureRecord], out_path: str) -> None:
    """Save raw capture records (image path + robot state, no Charuco analysis)."""
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "sample_index": r.sample_index,
                "image_path": r.image_path,
                "R_gripper2base": r.R_gripper2base,
                "t_gripper2base": r.t_gripper2base,
            }
            if r.robot_euler_deg is not None:
                row["robot_euler_deg"] = r.robot_euler_deg
            if r.euler_order_used is not None:
                row["euler_order_used"] = r.euler_order_used
            f.write(json.dumps(row, ensure_ascii=False, cls=NumpyEncoder) + "\n")


def load_capture_records_jsonl(path: str) -> list[CaptureRecord]:
    """Load raw capture records."""
    records: list[CaptureRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                CaptureRecord(
                    sample_index=int(row["sample_index"]),
                    image_path=str(row["image_path"]),
                    R_gripper2base=np.asarray(row["R_gripper2base"], dtype=np.float64).reshape(3, 3),
                    t_gripper2base=np.asarray(row["t_gripper2base"], dtype=np.float64).reshape(3),
                    robot_euler_deg=(
                        np.asarray(row["robot_euler_deg"], dtype=np.float64).reshape(3)
                        if "robot_euler_deg" in row else None
                    ),
                    euler_order_used=str(row["euler_order_used"]) if "euler_order_used" in row else None,
                )
            )
    return records



def save_sample_pairs_jsonl(pairs: list[SamplePair], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            row = {
                "sample_index": p.sample_index,
                "robot_row_index": p.robot_row_index,
                "image_path": p.image_path,
                "R_target2cam": p.R_target2cam,
                "t_target2cam": p.t_target2cam,
                "num_corners": p.num_corners,
            }
            # Save real-time robot state if available
            if p.R_gripper2base is not None and p.t_gripper2base is not None:
                row["R_gripper2base"] = p.R_gripper2base
                row["t_gripper2base"] = p.t_gripper2base
            if p.robot_euler_deg is not None:
                row["robot_euler_deg"] = p.robot_euler_deg
            if p.euler_order_used is not None:
                row["euler_order_used"] = p.euler_order_used
            f.write(json.dumps(row, ensure_ascii=False, cls=NumpyEncoder) + "\n")


def load_sample_pairs_jsonl(path: str) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            
            R_gripper2base = None
            t_gripper2base = None
            if "R_gripper2base" in row and "t_gripper2base" in row:
                R_gripper2base = np.asarray(row["R_gripper2base"], dtype=np.float64).reshape(3, 3)
                t_gripper2base = np.asarray(row["t_gripper2base"], dtype=np.float64).reshape(3)
            robot_euler_deg = None
            if "robot_euler_deg" in row:
                robot_euler_deg = np.asarray(row["robot_euler_deg"], dtype=np.float64).reshape(3)
            
            pairs.append(
                SamplePair(
                    sample_index=int(row["sample_index"]),
                    robot_row_index=int(row.get("robot_row_index", row["sample_index"])),
                    image_path=str(row.get("image_path", "")),
                    R_target2cam=np.asarray(row["R_target2cam"], dtype=np.float64).reshape(3, 3),
                    t_target2cam=np.asarray(row["t_target2cam"], dtype=np.float64).reshape(3, 1),
                    num_corners=int(row.get("num_corners", 0)),
                    R_gripper2base=R_gripper2base,
                    t_gripper2base=t_gripper2base,
                    robot_euler_deg=robot_euler_deg,
                    euler_order_used=str(row["euler_order_used"]) if "euler_order_used" in row else None,
                )
            )
    return pairs


def save_result_json(out_path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def write_t_matrix_npy(path: str, T: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.save(path, np.asarray(T, dtype=np.float64))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
