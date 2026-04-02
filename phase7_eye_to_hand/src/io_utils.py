from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
