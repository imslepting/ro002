"""Phase 7 - Eye-to-Hand calibration.

Usage:
    conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.intrinsics_io import load_calib_result
from phase7_eye_to_hand.src.charuco_pose_estimator import CharucoPoseEstimator
from phase7_eye_to_hand.src.handeye_solver import solve_eye_to_hand
from phase7_eye_to_hand.src.io_utils import (
    load_sample_pairs_jsonl,
    now_iso,
    save_result_json,
    save_sample_pairs_jsonl,
    write_t_matrix_npy,
)
from phase7_eye_to_hand.src.robot_pose_parser import load_robot_pose_csv
from phase7_eye_to_hand.src.sample_capture import capture_sample_pairs
from phase7_eye_to_hand.src.validation import validate_translation_error


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_robot_samples_by_pairs(robot_samples, pairs):
    out = []
    for p in pairs:
        out.append(robot_samples[p.robot_row_index])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase7 Eye-to-Hand calibration")
    parser.add_argument("--mode", choices=["capture", "solve", "all"], default="all")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    p7 = cfg.get("phase7_eye_to_hand", {})

    camera_name = p7.get("camera_name", "cam0")
    camera_idx = cfg["cameras"][camera_name]["index"]
    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")

    csv_path = p7.get("robot_pose_csv", "")
    if not csv_path:
        raise ValueError("phase7_eye_to_hand.robot_pose_csv is required")
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_ROOT, csv_path)

    out_base = os.path.join(_ROOT, "phase7_eye_to_hand", "outputs")
    sample_jsonl = os.path.join(out_base, "samples", "sample_pairs.jsonl")
    sample_img_dir = os.path.join(out_base, "samples", "images")
    result_json = os.path.join(out_base, "eye_to_hand_result.json")
    result_npy = os.path.join(out_base, "T_cam2arm.npy")
    report_json = os.path.join(out_base, "reports", "validation_report.json")

    robot_samples = load_robot_pose_csv(
        csv_path=csv_path,
        delimiter=p7.get("csv_delimiter", ","),
        has_header=bool(p7.get("csv_has_header", False)),
        pose_columns=p7.get("pose_columns", {}),
        unit_scale=float(p7.get("position_unit_scale", 0.001)),
        euler_order=p7.get("euler_order", "xyz"),
    )

    calib = load_calib_result(camera_name, intrinsics_path)
    if calib is None:
        raise ValueError(f"intrinsics for {camera_name} not found: {intrinsics_path}")

    estimator = CharucoPoseEstimator(
        charuco_cfg=cfg["calibration"]["charuco"],
        K=calib.K,
        D=calib.D,
        min_corners=int(p7.get("min_charuco_corners", 6)),
    )

    if args.mode in ("capture", "all"):
        pairs = capture_sample_pairs(
            camera_index=camera_idx,
            robot_row_count=len(robot_samples),
            estimator=estimator,
            out_dir=sample_img_dir,
            warmup_sec=float(p7.get("camera_warmup_sec", 1.0)),
        )
        save_sample_pairs_jsonl(pairs, sample_jsonl)
        print(f"[Phase7] captured pairs: {len(pairs)} -> {sample_jsonl}")

        if args.mode == "capture":
            return

    pairs = load_sample_pairs_jsonl(sample_jsonl)
    min_samples = int(p7.get("min_samples", 15))
    if len(pairs) < min_samples:
        raise ValueError(f"need at least {min_samples} pairs, got {len(pairs)}")

    paired_robot_samples = _extract_robot_samples_by_pairs(robot_samples, pairs)
    res = solve_eye_to_hand(
        robot_samples=paired_robot_samples,
        sample_pairs=pairs,
        method=p7.get("hand_eye_method", "tsai"),
    )

    target_offset = np.asarray(p7.get("target_offset_gripper_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    val_stats, errors_mm = validate_translation_error(
        T_cam2base=res.T_cam2base,
        robot_samples=paired_robot_samples,
        pairs=pairs,
        target_offset_gripper_m=target_offset,
    )

    payload = {
        "timestamp": now_iso(),
        "camera_name": camera_name,
        "camera_index": camera_idx,
        "method": res.method,
        "num_pairs": len(pairs),
        "euler_order": p7.get("euler_order", "xyz"),
        "position_unit_scale": float(p7.get("position_unit_scale", 0.001)),
        "T_cam2arm": res.T_cam2base,
        "validation": {
            "mean_mm": val_stats.mean_mm,
            "median_mm": val_stats.median_mm,
            "p95_mm": val_stats.p95_mm,
            "max_mm": val_stats.max_mm,
            "count": val_stats.count,
        },
    }

    save_result_json(result_json, payload)
    save_result_json(report_json, {"errors_mm": errors_mm, "summary": payload["validation"]})
    write_t_matrix_npy(result_npy, res.T_cam2base)

    print("[Phase7] done")
    print(f"[Phase7] T_cam2arm: {result_json}")
    print(f"[Phase7] matrix npy: {result_npy}")
    print(f"[Phase7] mean error: {val_stats.mean_mm:.2f} mm")


if __name__ == "__main__":
    main()
