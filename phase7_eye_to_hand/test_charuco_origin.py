#!/usr/bin/env python3
"""
Test different Charuco origin definitions.

用法：
1. 修改 settings.yaml 中的 charuco.origin 和 charuco.custom_origin
2. 執行此腳本來測試不同的origin定義

可用的origin mode:
- "first_corner": 第一內角 (預設)
- "center": 板幾何中心
- "custom": 自定義點 [x, y, z]
"""

from __future__ import annotations

import json
import sys
import os

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.intrinsics_io import load_calib_result
from phase7_eye_to_hand.src.charuco_pose_estimator import CharucoPoseEstimator
from phase7_eye_to_hand.src.io_utils import load_sample_pairs_jsonl


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_origin() -> None:
    """Test current origin configuration and show offset."""
    cfg = load_config()
    p7 = cfg.get("phase7_eye_to_hand", {})
    charuco_cfg = cfg["calibration"]["charuco"]
    
    camera_name = p7.get("camera_name", "cam0")
    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")
    
    calib = load_calib_result(camera_name, intrinsics_path)
    if calib is None:
        raise ValueError(f"intrinsics for {camera_name} not found")
    
    # Create estimator (this will apply the configured origin)
    estimator = CharucoPoseEstimator(
        charuco_cfg=charuco_cfg,
        K=calib.K,
        D=calib.D,
        min_corners=int(p7.get("min_charuco_corners", 6)),
    )
    
    print("=== Charuco Origin Configuration ===\n")
    print(f"Mode: {charuco_cfg.get('origin', 'first_corner')}")
    if charuco_cfg.get('origin') == 'custom':
        print(f"Custom origin: {charuco_cfg.get('custom_origin')}")
    
    print(f"\nOrigin offset applied: {estimator._origin_offset}")
    print(f"  (This is subtracted from solvePnP result)")
    print(f"  Magnitude: {np.linalg.norm(estimator._origin_offset):.6f} m = {np.linalg.norm(estimator._origin_offset)*1000:.2f} mm")


def test_origin_effect() -> None:
    """Show the effect of origin on analyzed samples."""
    cfg = load_config()
    
    sample_pairs_path = os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "samples", "sample_pairs.jsonl")
    if not os.path.exists(sample_pairs_path):
        print(f"Error: {sample_pairs_path} not found")
        print("Run --mode analyze first to generate sample pairs")
        return
    
    try:
        pairs = load_sample_pairs_jsonl(sample_pairs_path)
    except Exception as e:
        print(f"Error loading samples: {e}")
        return
    
    t_norms = []
    for pair in pairs:
        t = np.array(pair.t_target2cam).reshape(3)
        t_norms.append(np.linalg.norm(t))
    
    t_norms = np.array(t_norms) * 1000  # Convert to mm
    
    print("=== Current Samples Analysis ===\n")
    print(f"Number of samples: {len(pairs)}")
    print(f"\nt_target2cam distance from camera (mm):")
    print(f"  Mean: {np.mean(t_norms):.2f} mm")
    print(f"  Median: {np.median(t_norms):.2f} mm")
    print(f"  Std: {np.std(t_norms):.2f} mm")
    print(f"  Min: {np.min(t_norms):.2f} mm")
    print(f"  Max: {np.max(t_norms):.2f} mm")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Test Charuco origin configurations")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    
    print("=" * 60)
    
    try:
        test_origin()
        print()
        test_origin_effect()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("\nTo test a different origin:")
    print("  1. Edit config/settings.yaml")
    print("     - Change charuco.origin to 'first_corner', 'center', or 'custom'")
    print("     - If custom, set charuco.custom_origin to [x, y, z]")
    print("  2. Run analyze to generate new samples:")
    print("     conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode analyze")
    print("  3. Run solve to get calibration results:")
    print("     conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode solve")


if __name__ == "__main__":
    main()
