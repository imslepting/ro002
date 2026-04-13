#!/usr/bin/env python3
"""Re-process existing sample images with corrected charuco_pose_estimator.

This script:
1. Reads existing sample_pairs.jsonl
2. Re-detects Charuco in each saved image using the NEW board-center-based origin
3. Saves the updated poses to sample_pairs.corrected.jsonl
4. Then you can run Phase7 solver with --mode=solve using this corrected file

Usage:
    conda run -n ro002 python phase7_eye_to_hand/reprocess_samples_with_board_center.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.intrinsics_io import load_calib_result
from phase7_eye_to_hand.src.charuco_pose_estimator import CharucoPoseEstimator
from phase7_eye_to_hand.src.io_utils import load_sample_pairs_jsonl, save_sample_pairs_jsonl, SamplePair

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-process existing samples with board-center-based Charuco origin"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "samples", "sample_pairs.jsonl"),
        help="Input sample pairs JSONL (assumes images exist)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "samples", "sample_pairs.board_center.jsonl"),
        help="Output corrected sample pairs JSONL",
    )
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    p7 = cfg.get("phase7_eye_to_hand", {})
    camera_name = p7.get("camera_name", "cam0")
    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")

    calib = load_calib_result(camera_name, intrinsics_path)
    if calib is None:
        raise ValueError(f"intrinsics for {camera_name} not found: {intrinsics_path}")

    estimator = CharucoPoseEstimator(
        charuco_cfg=cfg["calibration"]["charuco"],
        K=calib.K,
        D=calib.D,
        min_corners=int(p7.get("min_charuco_corners", 6)),
    )

    # Load old pairs (to get robot state and image paths)
    old_pairs = load_sample_pairs_jsonl(args.input)
    logger.info(f"Loaded {len(old_pairs)} old sample pairs from {args.input}")

    # Re-detect Charuco poses using corrected board-center origin
    new_pairs = []
    for i, old_pair in enumerate(old_pairs):
        if not os.path.exists(old_pair.image_path):
            logger.warning(f"Image not found: {old_pair.image_path}, skipping")
            continue

        frame = cv2.imread(old_pair.image_path)
        if frame is None:
            logger.warning(f"Failed to read image: {old_pair.image_path}, skipping")
            continue

        # Re-estimate Charuco using NEW board-center-based origin
        est = estimator.estimate(frame)

        if not est.success or est.R_target2cam is None or est.t_target2cam is None:
            logger.warning(f"Re-detection failed for sample {old_pair.sample_index}, skipping")
            continue

        # Create new pair with updated Charuco pose, but preserve robot state
        new_pair = SamplePair(
            sample_index=old_pair.sample_index,
            robot_row_index=old_pair.robot_row_index,
            image_path=old_pair.image_path,
            R_target2cam=est.R_target2cam,  # Updated with board-center origin
            t_target2cam=est.t_target2cam,  # Updated with board-center origin
            num_corners=est.num_corners,
            R_gripper2base=old_pair.R_gripper2base,  # Preserved
            t_gripper2base=old_pair.t_gripper2base,  # Preserved
        )
        new_pairs.append(new_pair)
        logger.info(
            f"Re-detected sample {i}: {old_pair.sample_index}, corners={est.num_corners}, "
            f"t_target2cam_norm={np.linalg.norm(est.t_target2cam)*1000:.1f} mm"
        )

    if not new_pairs:
        raise ValueError("No valid pairs after re-detection")

    save_sample_pairs_jsonl(new_pairs, args.output)
    logger.info(f"Saved {len(new_pairs)} corrected pairs to {args.output}")
    logger.info("\n✓ Re-processing complete")
    logger.info(f"Next steps:")
    logger.info(f"  1. Update settings.yaml if needed (target_offset_gripper_m is no longer valid)")
    logger.info(f"  2. Run Phase7 solver with corrected pairs:")
    logger.info(f"     conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode=solve \\")
    logger.info(f"       --sample-pairs {args.output}")
    logger.info(f"  3. Compare results with/without board-center correction")


if __name__ == "__main__":
    main()
