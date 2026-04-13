from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

from shared.camera_manager import CameraReader

from .charuco_pose_estimator import CharucoPoseEstimator
from .io_utils import CaptureRecord, SamplePair, ensure_dir, save_capture_records_jsonl, load_capture_records_jsonl
from .robot_state_fetcher import fetch_robot_state

logger = logging.getLogger(__name__)


def capture_samples_only(
    camera_index: int,
    out_dir: str,
    warmup_sec: float = 1.0,
    server_url: str = "http://140.118.117.61:5000/get_status",
) -> list[CaptureRecord]:
    """Interactive capture: only images + robot state (no Charuco analysis).
    
    This allows capture to be separated from analysis.
    You can then modify Charuco origin definition or analysis parameters
    without needing to recapture.

    Keys:
      c: capture current frame with live robot state
      q: quit
    """
    ensure_dir(out_dir)

    reader = CameraReader(camera_index)
    reader.start()
    time.sleep(max(0.0, warmup_sec))

    records: list[CaptureRecord] = []
    sample_count = 0

    try:
        while True:
            frame = reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            show = frame.copy()
            status = f"captures={len(records)}"
            cv2.putText(show, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(show, "c=capture q=quit", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Phase7 Capture (Image + Robot State Only)", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                # Fetch real-time robot state
                robot_state = fetch_robot_state(server_url=server_url)
                if robot_state is None:
                    logger.warning(f"Failed to fetch robot state, skipping capture")
                    continue

                img_path = os.path.join(out_dir, f"raw_{sample_count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                records.append(
                    CaptureRecord(
                        sample_index=len(records),
                        image_path=img_path,
                        R_gripper2base=robot_state.R_gripper2base,
                        t_gripper2base=robot_state.t_gripper2base,
                    )
                )
                logger.info(f"Captured raw image {len(records)-1}: {img_path}")
                sample_count += 1
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        
        # Summary
        if len(records) > 0:
            poses = np.array([r.t_gripper2base for r in records])
            pose_std = np.std(poses, axis=0)
            pose_range = np.max(poses, axis=0) - np.min(poses, axis=0)
            
            logger.info(f"\n[Capture Summary]")
            logger.info(f"  Total captures: {len(records)}")
            logger.info(f"  Pose X range: {pose_range[0]:.4f} m, std: {pose_std[0]:.4f} m")
            logger.info(f"  Pose Y range: {pose_range[1]:.4f} m, std: {pose_std[1]:.4f} m")
            logger.info(f"  Pose Z range: {pose_range[2]:.4f} m, std: {pose_std[2]:.4f} m")
            
            # Warn if diversity is low
            min_range = 0.1  # minimum 10cm movement expected
            if np.any(pose_range < min_range):
                bad_axes = []
                if pose_range[0] < min_range: bad_axes.append("X")
                if pose_range[1] < min_range: bad_axes.append("Y")
                if pose_range[2] < min_range: bad_axes.append("Z")
                logger.warning(f"⚠️  Low pose diversity on axes: {', '.join(bad_axes)}")
                logger.warning(f"    Please collect captures with gripper at diverse positions.")

    return records


def analyze_samples(
    estimator: CharucoPoseEstimator,
    captures_path: str,
) -> list[SamplePair]:
    """Analyze captured images: estimate Charuco pose from raw capture records.
    
    This separates analysis from capture, allowing you to modify:
    - Charuco origin definition
    - Detection parameters
    - Solver settings
    
    without recapturing images.
    """
    logger.info(f"Loading capture records from {captures_path}...")
    records = load_capture_records_jsonl(captures_path)
    logger.info(f"Loaded {len(records)} records")

    pairs: list[SamplePair] = []
    failed_count = 0

    for i, record in enumerate(records):
        if not os.path.exists(record.image_path):
            logger.warning(f"Sample {i}: image not found at {record.image_path}")
            failed_count += 1
            continue

        frame = cv2.imread(record.image_path)
        if frame is None:
            logger.warning(f"Sample {i}: failed to read image {record.image_path}")
            failed_count += 1
            continue

        # Estimate Charuco pose
        est = estimator.estimate(frame)
        if not est.success or est.R_target2cam is None or est.t_target2cam is None:
            logger.warning(f"Sample {i}: Charuco detection failed ({est.num_corners} corners)")
            failed_count += 1
            continue

        # Create SamplePair
        pairs.append(
            SamplePair(
                sample_index=len(pairs),
                robot_row_index=i,
                image_path=record.image_path,
                R_target2cam=est.R_target2cam,
                t_target2cam=est.t_target2cam,
                num_corners=est.num_corners,
                R_gripper2base=record.R_gripper2base,
                t_gripper2base=record.t_gripper2base,
            )
        )
        logger.info(f"Analyzed sample {len(pairs)-1}: {est.num_corners} corners")

    logger.info(f"\n[Analysis Summary]")
    logger.info(f"  Successful: {len(pairs)} / {len(records)}")
    if failed_count > 0:
        logger.warning(f"  Failed: {failed_count}")

    return pairs
