from __future__ import annotations

import logging
import os
import time

import cv2

from shared.camera_manager import CameraReader

from .charuco_pose_estimator import CharucoPoseEstimator
from .io_utils import SamplePair, ensure_dir
from .robot_state_fetcher import fetch_robot_state

logger = logging.getLogger(__name__)


def capture_sample_pairs(
    camera_index: int,
    estimator: CharucoPoseEstimator,
    out_dir: str,
    warmup_sec: float = 1.0,
    server_url: str = "http://140.118.117.61:5000/get_status",
) -> list[SamplePair]:
    """Interactive capture: capture charuco pose and real-time robot state.

    Keys:
      c: capture current frame with live robot state
      q: quit
    """
    ensure_dir(out_dir)

    reader = CameraReader(camera_index)
    reader.start()
    time.sleep(max(0.0, warmup_sec))

    pairs: list[SamplePair] = []
    sample_count = 0

    try:
        while True:
            frame = reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            est = estimator.estimate(frame)
            show = est.debug_frame if est.debug_frame is not None else frame.copy()
            status = f"samples={len(pairs)} corners={est.num_corners}"
            cv2.putText(show, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(show, "c=capture q=quit", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Phase7 Eye-to-Hand Capture", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                if not est.success or est.R_target2cam is None or est.t_target2cam is None:
                    logger.warning(f"Detection failed, not captured")
                    continue

                # Fetch real-time robot state
                robot_state = fetch_robot_state(server_url=server_url)
                if robot_state is None:
                    logger.warning(f"Failed to fetch robot state, skipping capture")
                    continue

                img_path = os.path.join(out_dir, f"sample_{sample_count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                pairs.append(
                    SamplePair(
                        sample_index=len(pairs),
                        robot_row_index=len(pairs),  # Use sample_index as row_index for compatibility
                        image_path=img_path,
                        R_target2cam=est.R_target2cam,
                        t_target2cam=est.t_target2cam,
                        num_corners=est.num_corners,
                        R_gripper2base=robot_state.R_gripper2base,
                        t_gripper2base=robot_state.t_gripper2base,
                    )
                )
                logger.info(f"Captured sample {len(pairs)-1}: t_gripper2base={robot_state.t_gripper2base}, gripper={robot_state.gripper_state}")
                sample_count += 1
    finally:
        reader.stop()
        cv2.destroyAllWindows()

    return pairs
