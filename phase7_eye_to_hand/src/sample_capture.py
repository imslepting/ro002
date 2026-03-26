from __future__ import annotations

import os
import time

import cv2

from shared.camera_manager import CameraReader

from .charuco_pose_estimator import CharucoPoseEstimator
from .io_utils import SamplePair, ensure_dir


def capture_sample_pairs(
    camera_index: int,
    robot_row_count: int,
    estimator: CharucoPoseEstimator,
    out_dir: str,
    warmup_sec: float = 1.0,
) -> list[SamplePair]:
    """Interactive capture: one capture per robot CSV row in order.

    Keys:
      c: capture current frame and bind to current row index
      s: skip current row index
      q: quit
    """
    ensure_dir(out_dir)

    reader = CameraReader(camera_index)
    reader.start()
    time.sleep(max(0.0, warmup_sec))

    pairs: list[SamplePair] = []
    row_idx = 0

    try:
        while row_idx < robot_row_count:
            frame = reader.frame
            if frame is None:
                cv2.waitKey(1)
                continue

            est = estimator.estimate(frame)
            show = est.debug_frame if est.debug_frame is not None else frame.copy()
            status = f"row={row_idx + 1}/{robot_row_count} corners={est.num_corners}"
            cv2.putText(show, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(show, "c=capture s=skip q=quit", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Phase7 Eye-to-Hand Capture", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                row_idx += 1
                continue
            if key == ord("c"):
                if not est.success or est.R_target2cam is None or est.t_target2cam is None:
                    print(f"[Phase7] row={row_idx}: detection failed, not captured")
                    continue

                img_path = os.path.join(out_dir, f"sample_{row_idx:04d}.jpg")
                cv2.imwrite(img_path, frame)
                pairs.append(
                    SamplePair(
                        sample_index=len(pairs),
                        robot_row_index=row_idx,
                        image_path=img_path,
                        R_target2cam=est.R_target2cam,
                        t_target2cam=est.t_target2cam,
                        num_corners=est.num_corners,
                    )
                )
                print(f"[Phase7] captured row={row_idx} -> {img_path}")
                row_idx += 1
    finally:
        reader.stop()
        cv2.destroyAllWindows()

    return pairs
