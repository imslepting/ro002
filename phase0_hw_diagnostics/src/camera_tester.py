"""單台相機連接測試與基本指標採集"""

from __future__ import annotations

import time
import cv2
import numpy as np

from shared.types import CameraTestResult
from shared.camera_manager import open_camera, release_camera


def test_camera(
    cam_index: int,
    n_frames: int = 30,
    sharpness_threshold: float = 50.0,
    brightness_range: tuple[float, float] = (30.0, 230.0),
    resolution: tuple[int, int] | None = None,
) -> CameraTestResult:
    """對指定相機讀取 n_frames 幀，採集各項指標

    Args:
        resolution: 指定測試分辨率 (w, h)，None 使用相機默認
    """
    cap = open_camera(cam_index, resolution=resolution)

    if not cap.isOpened():
        return CameraTestResult(
            cam_index=cam_index,
            resolution=(0, 0),
            fps_reported=0.0,
            fps_measured=0.0,
            frame_success_rate=0.0,
            mean_brightness=0.0,
            sharpness_score=0.0,
            is_color=False,
            status="ERROR",
            warnings=["無法打開相機"],
        )

    fps_reported = cap.get(cv2.CAP_PROP_FPS)

    success_count = 0
    actual_w, actual_h = 0, 0
    brightness_vals: list[float] = []
    sharpness_vals: list[float] = []
    is_color = True

    t_start = time.perf_counter()
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        success_count += 1

        # 從實際幀取分辨率（比 cap.get 更可靠）
        if actual_w == 0:
            actual_h, actual_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        brightness_vals.append(float(np.mean(gray)))
        sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        if frame.ndim == 2 or frame.shape[2] == 1:
            is_color = False
    t_elapsed = time.perf_counter() - t_start
    release_camera(cap)

    w, h = actual_w, actual_h

    if success_count == 0:
        return CameraTestResult(
            cam_index=cam_index,
            resolution=(w, h),
            fps_reported=fps_reported,
            fps_measured=0.0,
            frame_success_rate=0.0,
            mean_brightness=0.0,
            sharpness_score=0.0,
            is_color=False,
            status="ERROR",
            warnings=["所有幀讀取失敗"],
        )

    fps_measured = success_count / t_elapsed if t_elapsed > 0 else 0.0
    mean_brightness = float(np.mean(brightness_vals))
    mean_sharpness = float(np.mean(sharpness_vals))
    frame_success_rate = success_count / n_frames

    # 判定狀態與警告
    warnings: list[str] = []
    status = "OK"

    if frame_success_rate < 1.0:
        warnings.append(f"丟幀率 {(1 - frame_success_rate) * 100:.1f}%")
        status = "WARNING"

    if mean_sharpness < sharpness_threshold:
        warnings.append(f"清晰度偏低 ({mean_sharpness:.1f} < {sharpness_threshold})")
        status = "WARNING"

    lo, hi = brightness_range
    if mean_brightness < lo:
        warnings.append(f"畫面過暗 (亮度 {mean_brightness:.0f})")
        status = "WARNING"
    elif mean_brightness > hi:
        warnings.append(f"畫面過亮 (亮度 {mean_brightness:.0f})")
        status = "WARNING"

    return CameraTestResult(
        cam_index=cam_index,
        resolution=(w, h),
        fps_reported=fps_reported,
        fps_measured=fps_measured,
        frame_success_rate=frame_success_rate,
        mean_brightness=mean_brightness,
        sharpness_score=mean_sharpness,
        is_color=is_color,
        status=status,
        warnings=warnings,
    )
