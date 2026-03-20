"""ChArUco 角點偵測 + cornerSubPix 精煉"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionResult:
    success: bool
    charuco_corners: np.ndarray | None  # (N,1,2)
    charuco_ids: np.ndarray | None      # (N,1)
    aruco_corners: list | None
    aruco_ids: np.ndarray | None
    num_corners: int


def detect_charuco(
    gray: np.ndarray,
    board,
    dictionary,
    refine_subpix: bool = True,
    min_corners: int = 6,
) -> DetectionResult:
    """偵測 ChArUco 角點

    Args:
        gray: 灰度圖
        board: CharucoBoard 對象
        dictionary: ArUco 字典
        refine_subpix: True 用於標定（精確），False 用於即時預覽（快速）
        min_corners: 最少需要的角點數
    """
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    aruco_corners, aruco_ids, _ = aruco_detector.detectMarkers(gray)

    if aruco_ids is None or len(aruco_ids) < 2:
        return DetectionResult(
            success=False,
            charuco_corners=None,
            charuco_ids=None,
            aruco_corners=aruco_corners if aruco_ids is not None else None,
            aruco_ids=aruco_ids,
            num_corners=0,
        )

    result = cv2.aruco.interpolateCornersCharuco(
        aruco_corners, aruco_ids, gray, board
    )
    # OpenCV 4.13+: 返回 (num, corners, ids)
    # 舊版: 返回 (corners, ids, ...) 其中 [0] 是 ndarray
    if isinstance(result[0], int):
        # 4.13+ 格式: (num, corners, ids)
        charuco_corners, charuco_ids = result[1], result[2]
    else:
        # 舊版格式: (corners, ids, ...)
        charuco_corners, charuco_ids = result[0], result[1]

    if charuco_corners is None or len(charuco_corners) < min_corners:
        return DetectionResult(
            success=False,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            aruco_corners=aruco_corners,
            aruco_ids=aruco_ids,
            num_corners=len(charuco_corners) if charuco_corners is not None else 0,
        )

    if refine_subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        charuco_corners = cv2.cornerSubPix(
            gray, charuco_corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
        )

    return DetectionResult(
        success=True,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        aruco_corners=aruco_corners,
        aruco_ids=aruco_ids,
        num_corners=len(charuco_corners),
    )


def draw_detection_overlay(frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """繪製偵測結果疊加層"""
    img = frame.copy()

    if detection.aruco_corners and detection.aruco_ids is not None:
        cv2.aruco.drawDetectedMarkers(img, detection.aruco_corners, detection.aruco_ids)

    if detection.charuco_corners is not None and detection.charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            img, detection.charuco_corners, detection.charuco_ids, cornerColor=(0, 255, 0)
        )

    return img
