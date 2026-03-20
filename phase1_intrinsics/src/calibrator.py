"""相機內參標定 — calibrateCameraCharuco 封裝"""

from __future__ import annotations

import cv2
import numpy as np

from shared.types import CalibResult
from phase1_intrinsics.src.charuco_detector import detect_charuco


def calibrate_camera(
    image_paths: list[str],
    board,
    dictionary,
    cam_name: str,
    min_corners: int = 6,
) -> CalibResult | None:
    """從採集的圖片執行內參標定

    Args:
        image_paths: 已採集的圖片路徑列表
        board: CharucoBoard 對象
        dictionary: ArUco 字典
        cam_name: 相機名稱（如 "cam0"）
        min_corners: 每幀最少角點數

    Returns:
        CalibResult 或 None（如果有效幀不足）
    """
    all_corners = []
    all_ids = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        det = detect_charuco(gray, board, dictionary, refine_subpix=True, min_corners=min_corners)
        if det.success:
            all_corners.append(det.charuco_corners)
            all_ids.append(det.charuco_ids)

    if len(all_corners) < 3:
        print(f"[Calibrator] {cam_name}: 有效幀不足（{len(all_corners)}），需要至少 3 幀")
        return None

    print(f"[Calibrator] {cam_name}: 使用 {len(all_corners)}/{len(image_paths)} 幀進行標定...")

    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    print(f"[Calibrator] {cam_name}: RMS = {rms:.4f} px")

    return CalibResult(
        cam_name=cam_name,
        K=K,
        D=D,
        image_size=image_size,
        rms=rms,
    )


def compute_per_frame_errors(
    image_paths: list[str],
    board,
    dictionary,
    K: np.ndarray,
    D: np.ndarray,
    min_corners: int = 6,
) -> list[tuple[str, float]]:
    """計算每幀的重投影誤差

    Returns:
        [(image_path, rms_error), ...] 僅包含有效幀
    """
    results = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        det = detect_charuco(gray, board, dictionary, refine_subpix=True, min_corners=min_corners)
        if not det.success:
            continue

        # 用已知內參求解 PnP
        obj_points, img_points = board.matchImagePoints(
            det.charuco_corners, det.charuco_ids
        )
        if obj_points is None or len(obj_points) < min_corners:
            continue

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        if not success:
            continue

        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        errors = np.sqrt(np.sum((img_points - projected.reshape(-1, 1, 2)) ** 2, axis=2))
        rms = float(np.sqrt(np.mean(errors ** 2)))
        results.append((path, rms))

    return results
