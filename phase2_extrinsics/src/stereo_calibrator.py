"""雙目外參標定 — stereoCalibrate 封裝 + 校正可視化"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from phase1_intrinsics.src.charuco_detector import detect_charuco


@dataclass
class StereoCalibResult:
    """雙目標定結果"""
    pair_name: str
    cam_left: str
    cam_right: str
    R: np.ndarray          # 3x3 旋轉矩陣
    T: np.ndarray          # 3x1 平移向量
    E: np.ndarray          # 3x3 本質矩陣
    F: np.ndarray          # 3x3 基礎矩陣
    rms: float
    T44: np.ndarray        # 4x4 齊次變換
    num_pairs_used: int


def detect_stereo_pair(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    board,
    dictionary,
    min_corners: int = 6,
):
    """偵測左右圖的共同 ChArUco 角點

    Returns:
        (obj_points, img_pts_left, img_pts_right) 或 None
    """
    det_left = detect_charuco(left_gray, board, dictionary,
                              refine_subpix=True, min_corners=min_corners)
    det_right = detect_charuco(right_gray, board, dictionary,
                               refine_subpix=True, min_corners=min_corners)

    if not det_left.success or not det_right.success:
        return None

    # 找共同角點 ID 交集
    ids_left = det_left.charuco_ids.flatten()
    ids_right = det_right.charuco_ids.flatten()
    common_ids = np.intersect1d(ids_left, ids_right)

    if len(common_ids) < min_corners:
        return None

    # 過濾左圖只保留共同 ID
    left_mask = np.isin(ids_left, common_ids)
    filtered_corners_left = det_left.charuco_corners[left_mask]
    filtered_ids_left = det_left.charuco_ids[left_mask]

    # 過濾右圖只保留共同 ID
    right_mask = np.isin(ids_right, common_ids)
    filtered_corners_right = det_right.charuco_corners[right_mask]
    filtered_ids_right = det_right.charuco_ids[right_mask]

    # 用 board.matchImagePoints 取得 obj_points 和排序後的 img_points
    obj_pts_left, img_pts_left = board.matchImagePoints(
        filtered_corners_left, filtered_ids_left
    )
    obj_pts_right, img_pts_right = board.matchImagePoints(
        filtered_corners_right, filtered_ids_right
    )

    if obj_pts_left is None or obj_pts_right is None:
        return None
    if len(obj_pts_left) < min_corners:
        return None

    # obj_points 應該一致（同一塊板），用左邊的即可
    return obj_pts_left, img_pts_left, img_pts_right


def calibrate_stereo(
    pair_dir: str,
    K_l: np.ndarray,
    D_l: np.ndarray,
    K_r: np.ndarray,
    D_r: np.ndarray,
    image_size: tuple[int, int],
    board,
    dictionary,
    pair_name: str,
    cam_left: str,
    cam_right: str,
    min_corners: int = 6,
) -> StereoCalibResult | None:
    """從配對圖片目錄執行雙目標定

    Args:
        pair_dir: 含 left/*.png + right/*.png 的目錄
        K_l, D_l: 左相機內參
        K_r, D_r: 右相機內參
        image_size: (width, height)
        board: CharucoBoard
        dictionary: ArUco 字典
        pair_name: 配對名稱 e.g. "cam1_cam2"
        cam_left, cam_right: 相機名稱
        min_corners: 最少共同角點數
    """
    left_dir = os.path.join(pair_dir, "left")
    right_dir = os.path.join(pair_dir, "right")

    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        print(f"[StereoCalib] 目錄不存在: {left_dir} 或 {right_dir}")
        return None

    left_files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
    right_files = sorted([f for f in os.listdir(right_dir) if f.endswith(".png")])

    n_pairs = min(len(left_files), len(right_files))
    if n_pairs == 0:
        print("[StereoCalib] 無圖片對")
        return None

    all_obj = []
    all_img_l = []
    all_img_r = []

    for i in range(n_pairs):
        left_path = os.path.join(left_dir, left_files[i])
        right_path = os.path.join(right_dir, right_files[i])

        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue

        result = detect_stereo_pair(left_img, right_img, board, dictionary, min_corners)
        if result is None:
            continue

        obj_pts, img_l, img_r = result
        all_obj.append(obj_pts)
        all_img_l.append(img_l)
        all_img_r.append(img_r)

    if len(all_obj) < 3:
        print(f"[StereoCalib] 有效配對不足（{len(all_obj)}），需要至少 3 對")
        return None

    print(f"[StereoCalib] {pair_name}: 使用 {len(all_obj)}/{n_pairs} 對進行標定...")

    rms, K_l_out, D_l_out, K_r_out, D_r_out, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=all_obj,
        imagePoints1=all_img_l,
        imagePoints2=all_img_r,
        cameraMatrix1=K_l.copy(),
        distCoeffs1=D_l.copy(),
        cameraMatrix2=K_r.copy(),
        distCoeffs2=D_r.copy(),
        imageSize=image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    print(f"[StereoCalib] {pair_name}: RMS = {rms:.4f} px")

    from phase2_extrinsics.src.extrinsics_io import Rt_to_T44
    T44 = Rt_to_T44(R, T)

    return StereoCalibResult(
        pair_name=pair_name,
        cam_left=cam_left,
        cam_right=cam_right,
        R=R, T=T, E=E, F=F,
        rms=rms,
        T44=T44,
        num_pairs_used=len(all_obj),
    )


def compute_stereo_rectification(
    K_l: np.ndarray,
    D_l: np.ndarray,
    K_r: np.ndarray,
    D_r: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: tuple[int, int],
):
    """計算校正映射

    Returns:
        (map1_l, map2_l, map1_r, map2_r, Q)
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        alpha=0,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_16SC2
    )

    return map1_l, map2_l, map1_r, map2_r, Q


def compute_epipolar_error(
    pair_dir: str,
    board,
    dictionary,
    K_l: np.ndarray,
    D_l: np.ndarray,
    K_r: np.ndarray,
    D_r: np.ndarray,
    F: np.ndarray,
    min_corners: int = 6,
) -> list[tuple[str, str, float]]:
    """計算每對圖的極線誤差

    Returns:
        [(left_path, right_path, mean_error), ...]
    """
    left_dir = os.path.join(pair_dir, "left")
    right_dir = os.path.join(pair_dir, "right")

    left_files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
    right_files = sorted([f for f in os.listdir(right_dir) if f.endswith(".png")])

    n_pairs = min(len(left_files), len(right_files))
    results = []

    for i in range(n_pairs):
        left_path = os.path.join(left_dir, left_files[i])
        right_path = os.path.join(right_dir, right_files[i])

        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue

        result = detect_stereo_pair(left_img, right_img, board, dictionary, min_corners)
        if result is None:
            continue

        _, img_l, img_r = result

        # 計算極線距離: x'^T F x 對每個對應點
        pts_l = img_l.reshape(-1, 2).astype(np.float64)
        pts_r = img_r.reshape(-1, 2).astype(np.float64)

        # 極線 l = F x_l, 點到極線距離 = |l^T x_r| / sqrt(l[0]^2 + l[1]^2)
        pts_l_h = np.hstack([pts_l, np.ones((len(pts_l), 1))])  # (N, 3)
        lines_r = (F @ pts_l_h.T).T  # (N, 3) 右圖極線

        pts_r_h = np.hstack([pts_r, np.ones((len(pts_r), 1))])
        numerator = np.abs(np.sum(lines_r * pts_r_h, axis=1))
        denominator = np.sqrt(lines_r[:, 0] ** 2 + lines_r[:, 1] ** 2)
        distances = numerator / (denominator + 1e-8)

        mean_err = float(np.mean(distances))
        results.append((left_path, right_path, mean_err))

    return results
