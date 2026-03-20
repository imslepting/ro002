"""視差 → 度量深度 + 點雲轉換"""

from __future__ import annotations

import numpy as np
import cv2

from shared.types import PointCloudResult


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> np.ndarray:
    """視差圖轉度量深度圖（公尺）

    depth = focal * baseline / disparity
    disp <= 0 或超出 [min_depth, max_depth] 的像素設為 0（無效）

    Args:
        disparity: float32 (H,W)，像素單位
        focal_length: 校正後焦距（像素）
        baseline: 基線長度（公尺）
        min_depth: 最小有效深度（公尺）
        max_depth: 最大有效深度（公尺）

    Returns:
        depth: float32 (H,W)，公尺，無效=0
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length * baseline) / disparity[valid]

    # 過濾超範圍
    out_of_range = (depth < min_depth) | (depth > max_depth)
    depth[out_of_range] = 0.0

    return depth


def depth_to_pointcloud(
    depth: np.ndarray,
    Q: np.ndarray,
    color_image: np.ndarray | None = None,
    max_depth: float = 10.0,
) -> PointCloudResult:
    """深度圖 → 3D 點雲

    使用 cv2.reprojectImageTo3D + Q 矩陣投影，過濾無效點。

    Args:
        depth: float32 (H,W) 度量深度（公尺）
        Q: 4x4 視差到深度投影矩陣
        color_image: BGR uint8 (H,W,3)，可選
        max_depth: 過濾閾值

    Returns:
        PointCloudResult
    """
    H, W = depth.shape[:2]

    # 從深度反推視差: disp = focal * baseline / depth
    # 但更直接的方式是用 Q 矩陣的逆向
    # Q[2,3] = focal, Q[3,2] = -1/baseline
    focal = Q[2, 3]
    inv_baseline = abs(Q[3, 2])
    if inv_baseline < 1e-8:
        return PointCloudResult(points=np.zeros((0, 3), dtype=np.float32))

    baseline = 1.0 / inv_baseline

    # 從深度計算視差
    disparity_from_depth = np.zeros_like(depth)
    valid_mask = depth > 0
    disparity_from_depth[valid_mask] = (focal * baseline) / depth[valid_mask]

    # reprojectImageTo3D
    points_3d = cv2.reprojectImageTo3D(disparity_from_depth, Q, handleMissingValues=True)

    # 過濾無效點
    mask = valid_mask & (depth <= max_depth) & (points_3d[:, :, 2] < 10000)
    points = points_3d[mask].reshape(-1, 3).astype(np.float32)

    colors = None
    if color_image is not None:
        # BGR → RGB
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        colors = rgb[mask].reshape(-1, 3)

    return PointCloudResult(points=points, colors=colors)


def depth_to_pointcloud_fast(
    depth: np.ndarray,
    Q: np.ndarray,
    color_image: np.ndarray | None = None,
    max_depth: float = 10.0,
    subsample: int = 2,
) -> tuple[np.ndarray, np.ndarray | None]:
    """快速深度 → 點雲（用於即時 3D 視覺化）

    跳過 cv2.reprojectImageTo3D，直接用 numpy 計算 XYZ。
    空間降採樣減少點數，返回 float64 供 Open3D 直接使用。

    Args:
        depth: float32 (H,W)
        Q: 4x4 投影矩陣（取 cx, cy, focal）
        color_image: BGR uint8 (H,W,3)，可選
        max_depth: 過濾閾值
        subsample: 空間降採樣倍率（2=每隔一個像素）

    Returns:
        (points_f64, colors_f64) — points (N,3), colors (N,3) 0-1 or None
    """
    cx = -Q[0, 3]
    cy = -Q[1, 3]
    focal = Q[2, 3]

    d = depth[::subsample, ::subsample]
    valid = (d > 0) & (d <= max_depth)
    vs, us = np.where(valid)
    zs = d[valid]

    xs = (us * subsample - cx) * zs / focal
    ys = (vs * subsample - cy) * zs / focal

    points = np.stack([xs, ys, zs], axis=-1).astype(np.float64)

    colors = None
    if color_image is not None:
        c = color_image[::subsample, ::subsample]
        bgr = c[valid]
        colors = bgr[:, ::-1].astype(np.float64) / 255.0  # BGR → RGB, 0-1

    return points, colors


def save_pointcloud_ply(pc: PointCloudResult, path: str) -> None:
    """輸出 .ply（ASCII 格式）"""
    n = len(pc.points)
    has_color = pc.colors is not None and len(pc.colors) == n

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = pc.points[i]
            if has_color:
                r, g, b = pc.colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"[DepthConverter] Saved point cloud ({n} pts) → {path}")
