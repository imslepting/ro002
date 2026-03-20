"""點雲裁剪與座標變換工具

從 depth map + mask 提取 3D 點雲，並提供座標變換與工作空間過濾。
"""

from __future__ import annotations

import numpy as np


def mask_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    color_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """mask 區域的像素 → 3D 點雲（相機座標系）

    Parameters
    ----------
    depth : float32 (H, W) metric depth（公尺）
    K : 3x3 相機內參矩陣
    mask : bool (H, W)
    color_image : BGR uint8 (H, W, 3)，可選

    Returns
    -------
    points : (M, 3) float32，相機座標系下的 XYZ
    colors : (M, 3) float32 [0, 1] RGB，或 None
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 取 mask 內且深度有效的像素
    valid = mask & (depth > 0)
    vs, us = np.where(valid)
    zs = depth[valid].astype(np.float32)

    if len(zs) == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    xs = ((us - cx) * zs / fx).astype(np.float32)
    ys = ((vs - cy) * zs / fy).astype(np.float32)

    points = np.stack([xs, ys, zs], axis=-1)  # (M, 3)

    colors = None
    if color_image is not None:
        bgr = color_image[vs, us]  # (M, 3) uint8 BGR
        colors = bgr[:, ::-1].astype(np.float32) / 255.0  # → RGB [0,1]

    return points, colors


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N, 3) 點雲 × 4×4 齊次變換 → (N, 3)

    Parameters
    ----------
    points : (N, 3) float
    T : 4×4 齊次變換矩陣

    Returns
    -------
    (N, 3) 變換後的點雲
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def filter_workspace(
    points: np.ndarray,
    colors: np.ndarray | None,
    limits: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """按 workspace_limits 裁剪點雲

    Parameters
    ----------
    points : (N, 3)
    colors : (N, 3) or None
    limits : dict with keys 'x', 'y', 'z', each a [min, max] list

    Returns
    -------
    filtered points, filtered colors
    """
    x_min, x_max = limits["x"]
    y_min, y_max = limits["y"]
    z_min, z_max = limits["z"]

    keep = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    filtered_pts = points[keep]
    filtered_colors = colors[keep] if colors is not None else None

    return filtered_pts, filtered_colors
