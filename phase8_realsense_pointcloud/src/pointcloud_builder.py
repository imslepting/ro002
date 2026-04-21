"""Utilities for converting aligned RGB-D to point cloud arrays."""

from __future__ import annotations

import numpy as np


def colorize_depth(depth_m: np.ndarray, max_depth: float = 3.0) -> np.ndarray:
    import cv2

    valid = depth_m > 0
    norm = np.clip(depth_m / max_depth, 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def build_pointcloud(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float,
    max_depth: float,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build XYZ and RGB arrays from aligned depth/color.

    Returns:
        points: (N, 3) float32 in meters, camera frame
        colors: (N, 3) float32 in [0, 1], RGB order
    """
    h, w = depth_m.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    z = depth_m[0:h:stride, 0:w:stride]
    valid = (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1)[valid].astype(np.float32)
    # Convert BGR to RGB and normalize to [0, 1]
    rgb = color_bgr[0:h:stride, 0:w:stride, ::-1][valid].astype(np.float32) / 255.0
    return xyz, rgb
