"""深度/視差可視化與 I/O 工具"""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np


def colorize_disparity(disparity: np.ndarray, max_disp: float = 256.0) -> np.ndarray:
    """視差圖 → TURBO 偽彩色 BGR uint8"""
    disp_norm = np.clip(disparity / max_disp, 0, 1)
    disp_u8 = (disp_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
    # 無效區域（disp<=0）設為黑色
    colored[disparity <= 0] = 0
    return colored


def colorize_depth(depth: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """深度圖 → TURBO 偽彩色 BGR uint8，無效=黑色"""
    valid = depth > 0
    depth_norm = np.clip(depth / max_depth, 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def draw_rectification_check(
    left_rect: np.ndarray, right_rect: np.ndarray, n_lines: int = 16,
) -> np.ndarray:
    """左右拼接 + 水平對極線，用於視覺 QA"""
    h, w = left_rect.shape[:2]
    canvas = np.hstack([left_rect, right_rect])
    step = h // n_lines
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w * 2, y), (0, 255, 0), 1)
    return canvas


def save_depth(
    depth: np.ndarray, output_dir: str, timestamp: str | None = None,
) -> str:
    """存 .npy 深度圖，返回路徑"""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"depth_{timestamp}.npy")
    np.save(path, depth)
    print(f"[DepthUtils] Saved depth → {path}")
    return path


def save_disparity_vis(
    disparity_color: np.ndarray, output_dir: str, timestamp: str | None = None,
) -> str:
    """存 .png 視差可視化，返回路徑"""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"disparity_{timestamp}.png")
    cv2.imwrite(path, disparity_color)
    print(f"[DepthUtils] Saved disparity vis → {path}")
    return path


def pixel_to_3d(u: int, v: int, depth: np.ndarray, Q: np.ndarray):
    """像素座標 + 深度 → 3D 點 (x, y, z) 公尺，無效則返回 None"""
    if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
        return None
    z = float(depth[v, u])
    if z <= 0:
        return None
    cx, cy, focal = -Q[0, 3], -Q[1, 3], Q[2, 3]
    x = (u - cx) * z / focal
    y = (v - cy) * z / focal
    return (x, y, z)


def measure_distance_3d(
    pt_a: tuple[int, int],
    pt_b: tuple[int, int],
    depth: np.ndarray,
    Q: np.ndarray,
) -> float | None:
    """兩像素點之間的 3D 歐氏距離（公尺），任一點無效則返回 None"""
    p1 = pixel_to_3d(pt_a[0], pt_a[1], depth, Q)
    p2 = pixel_to_3d(pt_b[0], pt_b[1], depth, Q)
    if p1 is None or p2 is None:
        return None
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2))


def draw_measurement_overlay(
    img: np.ndarray,
    pt_a: tuple[int, int] | None,
    pt_b: tuple[int, int] | None,
    distance: float | None = None,
    depth: np.ndarray | None = None,
    Q: np.ndarray | None = None,
) -> np.ndarray:
    """在影像上畫測量標記：圓點 + 連線 + 距離文字"""
    out = img.copy()
    color_a = (0, 255, 255)   # 青色
    color_b = (255, 0, 255)   # 品紅
    color_line = (255, 255, 0)  # 黃色

    if pt_a is not None:
        cv2.circle(out, pt_a, 6, color_a, -1)
        cv2.circle(out, pt_a, 6, (0, 0, 0), 1)
        # 顯示該點深度
        if depth is not None and Q is not None:
            p3d = pixel_to_3d(pt_a[0], pt_a[1], depth, Q)
            if p3d is not None:
                cv2.putText(out, f"{p3d[2]:.2f}m", (pt_a[0]+8, pt_a[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_a, 1)

    if pt_b is not None:
        cv2.circle(out, pt_b, 6, color_b, -1)
        cv2.circle(out, pt_b, 6, (0, 0, 0), 1)
        if depth is not None and Q is not None:
            p3d = pixel_to_3d(pt_b[0], pt_b[1], depth, Q)
            if p3d is not None:
                cv2.putText(out, f"{p3d[2]:.2f}m", (pt_b[0]+8, pt_b[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_b, 1)

    if pt_a is not None and pt_b is not None:
        cv2.line(out, pt_a, pt_b, color_line, 2)
        if distance is not None:
            mx = (pt_a[0] + pt_b[0]) // 2
            my = (pt_a[1] + pt_b[1]) // 2
            label = f"{distance*100:.1f} cm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (mx-2, my-th-4), (mx+tw+2, my+4), (0, 0, 0), -1)
            cv2.putText(out, label, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line, 2)

    return out


def compute_depth_stats(depth: np.ndarray) -> dict:
    """計算深度統計：min, max, mean, median, valid_ratio"""
    valid = depth[depth > 0]
    total = depth.size
    if len(valid) == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
            "valid_ratio": 0.0, "valid_pixels": 0, "total_pixels": total,
        }
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "valid_ratio": float(len(valid) / total),
        "valid_pixels": int(len(valid)),
        "total_pixels": total,
    }
