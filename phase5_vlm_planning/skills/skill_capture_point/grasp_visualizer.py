"""抓取位姿可視化工具

在 RGB 圖像上疊加抓取位置標記（十字 + 夾爪方向線 + score）。
"""

from __future__ import annotations

import cv2
import numpy as np


def project_point_to_pixel(
    point_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[int, int]:
    """3D 點（相機座標系）→ 像素座標 (u, v)

    Parameters
    ----------
    point_cam : (3,) 相機座標系下的 3D 位置
    K : 3×3 內參矩陣

    Returns
    -------
    (u, v) 像素座標
    """
    x, y, z = point_cam
    if z <= 0:
        return 0, 0
    u = int(round(K[0, 0] * x / z + K[0, 2]))
    v = int(round(K[1, 1] * y / z + K[1, 2]))
    return u, v


# 向後兼容
project_pose_to_pixel = project_point_to_pixel


def compute_contact_point(
    grasp_pose: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    """沿 approach 方向找到最接近物體點雲的接觸點

    GraspGen 的 grasp pose 是夾爪 TCP 位置，通常在物體表面外。
    此函數沿 approach 方向（z 軸）將 TCP 投影到物體點雲質心的深度。

    Parameters
    ----------
    grasp_pose : (4, 4) 抓取位姿（arm 座標系）
    object_points : (N, 3) 物體點雲（arm 座標系）

    Returns
    -------
    contact_point : (3,) 接觸點（arm 座標系）
    """
    tcp = grasp_pose[:3, 3]
    approach = grasp_pose[:3, 2]  # z 軸 = approach direction

    # 物體點雲在 approach 軸上的投影
    # 找到 TCP 沿 approach 到物體最近距離的 t
    diff = object_points - tcp  # (N, 3)
    proj_t = diff @ approach    # (N,) 各點在 approach 方向的投影距離

    # 取中位數作為穩健的接觸深度
    t_contact = float(np.median(proj_t))

    contact = tcp + t_contact * approach
    return contact


def annotate_grasp(
    rgb_image: np.ndarray,
    grasp_pixel: tuple[int, int],
    grasp_score: float,
    grasp_width_px: float,
    rotation_2d: float,
    label: str = "",
    tcp_pixel: tuple[int, int] | None = None,
) -> np.ndarray:
    """在 RGB 圖上畫抓取位置標記

    繪製：十字準星（接觸點）+ 夾爪方向線（依 rotation_2d）+ score 文字
    若提供 tcp_pixel，額外畫 TCP→contact 的 approach 箭頭。

    Parameters
    ----------
    rgb_image : BGR uint8 (H, W, 3)
    grasp_pixel : (u, v) 抓取接觸點像素座標（物體表面）
    grasp_score : 置信度
    grasp_width_px : 投影後的像素寬度（夾爪張開寬度）
    rotation_2d : 投影後的旋轉角度（弧度），0 = 水平
    label : 可選文字標籤
    tcp_pixel : (u, v) TCP 像素座標，用於畫 approach 箭頭（可選）

    Returns
    -------
    annotated : BGR uint8，標注後的圖像（副本）
    """
    img = rgb_image.copy()
    u, v = grasp_pixel
    h, w = img.shape[:2]

    # 顏色：綠色系，score 越高越亮
    intensity = int(128 + 127 * min(grasp_score, 1.0))
    color = (0, intensity, 0)  # BGR green
    color_text = (0, 255, 255)  # BGR yellow
    color_approach = (0, 0, 255)  # BGR 亮紅 — approach 方向

    # approach 箭頭：TCP → contact
    if tcp_pixel is not None:
        tu, tv = tcp_pixel
        cv2.arrowedLine(img, (tu, tv), (u, v), color_approach, 2, tipLength=0.15)
        # TCP 小圓點
        cv2.circle(img, (tu, tv), 4, color_approach, -1)

    # 十字準星（接觸點）
    cross_size = 12
    thickness = 2
    cv2.line(img, (u - cross_size, v), (u + cross_size, v), color, thickness)
    cv2.line(img, (u, v - cross_size), (u, v + cross_size), color, thickness)

    # 夾爪方向線（兩端）
    half_w = max(grasp_width_px / 2, 15)
    dx = half_w * np.cos(rotation_2d)
    dy = half_w * np.sin(rotation_2d)

    p1 = (int(u - dx), int(v - dy))
    p2 = (int(u + dx), int(v + dy))
    cv2.line(img, p1, p2, color, thickness + 1)

    # 夾爪端點垂直短線（表示夾爪指尖）
    perp_len = 10
    px = perp_len * np.cos(rotation_2d + np.pi / 2)
    py = perp_len * np.sin(rotation_2d + np.pi / 2)
    for pt in [p1, p2]:
        cv2.line(
            img,
            (int(pt[0] - px), int(pt[1] - py)),
            (int(pt[0] + px), int(pt[1] + py)),
            color, thickness,
        )

    # Score 文字
    text = f"{grasp_score:.2f}"
    if label:
        text = f"{label} {text}"
    text_pos = (u + cross_size + 4, v - 4)
    # 確保文字不超出邊界
    text_pos = (max(0, min(text_pos[0], w - 100)), max(15, text_pos[1]))
    cv2.putText(
        img, text, text_pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA,
    )

    return img
