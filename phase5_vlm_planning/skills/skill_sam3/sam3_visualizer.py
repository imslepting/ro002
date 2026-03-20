"""SAM3 分割結果可視化工具"""

from __future__ import annotations

import cv2
import numpy as np

# 預定義調色板（BGR）
_PALETTE = [
    (0, 255, 0),     # 綠
    (255, 0, 0),     # 藍
    (0, 0, 255),     # 紅
    (0, 255, 255),   # 黃
    (255, 0, 255),   # 洋紅
    (255, 255, 0),   # 青
    (128, 0, 255),   # 紫
    (0, 128, 255),   # 橙
]


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """bool mask → (x1, y1, x2, y2) bounding box"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (int(x1), int(y1), int(x2), int(y2))


def draw_single_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """單個 mask 半透明疊加"""
    out = image.copy()
    overlay = out.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def annotate_image(
    rgb_image: np.ndarray,
    masks: list[np.ndarray],
    scores: list[float],
    boxes: list[np.ndarray],
    object_description: str,
    alpha: float = 0.4,
) -> np.ndarray:
    """在原圖上疊加 mask 半透明色塊 + bbox + label + score

    Parameters
    ----------
    rgb_image : BGR uint8 (H, W, 3)
    masks : list of (H, W) bool masks
    scores : 每個 mask 的置信度
    boxes : 每個 mask 的 bbox [x1, y1, x2, y2]
    object_description : 物件文字描述
    alpha : mask 疊加透明度

    Returns
    -------
    annotated : BGR uint8 (H, W, 3)
    """
    out = rgb_image.copy()

    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        color = _PALETTE[i % len(_PALETTE)]

        # 疊加半透明 mask
        out = draw_single_mask(out, mask, color, alpha)

        # bbox
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # label
        label = f"#{i} {object_description} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 文字背景
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, label, (x1, y1 - baseline - 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return out
