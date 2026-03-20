"""設備狀態面板渲染（純 OpenCV 繪圖）"""

from __future__ import annotations

import cv2
import numpy as np

from shared.types import CameraTestResult

# 狀態顏色 (BGR)
_COLORS = {
    "OK":      (0, 200, 0),
    "WARNING": (0, 200, 255),
    "ERROR":   (0, 0, 220),
}


def render_status_panel(
    test_results: dict[str, CameraTestResult | None],
    expected_cameras: dict,
    panel_size: tuple[int, int] = (400, 960),
) -> np.ndarray:
    """渲染狀態摘要面板，返回 BGR 圖像

    Args:
        test_results: {cam_name: CameraTestResult | None}
        expected_cameras: settings.yaml 中的 cameras 配置
        panel_size: (width, height)
    """
    pw, ph = panel_size
    panel = np.zeros((ph, pw, 3), dtype=np.uint8) + 30  # 深灰底

    y = 40
    # 標題
    cv2.putText(panel, "DEVICE STATUS", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += 15
    cv2.line(panel, (20, y), (pw - 20, y), (100, 100, 100), 1)
    y += 35

    # 統計
    n_expected = len(expected_cameras)
    n_ok = sum(1 for r in test_results.values() if r and r.status == "OK")
    n_warn = sum(1 for r in test_results.values() if r and r.status == "WARNING")
    n_err = sum(1 for r in test_results.values() if r is None or r.status == "ERROR")

    # 逐台相機
    for cam_name in sorted(expected_cameras.keys()):
        cam_info = expected_cameras[cam_name]
        result = test_results.get(cam_name)
        role = cam_info.get("role", "")

        if result is None:
            status = "ERROR"
            status_text = "NOT FOUND"
        else:
            status = result.status
            status_text = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"

        color = _COLORS.get(status, (128, 128, 128))

        # 狀態圓點
        cv2.circle(panel, (35, y - 5), 8, color, -1)
        # 相機名稱 + 角色
        cv2.putText(panel, f"{cam_name} [{role}]", (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y += 25
        # 詳細資訊
        cv2.putText(panel, f"  {status_text}", (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        y += 20

        # 警告
        if result and result.warnings:
            for w in result.warnings:
                cv2.putText(panel, f"  ! {w}", (55, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y += 18
        y += 15

    # 分隔線
    cv2.line(panel, (20, y), (pw - 20, y), (100, 100, 100), 1)
    y += 30

    # 總結
    if n_err == 0 and n_warn == 0:
        overall = "READY"
        overall_color = _COLORS["OK"]
    elif n_err == 0:
        overall = "WARNINGS"
        overall_color = _COLORS["WARNING"]
    else:
        overall = "ISSUES DETECTED"
        overall_color = _COLORS["ERROR"]

    cv2.putText(panel, f"Cameras: {n_ok}/{n_expected} OK", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += 30
    cv2.putText(panel, overall, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)
    y += 50

    # 快捷鍵
    cv2.line(panel, (20, y), (pw - 20, y), (100, 100, 100), 1)
    y += 30
    shortcuts = [
        ("[Q] Quit", (200, 200, 200)),
        ("[S] Save Report", (200, 200, 200)),
        ("[R] Re-scan", (200, 200, 200)),
    ]
    for text, color in shortcuts:
        cv2.putText(panel, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 28

    return panel
