"""多相機即時畫面拼接顯示"""

from __future__ import annotations

import cv2
import numpy as np

from shared.types import CameraTestResult
from shared.camera_manager import CameraReader
from phase0_hw_diagnostics.src.status_panel import render_status_panel

# 狀態邊框顏色 (BGR)
_BORDER_COLORS = {
    "OK":      (0, 200, 0),
    "WARNING": (0, 200, 255),
    "ERROR":   (0, 0, 220),
}


def _make_no_signal(tile_w: int, tile_h: int) -> np.ndarray:
    """生成 NO SIGNAL 佔位圖"""
    img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8) + 40
    text = "NO SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (tile_w - tw) // 2
    y = (tile_h + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 180), thickness)
    return img


def _overlay_info(
    frame: np.ndarray,
    cam_name: str,
    role: str,
    result: CameraTestResult | None,
) -> np.ndarray:
    """在畫面左上角疊加相機資訊"""
    img = frame.copy()
    status = result.status if result else "ERROR"
    color = _BORDER_COLORS.get(status, (128, 128, 128))

    # 邊框
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)

    # 半透明背景條
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # 文字
    line1 = f"{cam_name} [{role}]"
    if result and result.resolution[0] > 0:
        line2 = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
    else:
        line2 = "---"

    cv2.putText(img, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img


def run_live_display(
    camera_configs: dict,
    test_results: dict[str, CameraTestResult | None],
    tile_size: tuple[int, int] = (640, 480),
    on_save: callable = None,
    on_rescan: callable = None,
) -> None:
    """啟動即時多相機顯示窗口

    Args:
        camera_configs: settings.yaml cameras 段
        test_results: {cam_name: CameraTestResult | None}
        tile_size: 每個相機畫面的目標尺寸 (w, h)
        on_save: 按 S 時的回調
        on_rescan: 按 R 時的回調
    """
    tile_w, tile_h = tile_size
    cam_names = sorted(camera_configs.keys())

    # 啟動相機讀取線程
    readers: dict[str, CameraReader] = {}
    for name in cam_names:
        idx = camera_configs[name]["index"]
        reader = CameraReader(idx)
        reader.start()
        readers[name] = reader

    window_name = "RO002 Hardware Diagnostics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    no_signal = _make_no_signal(tile_w, tile_h)
    panel_w = 400

    try:
        while True:
            tiles: list[np.ndarray] = []
            for name in cam_names:
                reader = readers[name]
                role = camera_configs[name].get("role", "")
                result = test_results.get(name)

                if reader.frame is not None:
                    frame = cv2.resize(reader.frame, (tile_w, tile_h))
                else:
                    frame = no_signal.copy()

                frame = _overlay_info(frame, name, role, result)
                tiles.append(frame)

            # 填充到偶數（2x2 網格）
            while len(tiles) < 4:
                tiles.append(no_signal.copy())

            # 拼接 2x2
            row_top = np.hstack(tiles[:2])
            row_bot = np.hstack(tiles[2:4])
            grid = np.vstack([row_top, row_bot])

            # 狀態面板
            panel = render_status_panel(
                test_results, camera_configs,
                panel_size=(panel_w, grid.shape[0]),
            )

            # 合併
            canvas = np.hstack([grid, panel])
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                if on_save:
                    # 收集當前幀
                    frames = {}
                    for name in cam_names:
                        r = readers[name]
                        if r.frame is not None:
                            frames[name] = r.frame.copy()
                    on_save(frames)
            elif key == ord("r") or key == ord("R"):
                if on_rescan:
                    test_results = on_rescan()
    finally:
        for reader in readers.values():
            reader.stop()
        cv2.destroyAllWindows()
