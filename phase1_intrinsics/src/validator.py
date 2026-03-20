"""驗收畫面 — 重投影誤差可視化 + 去畸變對比"""

from __future__ import annotations

import os

import cv2
import numpy as np

from shared.types import CalibResult
from phase1_intrinsics.src.calibrator import compute_per_frame_errors


def run_validation(
    calib_result: CalibResult,
    image_paths: list[str],
    board,
    dictionary,
    max_rms: float,
    reports_dir: str,
) -> bool:
    """顯示驗收 GUI，返回 True=接受 / False=重拍

    Layout:
    +---------------------+---------------------+-----------+
    |     ORIGINAL        |    UNDISTORTED       | 資訊面板   |
    +---------------------+---------------------+           |
    |  逐幀誤差條形圖                             |           |
    +--------------------------------------------+-----------+
    """
    K, D = calib_result.K, calib_result.D
    cam_name = calib_result.cam_name

    # 計算逐幀誤差
    per_frame = compute_per_frame_errors(image_paths, board, dictionary, K, D)
    if not per_frame:
        print("[Validator] 無有效幀可供驗證")
        return False

    errors = [e for _, e in per_frame]
    valid_paths = [p for p, _ in per_frame]
    passed = calib_result.rms <= max_rms

    # 保存誤差報告圖
    bar_chart = _draw_bar_chart(errors, max_rms, cam_name, width=800, height=250)
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{cam_name}_reprojection_errors.png")
    cv2.imwrite(report_path, bar_chart)
    print(f"[Validator] 誤差報告已保存: {report_path}")

    # GUI
    sample_idx = 0
    window_name = f"Validation - {cam_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        canvas = _build_validation_canvas(
            valid_paths[sample_idx], K, D, calib_result, bar_chart, errors,
            sample_idx, len(valid_paths), passed, max_rms,
        )
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("a") or key == ord("A"):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord("r") or key == ord("R"):
            cv2.destroyWindow(window_name)
            return False
        elif key == 81 or key == 2:  # LEFT arrow
            sample_idx = (sample_idx - 1) % len(valid_paths)
        elif key == 83 or key == 3:  # RIGHT arrow
            sample_idx = (sample_idx + 1) % len(valid_paths)
        elif key == ord("q") or key == ord("Q") or key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return False


def _build_validation_canvas(
    image_path: str,
    K: np.ndarray,
    D: np.ndarray,
    calib: CalibResult,
    bar_chart: np.ndarray,
    errors: list[float],
    sample_idx: int,
    total_samples: int,
    passed: bool,
    max_rms: float,
) -> np.ndarray:
    """組合驗收畫面"""
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    display_w, display_h = 480, 360

    # Original
    orig_resized = cv2.resize(img, (display_w, display_h))
    cv2.putText(orig_resized, "ORIGINAL", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Undistorted
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undist = cv2.undistort(img, K, D, None, new_K)
    undist_resized = cv2.resize(undist, (display_w, display_h))
    cv2.putText(undist_resized, "UNDISTORTED", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Top row: original + undistorted
    top_row = np.hstack([orig_resized, undist_resized])

    # Bottom row: bar chart resized to match width
    chart_resized = cv2.resize(bar_chart, (top_row.shape[1], 250))

    # Left panel (images + chart)
    left_panel = np.vstack([top_row, chart_resized])

    # Right info panel
    panel_h = left_panel.shape[0]
    panel_w = 280
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 30

    y = 40
    _put_text(panel, f"{calib.cam_name}", (15, y), scale=0.9, color=(255, 255, 255))
    y += 40
    _put_text(panel, f"RMS: {calib.rms:.4f} px", (15, y), scale=0.7)
    y += 30

    status_color = (0, 200, 0) if passed else (0, 0, 220)
    status_text = "PASS" if passed else "FAIL"
    _put_text(panel, status_text, (15, y), scale=0.8, color=status_color)
    y += 30
    _put_text(panel, f"Threshold: {max_rms:.2f} px", (15, y), scale=0.5, color=(180, 180, 180))
    y += 40

    _put_text(panel, f"Image: {sample_idx+1}/{total_samples}", (15, y), scale=0.5)
    y += 30
    _put_text(panel, f"Size: {calib.image_size[0]}x{calib.image_size[1]}", (15, y), scale=0.5)
    y += 40

    # K matrix display
    _put_text(panel, "K (intrinsics):", (15, y), scale=0.5, color=(180, 180, 180))
    y += 22
    for row in calib.K:
        row_str = " ".join(f"{v:8.1f}" for v in row)
        _put_text(panel, row_str, (15, y), scale=0.4)
        y += 18

    y += 15
    _put_text(panel, "D (distortion):", (15, y), scale=0.5, color=(180, 180, 180))
    y += 22
    d_str = " ".join(f"{v:.4f}" for v in calib.D.flatten()[:5])
    _put_text(panel, d_str, (15, y), scale=0.4)

    y += 50
    _put_text(panel, "A = Accept", (15, y), scale=0.6, color=(0, 200, 0))
    y += 30
    _put_text(panel, "R = Redo", (15, y), scale=0.6, color=(0, 0, 220))
    y += 30
    _put_text(panel, "LEFT/RIGHT = Switch", (15, y), scale=0.5, color=(180, 180, 180))

    canvas = np.hstack([left_panel, panel])
    return canvas


def _draw_bar_chart(
    errors: list[float],
    threshold: float,
    cam_name: str,
    width: int = 800,
    height: int = 250,
) -> np.ndarray:
    """繪製逐幀誤差條形圖"""
    chart = np.zeros((height, width, 3), dtype=np.uint8) + 30
    n = len(errors)
    if n == 0:
        return chart

    margin_left, margin_right = 60, 20
    margin_top, margin_bottom = 30, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_err = max(max(errors), threshold * 1.2)
    bar_w = max(2, plot_w // n - 2)

    # Threshold line
    thr_y = margin_top + int((1 - threshold / max_err) * plot_h)
    cv2.line(chart, (margin_left, thr_y), (width - margin_right, thr_y), (0, 0, 180), 1)
    _put_text(chart, f"{threshold:.2f}", (5, thr_y + 4), scale=0.35, color=(0, 0, 180))

    # Bars
    for i, err in enumerate(errors):
        bar_h = int((err / max_err) * plot_h)
        x = margin_left + i * (plot_w // n)
        y_top = margin_top + plot_h - bar_h
        y_bot = margin_top + plot_h

        color = (0, 200, 0) if err <= threshold else (0, 0, 220)
        cv2.rectangle(chart, (x, y_top), (x + bar_w, y_bot), color, -1)

    # Axis labels
    _put_text(chart, f"Per-frame reprojection error ({cam_name})",
              (margin_left, 18), scale=0.5, color=(200, 200, 200))
    _put_text(chart, "Frame #", (width // 2 - 20, height - 8),
              scale=0.4, color=(150, 150, 150))

    return chart


def _put_text(img, text, pos, scale=0.5, color=(200, 200, 200), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
