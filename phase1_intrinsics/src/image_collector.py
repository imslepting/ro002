"""互動式採集 GUI — 相機選擇、模式選擇、拍照採集"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from shared.camera_manager import CameraReader, open_camera, release_camera
from phase1_intrinsics.src.charuco_detector import detect_charuco, draw_detection_overlay, DetectionResult


# ── 常數 ──

TARGET_FRAMES = 30
MIN_FRAMES = 20
COUNTDOWN_SECONDS = 3
COOLDOWN_SECONDS = 1.0
MIN_CORNERS_DETECT = 6


# ── 畫面 1：相機選擇 ──

def run_camera_selection(
    camera_configs: dict,
    calibrated_cameras: set[str],
) -> str | None:
    """顯示相機選擇畫面，返回選中的 cam_name 或 None（退出）

    快捷鍵：1-4 選擇相機，G 生成標定板，Q 退出
    """
    cam_names = sorted(camera_configs.keys())
    window_name = "Phase 1 - Camera Selection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    selected = None

    while True:
        canvas = _draw_camera_selection(cam_names, camera_configs, calibrated_cameras)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            cv2.destroyWindow(window_name)
            return None
        elif key == ord("g") or key == ord("G"):
            cv2.destroyWindow(window_name)
            return "__GENERATE_BOARD__"

        # 數字鍵 1-9
        for i, name in enumerate(cam_names):
            if key == ord(str(i + 1)):
                selected = name
                break

        if selected is not None:
            # 確認：開啟即時預覽
            confirmed = _confirm_camera(selected, camera_configs[selected]["index"])
            if confirmed:
                cv2.destroyWindow(window_name)
                return selected
            else:
                selected = None


def _draw_camera_selection(
    cam_names: list[str],
    camera_configs: dict,
    calibrated: set[str],
) -> np.ndarray:
    """繪製相機選擇畫面"""
    w, h = 700, 450
    canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30

    # Title
    cv2.putText(canvas, "Phase 1 - Intrinsics Calibration",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, "Select camera to calibrate:",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    y = 140
    for i, name in enumerate(cam_names):
        role = camera_configs[name].get("role", "")
        is_cal = name in calibrated
        marker = "●" if is_cal else "○"
        status = "CALIBRATED" if is_cal else "NOT CALIBRATED"
        color = (0, 200, 0) if is_cal else (100, 100, 100)

        line = f"[{i+1}] {name} ({role})"
        cv2.putText(canvas, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        cv2.putText(canvas, status, (420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Calibrated indicator circle
        cx, cy = 35, y - 7
        if is_cal:
            cv2.circle(canvas, (cx, cy), 6, (0, 200, 0), -1)
        else:
            cv2.circle(canvas, (cx, cy), 6, (100, 100, 100), 1)

        y += 50

    # Footer
    y += 20
    cv2.putText(canvas, "Press number key to select  |  G = Generate board PDF  |  Q = Quit",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return canvas


def _confirm_camera(cam_name: str, cam_index: int) -> bool:
    """開啟即時預覽確認相機，ENTER 確認，ESC 返回"""
    window_name = f"Preview - {cam_name} (ENTER=confirm, ESC=back)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    reader = CameraReader(cam_index)
    reader.start()

    confirmed = False
    start = time.time()

    try:
        while True:
            frame = reader.frame
            if frame is not None:
                display = frame.copy()
                cv2.putText(display, f"{cam_name} - ENTER to confirm, ESC to go back",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                display = np.zeros((480, 640, 3), dtype=np.uint8) + 40
                elapsed = time.time() - start
                cv2.putText(display, f"Opening {cam_name}... ({elapsed:.0f}s)",
                            (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # ENTER
                confirmed = True
                break
            elif key == 27:  # ESC
                break
    finally:
        reader.stop()
        cv2.destroyWindow(window_name)

    return confirmed


# ── 畫面 1.5：分辨率選擇 ──

# 常見分辨率候選表（寬, 高）
_COMMON_RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1280, 1024),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def run_resolution_selection(cam_name: str, cam_index: int) -> tuple[int, int] | None:
    """探測相機支持的分辨率，讓用戶選擇。返回 (w, h) 或 None（返回）"""
    print(f"[Resolution] 探測 {cam_name} 支持的分辨率...")
    supported = _probe_resolutions(cam_index)

    if not supported:
        print(f"[Resolution] 無法探測到任何分辨率，使用相機默認")
        return None

    window_name = f"Phase 1 - Resolution ({cam_name})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    selected_idx = 0

    while True:
        canvas = np.zeros((400, 600, 3), dtype=np.uint8) + 30

        cv2.putText(canvas, f"Select resolution for {cam_name}:",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, "UP/DOWN to navigate, ENTER to confirm, ESC to back",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        y = 115
        for i, (w, h) in enumerate(supported):
            is_selected = (i == selected_idx)
            prefix = ">" if is_selected else " "
            label = f"{prefix} [{i+1}] {w} x {h}"
            color = (0, 255, 255) if is_selected else (200, 200, 200)
            thickness = 2 if is_selected else 1
            cv2.putText(canvas, label, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            y += 32

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # ENTER
            cv2.destroyWindow(window_name)
            chosen = supported[selected_idx]
            print(f"[Resolution] {cam_name}: 選擇 {chosen[0]}x{chosen[1]}")
            return chosen
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None
        elif key == 82 or key == 0:  # UP arrow
            selected_idx = (selected_idx - 1) % len(supported)
        elif key == 84 or key == 1:  # DOWN arrow
            selected_idx = (selected_idx + 1) % len(supported)
        else:
            # 數字鍵 1-9 快速選擇
            for i in range(min(len(supported), 9)):
                if key == ord(str(i + 1)):
                    cv2.destroyWindow(window_name)
                    chosen = supported[i]
                    print(f"[Resolution] {cam_name}: 選擇 {chosen[0]}x{chosen[1]}")
                    return chosen


def _probe_resolutions(cam_index: int) -> list[tuple[int, int]]:
    """探測相機實際支持的分辨率

    用單一 VideoCapture 依序 set → read → 取 frame.shape，
    避免反覆開關設備導致 USB 相機不穩定。
    """
    cap = open_camera(cam_index)
    if not cap.isOpened():
        return []

    supported = []
    for w, h in _COMMON_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(2):
            ret, frame = cap.read()
        if ret and frame is not None:
            actual_h, actual_w = frame.shape[:2]
            res = (actual_w, actual_h)
            if res not in supported:
                supported.append(res)

    release_camera(cap)
    supported.sort(key=lambda r: r[0] * r[1])
    return supported


# ── 畫面 2：模式選擇 ──

def run_mode_selection() -> str | None:
    """顯示模式選擇畫面，返回 'auto' / 'manual' / None（返回）"""
    window_name = "Phase 1 - Mode Selection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        canvas = np.zeros((300, 600, 3), dtype=np.uint8) + 30
        cv2.putText(canvas, "Select capture mode:", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(canvas, "[A] Auto - countdown after detection", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(canvas, "[M] Manual - press SPACE to capture", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(canvas, "ESC = Back", (50, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("a") or key == ord("A"):
            cv2.destroyWindow(window_name)
            return "auto"
        elif key == ord("m") or key == ord("M"):
            cv2.destroyWindow(window_name)
            return "manual"
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None


# ── 畫面 3：採集 ──

def run_collection(
    cam_name: str,
    cam_index: int,
    mode: str,
    board,
    dictionary,
    output_dir: str,
    resolution: tuple[int, int] | None = None,
) -> list[str]:
    """執行採集流程，返回已保存的圖片路徑列表

    Args:
        cam_name: 相機名稱
        cam_index: 相機設備索引
        mode: 'auto' or 'manual'
        board: CharucoBoard 對象
        dictionary: ArUco 字典
        output_dir: 圖片保存目錄
        resolution: 指定的採集分辨率 (w, h)，None 則使用相機默認

    Returns:
        已保存的圖片路徑列表
    """
    os.makedirs(output_dir, exist_ok=True)

    reader = CameraReader(cam_index, resolution=resolution)
    reader.start()

    saved_paths: list[str] = []
    window_name = f"Capture - {cam_name} ({mode})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Auto mode state
    countdown_start: float | None = None
    cooldown_until: float = 0.0
    auto_paused = False

    # Coverage tracking
    all_detected_corners: list[np.ndarray] = []

    try:
        while True:
            frame = reader.frame
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8) + 40
                cv2.putText(blank, "Waiting for camera...", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
                cv2.imshow(window_name, blank)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:
                    break
                continue

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            # Detect charuco (no subpix for preview speed)
            detection = detect_charuco(gray, board, dictionary, refine_subpix=False)

            # Draw detection overlay
            if detection.aruco_corners or detection.charuco_corners is not None:
                display = draw_detection_overlay(display, detection)

            # Calculate coverage
            coverage = _estimate_coverage(all_detected_corners, gray.shape)

            # ── Auto mode logic ──
            if mode == "auto" and not auto_paused:
                if now < cooldown_until:
                    # Cooldown period - show green flash
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                                  (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                elif detection.success and detection.num_corners >= MIN_CORNERS_DETECT:
                    if countdown_start is None:
                        countdown_start = now
                    elapsed = now - countdown_start
                    remaining = COUNTDOWN_SECONDS - elapsed

                    if remaining <= 0:
                        # Auto capture!
                        path = _save_frame(frame, cam_name, len(saved_paths), output_dir)
                        saved_paths.append(path)
                        if detection.charuco_corners is not None:
                            all_detected_corners.append(detection.charuco_corners.copy())
                        countdown_start = None
                        cooldown_until = now + COOLDOWN_SECONDS
                        print(f"[Capture] Auto: saved frame {len(saved_paths)} → {path}")
                    else:
                        # Show countdown
                        _draw_countdown(display, remaining)
                else:
                    # Lost detection, reset countdown
                    countdown_start = None

            # ── Draw info panel ──
            display = _draw_info_panel(
                display, cam_name, mode, len(saved_paths), detection.num_corners,
                coverage, auto_paused,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            # ── Key handling ──
            should_capture = False

            if key == ord(" "):  # SPACE = capture (both modes)
                should_capture = True
            elif key == ord("d") or key == ord("D"):  # Delete last
                if saved_paths:
                    removed = saved_paths.pop()
                    if all_detected_corners:
                        all_detected_corners.pop()
                    try:
                        os.remove(removed)
                    except OSError:
                        pass
                    print(f"[Capture] Deleted: {removed}")
            elif key == ord("p") or key == ord("P"):  # Pause/resume auto
                if mode == "auto":
                    auto_paused = not auto_paused
                    countdown_start = None
                    print(f"[Capture] Auto {'paused' if auto_paused else 'resumed'}")
            elif key == ord("q") or key == ord("Q"):
                if len(saved_paths) < MIN_FRAMES:
                    if not _confirm_quit(len(saved_paths), window_name):
                        continue
                break
            elif key == 27:  # ESC
                break

            if should_capture and detection.success:
                path = _save_frame(frame, cam_name, len(saved_paths), output_dir)
                saved_paths.append(path)
                if detection.charuco_corners is not None:
                    all_detected_corners.append(detection.charuco_corners.copy())
                print(f"[Capture] Manual: saved frame {len(saved_paths)} → {path}")

                # Brief green flash feedback
                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (display.shape[1], display.shape[0]),
                              (0, 255, 0), 8)
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)

    finally:
        reader.stop()
        cv2.destroyWindow(window_name)

    print(f"[Capture] {cam_name}: 採集完成，共 {len(saved_paths)} 幀")
    return saved_paths


def _save_frame(frame: np.ndarray, cam_name: str, idx: int, output_dir: str) -> str:
    """保存採集幀"""
    filename = f"{cam_name}_{idx:03d}.png"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame)
    return path


def _draw_countdown(display: np.ndarray, remaining: float) -> None:
    """在畫面中央繪製倒數大字"""
    h, w = display.shape[:2]
    text = str(int(remaining) + 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5.0
    thickness = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2

    # Semi-transparent background
    overlay = display.copy()
    cv2.circle(overlay, (w // 2, h // 2), 100, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

    cv2.putText(display, text, (x, y), font, scale, (0, 255, 255), thickness)


def _draw_info_panel(
    display: np.ndarray,
    cam_name: str,
    mode: str,
    frame_count: int,
    num_corners: int,
    coverage: float,
    auto_paused: bool,
) -> np.ndarray:
    """在右側繪製資訊面板"""
    h, w = display.shape[:2]
    panel_w = 220
    total_w = w + panel_w

    canvas = np.zeros((h, total_w, 3), dtype=np.uint8) + 30
    canvas[:h, :w] = display

    # Panel content
    x = w + 15
    y = 35
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, cam_name, (x, y), font, 0.8, (255, 255, 255), 2)
    y += 35
    mode_text = f"Mode: {mode.upper()}"
    if auto_paused:
        mode_text += " (PAUSED)"
    cv2.putText(canvas, mode_text, (x, y), font, 0.5, (180, 180, 180), 1)
    y += 40

    # Frame counter with color
    color = (0, 200, 0) if frame_count >= TARGET_FRAMES else (200, 200, 200)
    cv2.putText(canvas, f"Frames: {frame_count}/{TARGET_FRAMES}", (x, y), font, 0.6, color, 1)
    y += 30

    # Progress bar
    bar_x, bar_y = x, y
    bar_w, bar_h = panel_w - 30, 15
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * min(frame_count / TARGET_FRAMES, 1.0))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)
    y += 35

    cv2.putText(canvas, f"Corners: {num_corners}", (x, y), font, 0.5, (180, 180, 180), 1)
    y += 25
    cv2.putText(canvas, f"Coverage: {coverage:.0f}%", (x, y), font, 0.5, (180, 180, 180), 1)
    y += 50

    # Key hints
    if mode == "manual":
        hints = [
            ("SPACE", "Capture"),
            ("D", "Delete last"),
            ("Q", "Finish"),
        ]
    else:
        hints = [
            ("SPACE", "Force capture"),
            ("P", "Pause/Resume"),
            ("D", "Delete last"),
            ("Q", "Finish"),
        ]

    for key_name, desc in hints:
        cv2.putText(canvas, f"{key_name} = {desc}", (x, y), font, 0.45, (120, 120, 120), 1)
        y += 22

    # Status message at bottom
    if frame_count >= TARGET_FRAMES:
        cv2.putText(canvas, "Target reached!", (x, h - 50), font, 0.5, (0, 200, 0), 1)
        cv2.putText(canvas, "Press Q to calibrate", (x, h - 25), font, 0.5, (0, 200, 0), 1)
    elif frame_count >= MIN_FRAMES:
        cv2.putText(canvas, f"Min reached ({MIN_FRAMES})", (x, h - 50), font, 0.5, (0, 200, 200), 1)
        cv2.putText(canvas, "Q=calibrate or continue", (x, h - 25), font, 0.45, (0, 200, 200), 1)

    return canvas


def _estimate_coverage(corners_list: list[np.ndarray], image_shape: tuple) -> float:
    """估算角點在畫面中的覆蓋率（簡易：用凸包面積/畫面面積）"""
    if not corners_list:
        return 0.0

    all_pts = np.vstack([c.reshape(-1, 2) for c in corners_list])
    h, w = image_shape[:2]
    image_area = w * h

    if len(all_pts) < 3:
        return 0.0

    hull = cv2.convexHull(all_pts.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    return min(hull_area / image_area * 100, 100.0)


def _confirm_quit(frame_count: int, parent_window: str) -> bool:
    """未達最低幀數時的退出確認"""
    window_name = "Confirm Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        canvas = np.zeros((200, 500, 3), dtype=np.uint8) + 30
        cv2.putText(canvas, f"Only {frame_count}/{MIN_FRAMES} frames collected!",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        cv2.putText(canvas, "Calibration may be poor or fail.",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(canvas, "Y = Quit anyway  |  N = Continue capturing",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("y") or key == ord("Y"):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord("n") or key == ord("N") or key == 27:
            cv2.destroyWindow(window_name)
            return False
