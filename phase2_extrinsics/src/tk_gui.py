"""Phase 2 — 雙目外參標定 tkinter GUI

4 個畫面 + 畫面管理器：
  1. PairSelectionScreen — 配對選擇
  2. ModeSelectionScreen — 模式選擇
  3. StereoCollectionScreen — 雙目採集
  4. StereoValidationScreen — 校正驗收
"""

from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

from shared.camera_manager import CameraReader
from shared.tk_utils import (
    DARK_BG, PANEL_BG, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    FONT_TITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    BTN_STYLE, BTN_ACCENT, BTN_DANGER,
    cv_to_photoimage, CameraFeedWidget, run_in_thread,
)
from phase1_intrinsics.src.charuco_detector import detect_charuco, draw_detection_overlay
from phase1_intrinsics.src.intrinsics_io import load_calib_result, get_calibrated_cameras
from phase2_extrinsics.src.stereo_calibrator import (
    StereoCalibResult, calibrate_stereo,
    compute_stereo_rectification, compute_epipolar_error,
)
from phase2_extrinsics.src.extrinsics_io import (
    save_extrinsics, load_pair_result, get_calibrated_pairs,
)


# ── 常數 ──

TARGET_PAIRS = 30
MIN_PAIRS = 15
COUNTDOWN_SECONDS = 3
COOLDOWN_SECONDS = 1.0
MIN_CORNERS_DETECT = 6


# ── 畫面管理器 ──

class ScreenManager:
    """管理畫面切換"""
    def __init__(self, root: tk.Tk):
        self.root = root
        self._current: tk.Frame | None = None

    def show(self, screen: tk.Frame) -> None:
        if self._current is not None:
            self._current.pack_forget()
        self._current = screen
        screen.pack(fill="both", expand=True)


# ── 結果容器 ──

class GUIResult:
    """在畫面間傳遞結果"""
    def __init__(self):
        self.cam_left: str | None = None
        self.cam_left_index: int = 0
        self.cam_right: str | None = None
        self.cam_right_index: int = 0
        self.pair_name: str | None = None
        self.mode: str | None = None
        self.left_paths: list[str] = []
        self.right_paths: list[str] = []
        self.stereo_result: StereoCalibResult | None = None
        self.accepted: bool = False
        self.quit: bool = False


# ── 畫面 1：配對選擇 ──

class PairSelectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, camera_configs: dict,
                 intrinsics_path: str, extrinsics_path: str, result: GUIResult):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._camera_configs = camera_configs
        self._intrinsics_path = intrinsics_path
        self._extrinsics_path = extrinsics_path
        self._result = result
        self._preview_readers: list[CameraReader] = []
        self._preview_feeds: list[CameraFeedWidget] = []
        self._preview_win: tk.Toplevel | None = None

        self._build()

    def _build(self):
        tk.Label(
            self, text="Phase 2 — Extrinsics Calibration",
            font=("Helvetica", 20, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(30, 5))

        tk.Label(
            self, text="Select camera pair:",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_SECONDARY,
        ).pack(pady=(5, 20))

        calibrated_cams = get_calibrated_cameras(self._intrinsics_path)
        calibrated_pairs = get_calibrated_pairs(self._extrinsics_path)

        cam_names = sorted(self._camera_configs.keys())
        list_frame = tk.Frame(self, bg=DARK_BG)
        list_frame.pack(pady=5)

        # 產生所有配對組合
        for i in range(len(cam_names)):
            for j in range(i + 1, len(cam_names)):
                left, right = cam_names[i], cam_names[j]
                pair_name = f"{left}_{right}"
                has_left = left in calibrated_cams
                has_right = right in calibrated_cams
                both_cal = has_left and has_right

                # 已有外參？
                pair_result = load_pair_result(pair_name, self._extrinsics_path)

                self._add_pair_row(
                    list_frame, left, right, pair_name,
                    both_cal, has_left, has_right, pair_result,
                )

        # Quit button
        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=30)
        tk.Button(
            btn_frame, text="Quit", command=self._on_quit, **BTN_DANGER,
        ).pack(padx=10)

    def _add_pair_row(self, parent, left, right, pair_name,
                      both_cal, has_left, has_right, pair_result):
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill="x", padx=20, pady=4)

        # Status dot
        dot_canvas = tk.Canvas(row, width=20, height=20, bg=DARK_BG,
                               highlightthickness=0)
        dot_canvas.pack(side="left", padx=(0, 8))

        if pair_result is not None:
            dot_canvas.create_oval(4, 4, 16, 16, fill=ACCENT_GREEN, outline="")
        elif both_cal:
            dot_canvas.create_oval(4, 4, 16, 16, outline=TEXT_DIM, width=2)
        else:
            dot_canvas.create_oval(4, 4, 16, 16, outline=ACCENT_RED, width=2)

        # Pair name
        tk.Label(
            row, text=f"{left} \u2194 {right}",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_PRIMARY, width=16, anchor="w",
        ).pack(side="left")

        # Status text
        if pair_result is not None:
            status = f"CALIBRATED (RMS: {pair_result['rms']:.2f})"
            status_color = ACCENT_GREEN
        elif not both_cal:
            missing = []
            if not has_left:
                missing.append(left)
            if not has_right:
                missing.append(right)
            status = f"\u26a0 {', '.join(missing)} intrinsics missing"
            status_color = ACCENT_YELLOW
        else:
            status = "NOT CALIBRATED"
            status_color = TEXT_DIM

        tk.Label(
            row, text=status,
            font=FONT_SMALL, bg=DARK_BG, fg=status_color, width=32, anchor="w",
        ).pack(side="left")

        # Select button
        btn_state = "normal" if both_cal else "disabled"
        tk.Button(
            row, text="Select", state=btn_state,
            command=lambda l=left, r=right, p=pair_name: self._on_select(l, r, p),
            **BTN_STYLE,
        ).pack(side="right", padx=5)

    def _on_select(self, cam_left: str, cam_right: str, pair_name: str):
        idx_left = self._camera_configs[cam_left]["index"]
        idx_right = self._camera_configs[cam_right]["index"]
        self._show_preview(cam_left, idx_left, cam_right, idx_right, pair_name)

    def _show_preview(self, cam_left, idx_left, cam_right, idx_right, pair_name):
        if self._preview_win is not None:
            return

        reader_l = CameraReader(idx_left)
        reader_l.start()
        reader_r = CameraReader(idx_right)
        reader_r.start()
        self._preview_readers = [reader_l, reader_r]

        win = tk.Toplevel(self, bg=DARK_BG)
        win.title(f"Preview — {pair_name}")
        win.geometry("1000x560")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_preview(win))
        self._preview_win = win

        tk.Label(
            win, text=f"Preview: {cam_left} \u2194 {cam_right}",
            font=FONT_TITLE, bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(10, 5))

        feed_frame = tk.Frame(win, bg=DARK_BG)
        feed_frame.pack(pady=5)

        feed_l = CameraFeedWidget(feed_frame, reader_l, (460, 345))
        feed_l.pack(side="left", padx=5)
        feed_l.start_feed()

        feed_r = CameraFeedWidget(feed_frame, reader_r, (460, 345))
        feed_r.pack(side="left", padx=5)
        feed_r.start_feed()

        self._preview_feeds = [feed_l, feed_r]

        # Labels
        lbl_frame = tk.Frame(win, bg=DARK_BG)
        lbl_frame.pack()
        tk.Label(lbl_frame, text=f"LEFT: {cam_left}", font=FONT_SMALL,
                 bg=DARK_BG, fg=TEXT_SECONDARY, width=30).pack(side="left")
        tk.Label(lbl_frame, text=f"RIGHT: {cam_right}", font=FONT_SMALL,
                 bg=DARK_BG, fg=TEXT_SECONDARY, width=30).pack(side="left")

        btn_frame = tk.Frame(win, bg=DARK_BG)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Confirm",
            command=lambda: self._confirm_preview(cam_left, idx_left,
                                                   cam_right, idx_right,
                                                   pair_name, win),
            **BTN_ACCENT,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Back",
            command=lambda: self._close_preview(win),
            **BTN_STYLE,
        ).pack(side="left", padx=10)

    def _confirm_preview(self, cam_left, idx_left, cam_right, idx_right,
                         pair_name, win):
        self._result.cam_left = cam_left
        self._result.cam_left_index = idx_left
        self._result.cam_right = cam_right
        self._result.cam_right_index = idx_right
        self._result.pair_name = pair_name
        self._close_preview(win)
        self._result.quit = False
        self.event_generate("<<PairSelected>>")

    def _close_preview(self, win: tk.Toplevel):
        for feed in self._preview_feeds:
            feed.stop_feed()
        for reader in self._preview_readers:
            reader.stop()
        self._preview_readers = []
        self._preview_feeds = []
        win.destroy()
        self._preview_win = None

    def _on_quit(self):
        self._result.quit = True
        self.event_generate("<<PairSelected>>")

    def refresh(self):
        """Refresh the pair list"""
        for w in self.winfo_children():
            w.destroy()
        self._build()


# ── 畫面 2：模式選擇 ──

class ModeSelectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._build()

    def _build(self):
        tk.Label(
            self, text="Select capture mode:",
            font=("Helvetica", 18, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(40, 30))

        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Auto — countdown when both cameras detect board",
            command=lambda: self._select("auto"),
            width=50, height=2, **BTN_ACCENT,
        ).pack(pady=10)

        tk.Button(
            btn_frame, text="Manual — press button to capture pair",
            command=lambda: self._select("manual"),
            width=50, height=2, **BTN_STYLE,
        ).pack(pady=10)

        tk.Button(
            self, text="Back", command=self._on_back, **BTN_STYLE,
        ).pack(pady=20)

    def _select(self, mode: str):
        self._result.mode = mode
        self.event_generate("<<ModeSelected>>")

    def _on_back(self):
        self._result.mode = None
        self.event_generate("<<ModeSelected>>")


# ── 畫面 3：雙目採集 ──

class StereoCollectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult,
                 board, dictionary, pair_dir: str):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._board = board
        self._dictionary = dictionary
        self._pair_dir = pair_dir

        self._reader_l: CameraReader | None = None
        self._reader_r: CameraReader | None = None
        self._feed_l: CameraFeedWidget | None = None
        self._feed_r: CameraFeedWidget | None = None

        self._saved_left: list[str] = []
        self._saved_right: list[str] = []
        self._pair_count = 0

        # Detection state (updated from overlay)
        self._left_detection = None
        self._right_detection = None

        # Auto mode state
        self._countdown_start: float | None = None
        self._cooldown_until: float = 0.0
        self._auto_paused = False
        self._auto_timer_id: str | None = None

        self._build()

    def _build(self):
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left camera feed
        self._feed_frame_l = tk.Frame(main, bg=DARK_BG)
        self._feed_frame_l.pack(side="left", padx=(0, 2))

        # Right camera feed
        self._feed_frame_r = tk.Frame(main, bg=DARK_BG)
        self._feed_frame_r.pack(side="left", padx=(2, 5))

        # Right: info panel
        panel = tk.Frame(main, bg=PANEL_BG, width=260)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._pair_label = tk.Label(
            panel, text="", font=FONT_TITLE, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._pair_label.pack(pady=(15, 5))

        self._mode_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._mode_label.pack()

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        self._pairs_label = tk.Label(
            panel, text="Pairs: 0/30", font=FONT_BODY, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._pairs_label.pack(pady=(5, 2))

        self._progress = ttk.Progressbar(panel, length=220, mode="determinate",
                                         maximum=TARGET_PAIRS)
        self._progress.pack(padx=15, pady=5)

        self._left_det_label = tk.Label(
            panel, text="Left: -- pts", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._left_det_label.pack()

        self._right_det_label = tk.Label(
            panel, text="Right: -- pts", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._right_det_label.pack()

        self._both_label = tk.Label(
            panel, text="Both OK: --", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._both_label.pack()

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        # Buttons
        btn_frame = tk.Frame(panel, bg=PANEL_BG)
        btn_frame.pack(fill="x", padx=15)

        self._capture_btn = tk.Button(
            btn_frame, text="Capture", command=self._on_capture, **BTN_ACCENT,
        )
        self._capture_btn.pack(fill="x", pady=3)

        self._delete_btn = tk.Button(
            btn_frame, text="Delete Last", command=self._on_delete_last, **BTN_STYLE,
        )
        self._delete_btn.pack(fill="x", pady=3)

        self._pause_btn = tk.Button(
            btn_frame, text="Pause", command=self._on_pause, **BTN_STYLE,
        )

        self._finish_btn = tk.Button(
            btn_frame, text="Finish", command=self._on_finish, **BTN_DANGER,
        )
        self._finish_btn.pack(fill="x", pady=3)

        # Status
        self._status_msg = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=ACCENT_GREEN,
        )
        self._status_msg.pack(pady=10)

    def start_collection(self):
        """Initialize and start dual collection"""
        pair_name = self._result.pair_name
        mode = self._result.mode

        # Create output dirs
        self._left_dir = os.path.join(self._pair_dir, "left")
        self._right_dir = os.path.join(self._pair_dir, "right")
        os.makedirs(self._left_dir, exist_ok=True)
        os.makedirs(self._right_dir, exist_ok=True)

        self._saved_left = []
        self._saved_right = []
        self._pair_count = 0
        self._countdown_start = None
        self._cooldown_until = 0.0
        self._auto_paused = False
        self._left_detection = None
        self._right_detection = None

        self._pair_label.configure(text=pair_name)
        self._mode_label.configure(text=f"Mode: {mode.upper()}")
        self._update_info()

        # Show/hide pause button
        if mode == "auto":
            self._pause_btn.pack(fill="x", pady=3, before=self._finish_btn)
        else:
            self._pause_btn.pack_forget()

        # Create readers
        self._reader_l = CameraReader(self._result.cam_left_index)
        self._reader_l.start()
        self._reader_r = CameraReader(self._result.cam_right_index)
        self._reader_r.start()

        # Create feed widgets
        if self._feed_l:
            self._feed_l.stop_feed()
            self._feed_l.destroy()
        if self._feed_r:
            self._feed_r.stop_feed()
            self._feed_r.destroy()

        feed_size = (420, 315)

        self._feed_l = CameraFeedWidget(
            self._feed_frame_l, self._reader_l, feed_size,
            overlay_fn=self._overlay_left,
        )
        self._feed_l.pack()
        self._feed_l.start_feed()

        self._feed_r = CameraFeedWidget(
            self._feed_frame_r, self._reader_r, feed_size,
            overlay_fn=self._overlay_right,
        )
        self._feed_r.pack()
        self._feed_r.start_feed()

        # Start auto timer
        if mode == "auto":
            self._auto_tick()

        # Start detection info updater
        self._update_detection_info()

    def stop_collection(self):
        if self._auto_timer_id:
            self.after_cancel(self._auto_timer_id)
            self._auto_timer_id = None
        if self._feed_l:
            self._feed_l.stop_feed()
        if self._feed_r:
            self._feed_r.stop_feed()
        if self._reader_l:
            self._reader_l.stop()
            self._reader_l = None
        if self._reader_r:
            self._reader_r.stop()
            self._reader_r = None

    def _overlay_left(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_charuco(gray, self._board, self._dictionary, refine_subpix=False)
        self._left_detection = detection
        display = frame.copy()
        if detection.aruco_corners or detection.charuco_corners is not None:
            display = draw_detection_overlay(display, detection)

        # Draw countdown if active
        if (self._result.mode == "auto" and not self._auto_paused
                and self._countdown_start is not None):
            now = time.time()
            remaining = COUNTDOWN_SECONDS - (now - self._countdown_start)
            if remaining > 0:
                self._draw_countdown(display, remaining)

        # Label
        cv2.putText(display, "LEFT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return display

    def _overlay_right(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_charuco(gray, self._board, self._dictionary, refine_subpix=False)
        self._right_detection = detection
        display = frame.copy()
        if detection.aruco_corners or detection.charuco_corners is not None:
            display = draw_detection_overlay(display, detection)

        # Draw countdown if active
        if (self._result.mode == "auto" and not self._auto_paused
                and self._countdown_start is not None):
            now = time.time()
            remaining = COUNTDOWN_SECONDS - (now - self._countdown_start)
            if remaining > 0:
                self._draw_countdown(display, remaining)

        # Label
        cv2.putText(display, "RIGHT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return display

    def _draw_countdown(self, display: np.ndarray, remaining: float):
        h, w = display.shape[:2]
        text = str(int(remaining) + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 4.0
        thickness = 6
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        overlay = display.copy()
        cv2.circle(overlay, (w // 2, h // 2), 80, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, text, (x, y), font, scale, (0, 255, 255), thickness)

    def _both_detected(self) -> bool:
        l = self._left_detection
        r = self._right_detection
        return (l is not None and l.success and l.num_corners >= MIN_CORNERS_DETECT
                and r is not None and r.success and r.num_corners >= MIN_CORNERS_DETECT)

    def _auto_tick(self):
        if self._result.mode != "auto" or self._auto_paused:
            self._auto_timer_id = self.after(100, self._auto_tick)
            return

        now = time.time()

        if now < self._cooldown_until:
            pass
        elif self._both_detected():
            if self._countdown_start is None:
                self._countdown_start = now

            elapsed = now - self._countdown_start
            if elapsed >= COUNTDOWN_SECONDS:
                self._do_capture()
                self._countdown_start = None
                self._cooldown_until = now + COOLDOWN_SECONDS
        else:
            self._countdown_start = None

        self._auto_timer_id = self.after(100, self._auto_tick)

    def _on_capture(self):
        if not self._both_detected():
            self._status_msg.configure(text="Both cameras must detect board!", fg=ACCENT_RED)
            return
        self._do_capture()

    def _do_capture(self):
        if self._reader_l is None or self._reader_l.frame is None:
            return
        if self._reader_r is None or self._reader_r.frame is None:
            return

        frame_l = self._reader_l.frame.copy()
        frame_r = self._reader_r.frame.copy()

        idx = self._pair_count
        left_path = os.path.join(self._left_dir, f"pair_{idx:03d}.png")
        right_path = os.path.join(self._right_dir, f"pair_{idx:03d}.png")

        cv2.imwrite(left_path, frame_l)
        cv2.imwrite(right_path, frame_r)

        self._saved_left.append(left_path)
        self._saved_right.append(right_path)
        self._pair_count += 1

        self._update_info()
        self._status_msg.configure(
            text=f"Captured pair {self._pair_count}!", fg=ACCENT_GREEN,
        )

    def _on_delete_last(self):
        if self._pair_count == 0:
            return
        left_path = self._saved_left.pop()
        right_path = self._saved_right.pop()
        self._pair_count -= 1
        for p in [left_path, right_path]:
            try:
                os.remove(p)
            except OSError:
                pass
        self._update_info()
        self._status_msg.configure(text=f"Deleted pair {self._pair_count + 1}", fg=ACCENT_YELLOW)

    def _on_pause(self):
        self._auto_paused = not self._auto_paused
        self._countdown_start = None
        if self._auto_paused:
            self._pause_btn.configure(text="Resume")
            self._mode_label.configure(text="Mode: AUTO (PAUSED)")
        else:
            self._pause_btn.configure(text="Pause")
            self._mode_label.configure(text="Mode: AUTO")

    def _on_finish(self):
        n = self._pair_count
        if n < MIN_PAIRS:
            if not messagebox.askyesno(
                "Confirm",
                f"Only {n}/{MIN_PAIRS} pairs collected!\n"
                "Calibration may be poor or fail.\n\nQuit anyway?",
            ):
                return

        self.stop_collection()
        self._result.left_paths = list(self._saved_left)
        self._result.right_paths = list(self._saved_right)
        self.winfo_toplevel().event_generate("<<CollectionDone>>")

    def _update_info(self):
        n = self._pair_count
        self._pairs_label.configure(text=f"Pairs: {n}/{TARGET_PAIRS}")
        self._progress["value"] = min(n, TARGET_PAIRS)

        if n >= TARGET_PAIRS:
            self._status_msg.configure(text="Target reached! Press Finish.", fg=ACCENT_GREEN)
        elif n >= MIN_PAIRS:
            self._status_msg.configure(
                text=f"Min reached ({MIN_PAIRS}). Continue or Finish.",
                fg=ACCENT_YELLOW,
            )

    def _update_detection_info(self):
        """Periodically update detection labels"""
        l = self._left_detection
        r = self._right_detection

        l_pts = l.num_corners if l else 0
        r_pts = r.num_corners if r else 0
        l_ok = l is not None and l.success and l_pts >= MIN_CORNERS_DETECT
        r_ok = r is not None and r.success and r_pts >= MIN_CORNERS_DETECT

        l_sym = "\u2713" if l_ok else "\u2717"
        r_sym = "\u2713" if r_ok else "\u2717"
        both_sym = "\u2713" if (l_ok and r_ok) else "\u2717"

        self._left_det_label.configure(
            text=f"Left:  {l_pts} pts {l_sym}",
            fg=ACCENT_GREEN if l_ok else ACCENT_RED,
        )
        self._right_det_label.configure(
            text=f"Right: {r_pts} pts {r_sym}",
            fg=ACCENT_GREEN if r_ok else ACCENT_RED,
        )
        self._both_label.configure(
            text=f"Both OK: {both_sym}",
            fg=ACCENT_GREEN if (l_ok and r_ok) else ACCENT_RED,
        )

        if self._reader_l is not None:
            self.after(200, self._update_detection_info)


# ── 畫面 4：驗收 ──

class StereoValidationScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult,
                 board, dictionary, max_rms: float, reports_dir: str,
                 intrinsics_path: str):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._board = board
        self._dictionary = dictionary
        self._max_rms = max_rms
        self._reports_dir = reports_dir
        self._intrinsics_path = intrinsics_path

        self._sample_idx = 0
        self._rect_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._epipolar_errors: list[float] = []
        self._bar_chart: np.ndarray | None = None
        self._passed = False

        # Photo references
        self._left_photo = None
        self._right_photo = None
        self._chart_photo = None

        self._build()

    def _build(self):
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: images + chart
        left = tk.Frame(main, bg=DARK_BG)
        left.pack(side="left", fill="both", expand=True)

        # Top: rectified pair
        img_row = tk.Frame(left, bg=DARK_BG)
        img_row.pack(pady=5)

        self._left_label = tk.Label(img_row, bg=DARK_BG)
        self._left_label.pack(side="left", padx=2)

        self._right_label = tk.Label(img_row, bg=DARK_BG)
        self._right_label.pack(side="left", padx=2)

        # Chart
        self._chart_label = tk.Label(left, bg=DARK_BG)
        self._chart_label.pack(pady=5)

        # Right: info panel
        panel = tk.Frame(main, bg=PANEL_BG, width=280)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._pair_val_label = tk.Label(
            panel, text="", font=FONT_TITLE, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._pair_val_label.pack(pady=(15, 5))

        self._rms_label = tk.Label(
            panel, text="", font=FONT_BODY, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._rms_label.pack()

        self._pass_label = tk.Label(
            panel, text="", font=("Helvetica", 14, "bold"), bg=PANEL_BG,
        )
        self._pass_label.pack(pady=5)

        self._threshold_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._threshold_label.pack()

        self._baseline_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._baseline_label.pack()

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # R matrix
        tk.Label(panel, text="R (rotation):", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w", padx=10)
        self._r_label = tk.Label(
            panel, text="", font=FONT_MONO, bg=PANEL_BG, fg=TEXT_SECONDARY,
            justify="left",
        )
        self._r_label.pack(anchor="w", padx=10)

        # T vector
        tk.Label(panel, text="T (translation):", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w", padx=10, pady=(5, 0))
        self._t_label = tk.Label(
            panel, text="", font=FONT_MONO, bg=PANEL_BG, fg=TEXT_SECONDARY,
            justify="left",
        )
        self._t_label.pack(anchor="w", padx=10)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Image nav
        self._image_idx_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._image_idx_label.pack()

        nav_frame = tk.Frame(panel, bg=PANEL_BG)
        nav_frame.pack(fill="x", padx=15, pady=5)

        tk.Button(
            nav_frame, text="< Prev", command=self._prev, **BTN_STYLE,
        ).pack(side="left", padx=2)
        tk.Button(
            nav_frame, text="Next >", command=self._next, **BTN_STYLE,
        ).pack(side="right", padx=2)

        # Action buttons
        action_frame = tk.Frame(panel, bg=PANEL_BG)
        action_frame.pack(fill="x", padx=15, pady=10)

        tk.Button(
            action_frame, text="Accept", command=self._on_accept, **BTN_ACCENT,
        ).pack(fill="x", pady=3)
        tk.Button(
            action_frame, text="Redo", command=self._on_redo, **BTN_DANGER,
        ).pack(fill="x", pady=3)

    def load_results(self, stereo_result: StereoCalibResult, pair_dir: str):
        """Load stereo calibration results and display"""
        self._result.stereo_result = stereo_result
        pair_name = stereo_result.pair_name
        R, T = stereo_result.R, stereo_result.T

        # Load intrinsics
        calib_l = load_calib_result(stereo_result.cam_left, self._intrinsics_path)
        calib_r = load_calib_result(stereo_result.cam_right, self._intrinsics_path)
        if calib_l is None or calib_r is None:
            messagebox.showwarning("Warning", "Cannot load intrinsics for rectification!")
            self._result.accepted = False
            self.winfo_toplevel().event_generate("<<ValidationDone>>")
            return

        K_l, D_l = calib_l.K, calib_l.D
        K_r, D_r = calib_r.K, calib_r.D
        image_size = calib_l.image_size

        # Compute rectification maps
        map1_l, map2_l, map1_r, map2_r, Q = compute_stereo_rectification(
            K_l, D_l, K_r, D_r, R, T, image_size,
        )

        # Load image pairs and rectify
        left_dir = os.path.join(pair_dir, "left")
        right_dir = os.path.join(pair_dir, "right")
        left_files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
        right_files = sorted([f for f in os.listdir(right_dir) if f.endswith(".png")])
        n_pairs = min(len(left_files), len(right_files))

        self._rect_pairs = []
        for i in range(n_pairs):
            left_img = cv2.imread(os.path.join(left_dir, left_files[i]))
            right_img = cv2.imread(os.path.join(right_dir, right_files[i]))
            if left_img is None or right_img is None:
                continue
            rect_l = cv2.remap(left_img, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(right_img, map1_r, map2_r, cv2.INTER_LINEAR)
            self._rect_pairs.append((rect_l, rect_r))

        # Compute epipolar errors
        epi_results = compute_epipolar_error(
            pair_dir, self._board, self._dictionary,
            K_l, D_l, K_r, D_r, stereo_result.F,
        )
        self._epipolar_errors = [e for _, _, e in epi_results]

        self._passed = stereo_result.rms <= self._max_rms
        self._sample_idx = 0

        # Draw bar chart
        self._bar_chart = self._draw_error_chart(
            self._epipolar_errors, self._max_rms, pair_name,
        )

        # Save report
        os.makedirs(self._reports_dir, exist_ok=True)
        report_path = os.path.join(self._reports_dir, f"{pair_name}_epipolar_errors.png")
        cv2.imwrite(report_path, self._bar_chart)

        # Update info panel
        self._pair_val_label.configure(text=pair_name)
        self._rms_label.configure(text=f"RMS: {stereo_result.rms:.4f} px")
        self._pass_label.configure(
            text="PASS" if self._passed else "FAIL",
            fg=ACCENT_GREEN if self._passed else ACCENT_RED,
        )
        self._threshold_label.configure(text=f"Threshold: {self._max_rms:.2f} px")

        baseline = float(np.linalg.norm(T))
        self._baseline_label.configure(text=f"Baseline: {baseline:.4f} m")

        # R matrix
        r_text = "\n".join(" ".join(f"{v:8.4f}" for v in row) for row in R)
        self._r_label.configure(text=r_text)

        # T vector
        t_text = " ".join(f"{v:.4f}" for v in T.flatten())
        self._t_label.configure(text=t_text)

        # Chart
        self._chart_photo = cv_to_photoimage(self._bar_chart, (700, 200))
        self._chart_label.configure(image=self._chart_photo)

        # Show first rectified pair
        if self._rect_pairs:
            self._show_pair(0)

    def _show_pair(self, idx: int):
        if not self._rect_pairs:
            return
        idx = idx % len(self._rect_pairs)
        self._sample_idx = idx

        rect_l, rect_r = self._rect_pairs[idx]
        display_size = (380, 285)

        # Draw epipolar lines on rectified images
        left_disp = self._draw_epipolar_lines(rect_l.copy())
        right_disp = self._draw_epipolar_lines(rect_r.copy())

        cv2.putText(left_disp, "RECTIFIED LEFT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(right_disp, "RECTIFIED RIGHT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self._left_photo = cv_to_photoimage(left_disp, display_size)
        self._left_label.configure(image=self._left_photo)

        self._right_photo = cv_to_photoimage(right_disp, display_size)
        self._right_label.configure(image=self._right_photo)

        self._image_idx_label.configure(
            text=f"Pair: {idx + 1}/{len(self._rect_pairs)}",
        )

    def _draw_epipolar_lines(self, img: np.ndarray) -> np.ndarray:
        """Draw horizontal green lines every 30 pixels"""
        h, w = img.shape[:2]
        for y in range(0, h, 30):
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
        return img

    def _draw_error_chart(self, errors: list[float], threshold: float,
                          pair_name: str, width: int = 750, height: int = 220) -> np.ndarray:
        """Draw per-pair epipolar error bar chart"""
        chart = np.zeros((height, width, 3), dtype=np.uint8) + 30
        n = len(errors)
        if n == 0:
            return chart

        margin_left, margin_right = 60, 20
        margin_top, margin_bottom = 30, 35
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        max_err = max(max(errors), threshold * 1.2)
        bar_w = max(2, plot_w // n - 2)

        # Threshold line
        thr_y = margin_top + int((1 - threshold / max_err) * plot_h)
        cv2.line(chart, (margin_left, thr_y), (width - margin_right, thr_y),
                 (0, 0, 180), 1)
        cv2.putText(chart, f"{threshold:.2f}", (5, thr_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)

        # Bars
        for i, err in enumerate(errors):
            bar_h = int((err / max_err) * plot_h)
            x = margin_left + i * (plot_w // n)
            y_top = margin_top + plot_h - bar_h
            y_bot = margin_top + plot_h
            color = (0, 200, 0) if err <= threshold else (0, 0, 220)
            cv2.rectangle(chart, (x, y_top), (x + bar_w, y_bot), color, -1)

        # Title
        cv2.putText(chart, f"Epipolar error per pair ({pair_name})",
                    (margin_left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        cv2.putText(chart, "Pair #", (width // 2 - 20, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return chart

    def _prev(self):
        self._show_pair(self._sample_idx - 1)

    def _next(self):
        self._show_pair(self._sample_idx + 1)

    def _on_accept(self):
        self._result.accepted = True
        self.winfo_toplevel().event_generate("<<ValidationDone>>")

    def _on_redo(self):
        self._result.accepted = False
        self.winfo_toplevel().event_generate("<<ValidationDone>>")


# ── 主流程 ──

def run_extrinsics_gui_tk(
    cameras_cfg: dict,
    charuco_cfg: dict,
    max_rms: float,
    intrinsics_path: str,
    extrinsics_path: str,
    pair_base_dir: str,
    reports_dir: str,
    board,
    dictionary,
    save_extrinsics_fn,
    get_calibrated_pairs_fn,
) -> None:
    """啟動 tkinter 版外參標定 GUI 主流程"""

    root = tk.Tk()
    root.title("Phase 2 — Extrinsics Calibration")
    root.configure(bg=DARK_BG)
    root.geometry("1200x700")

    manager = ScreenManager(root)
    result = GUIResult()

    # Pre-create screens
    pair_screen = PairSelectionScreen(
        root, manager, cameras_cfg,
        intrinsics_path, extrinsics_path, result,
    )
    mode_screen = ModeSelectionScreen(root, manager, result)

    collection_screen: StereoCollectionScreen | None = None
    validation_screen: StereoValidationScreen | None = None

    # ── Flow control ──

    def show_pair_selection():
        result.cam_left = None
        result.cam_right = None
        result.pair_name = None
        result.mode = None
        result.quit = False
        pair_screen.refresh()
        manager.show(pair_screen)

    def on_pair_selected(event=None):
        if result.quit:
            root.destroy()
            return
        if result.pair_name is None:
            return
        manager.show(mode_screen)

    def on_mode_selected(event=None):
        if result.mode is None:
            show_pair_selection()
            return
        # Start stereo collection
        nonlocal collection_screen
        pair_dir = os.path.join(pair_base_dir, result.pair_name)
        collection_screen = StereoCollectionScreen(
            root, manager, result, board, dictionary, pair_dir,
        )
        manager.show(collection_screen)
        collection_screen.start_collection()

    def on_collection_done(event=None):
        nonlocal collection_screen
        if collection_screen:
            collection_screen.stop_collection()

        if not result.left_paths:
            show_pair_selection()
            return

        # Show calibrating message
        cal_frame = tk.Frame(root, bg=DARK_BG)
        tk.Label(
            cal_frame, text="Calibrating stereo pair...",
            font=("Helvetica", 20, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(expand=True)
        manager.show(cal_frame)

        # Load intrinsics
        calib_l = load_calib_result(result.cam_left, intrinsics_path)
        calib_r = load_calib_result(result.cam_right, intrinsics_path)
        if calib_l is None or calib_r is None:
            messagebox.showerror("Error", "Cannot load intrinsics!")
            show_pair_selection()
            return

        pair_dir = os.path.join(pair_base_dir, result.pair_name)

        def _calibrate():
            return calibrate_stereo(
                pair_dir=pair_dir,
                K_l=calib_l.K, D_l=calib_l.D,
                K_r=calib_r.K, D_r=calib_r.D,
                image_size=calib_l.image_size,
                board=board, dictionary=dictionary,
                pair_name=result.pair_name,
                cam_left=result.cam_left,
                cam_right=result.cam_right,
            )

        def _on_calibrated(stereo_result):
            nonlocal validation_screen
            if stereo_result is None:
                messagebox.showwarning("Calibration Failed",
                                       "Not enough valid pairs for stereo calibration.")
                show_pair_selection()
                return

            result.stereo_result = stereo_result
            validation_screen = StereoValidationScreen(
                root, manager, result, board, dictionary,
                max_rms, reports_dir, intrinsics_path,
            )
            manager.show(validation_screen)
            validation_screen.load_results(stereo_result, pair_dir)

        run_in_thread(_calibrate, _on_calibrated, root)

    def on_validation_done(event=None):
        if result.accepted and result.stereo_result:
            sr = result.stereo_result
            save_extrinsics_fn({
                "pair_name": sr.pair_name,
                "cam_left": sr.cam_left,
                "cam_right": sr.cam_right,
                "R": sr.R.tolist(),
                "T": sr.T.tolist(),
                "rms": sr.rms,
                "num_pairs_used": sr.num_pairs_used,
            })
        show_pair_selection()

    # Bind events
    pair_screen.bind("<<PairSelected>>", on_pair_selected)
    mode_screen.bind("<<ModeSelected>>", on_mode_selected)
    root.bind("<<CollectionDone>>", on_collection_done)
    root.bind("<<ValidationDone>>", on_validation_done)

    # Start
    show_pair_selection()
    root.mainloop()
