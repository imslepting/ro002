"""Phase 1 — 內參標定 tkinter GUI

5 個畫面 + 畫面管理器：
  1. CameraSelectionScreen — 相機選擇
  2. ResolutionSelectionScreen — 分辨率選擇
  3. ModeSelectionScreen — 模式選擇
  4. CollectionScreen — 採集（最複雜）
  5. ValidationScreen — 驗收
"""

from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

from shared.camera_manager import CameraReader
from shared.types import CalibResult
from shared.tk_utils import (
    DARK_BG, PANEL_BG, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    FONT_TITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    BTN_STYLE, BTN_ACCENT, BTN_DANGER,
    cv_to_photoimage, CameraFeedWidget, probe_resolutions, run_in_thread,
)
from phase1_intrinsics.src.charuco_detector import detect_charuco, draw_detection_overlay
from phase1_intrinsics.src.calibrator import calibrate_camera, compute_per_frame_errors
from phase1_intrinsics.src.image_collector import (
    _save_frame, _estimate_coverage, TARGET_FRAMES, MIN_FRAMES,
    COUNTDOWN_SECONDS, COOLDOWN_SECONDS, MIN_CORNERS_DETECT,
)
from phase1_intrinsics.src.validator import _draw_bar_chart


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
        self.cam_name: str | None = None
        self.cam_index: int = 0
        self.resolution: tuple[int, int] | None = None
        self.mode: str | None = None
        self.image_paths: list[str] = []
        self.calib_result: CalibResult | None = None
        self.accepted: bool = False
        self.quit: bool = False
        self.generate_board: bool = False


# ── 畫面 1：相機選擇 ──

class CameraSelectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, camera_configs: dict,
                 calibrated: set[str], result: GUIResult, intrinsics_path: str):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._camera_configs = camera_configs
        self._calibrated = calibrated
        self._result = result
        self._intrinsics_path = intrinsics_path
        self._preview_reader: CameraReader | None = None
        self._preview_win: tk.Toplevel | None = None

        self._build()

    def _build(self):
        tk.Label(
            self, text="Phase 1 — Intrinsics Calibration",
            font=("Helvetica", 20, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(30, 5))

        tk.Label(
            self, text="Select camera to calibrate:",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_SECONDARY,
        ).pack(pady=(5, 20))

        cam_names = sorted(self._camera_configs.keys())
        list_frame = tk.Frame(self, bg=DARK_BG)
        list_frame.pack(pady=5)

        for name in cam_names:
            role = self._camera_configs[name].get("role", "")
            is_cal = name in self._calibrated
            self._add_camera_row(list_frame, name, role, is_cal)

        # Bottom buttons
        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=30)

        tk.Button(
            btn_frame, text="Generate Board PDF",
            command=self._on_generate_board, **BTN_STYLE,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Quit",
            command=self._on_quit, **BTN_DANGER,
        ).pack(side="left", padx=10)

    def _add_camera_row(self, parent, name, role, is_cal):
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill="x", padx=20, pady=4)

        # Dot
        dot_canvas = tk.Canvas(row, width=20, height=20, bg=DARK_BG,
                               highlightthickness=0)
        dot_canvas.pack(side="left", padx=(0, 8))
        dot_color = ACCENT_GREEN if is_cal else TEXT_DIM
        if is_cal:
            dot_canvas.create_oval(4, 4, 16, 16, fill=dot_color, outline="")
        else:
            dot_canvas.create_oval(4, 4, 16, 16, outline=dot_color, width=2)

        # Name + role
        tk.Label(
            row, text=f"{name} ({role})",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_PRIMARY, width=20, anchor="w",
        ).pack(side="left")

        # Status
        status_text = "CALIBRATED" if is_cal else "NOT CALIBRATED"
        status_color = ACCENT_GREEN if is_cal else TEXT_DIM
        tk.Label(
            row, text=status_text,
            font=FONT_SMALL, bg=DARK_BG, fg=status_color, width=16, anchor="w",
        ).pack(side="left")

        # Select button
        tk.Button(
            row, text="Select",
            command=lambda n=name: self._on_select(n),
            **BTN_STYLE,
        ).pack(side="right", padx=5)

    def _on_select(self, cam_name: str):
        """Open live preview confirmation dialog"""
        cam_index = self._camera_configs[cam_name]["index"]
        self._show_preview(cam_name, cam_index)

    def _show_preview(self, cam_name: str, cam_index: int):
        """Show preview Toplevel with CameraFeedWidget + Confirm/Back"""
        if self._preview_win is not None:
            return

        self._preview_reader = CameraReader(cam_index)
        self._preview_reader.start()

        win = tk.Toplevel(self, bg=DARK_BG)
        win.title(f"Preview — {cam_name}")
        win.geometry("700x620")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_preview(win))
        self._preview_win = win

        tk.Label(
            win, text=f"Preview: {cam_name}",
            font=FONT_TITLE, bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(10, 5))

        feed = CameraFeedWidget(win, self._preview_reader, (640, 480))
        feed.pack(pady=5)
        feed.start_feed()
        self._preview_feed = feed

        btn_frame = tk.Frame(win, bg=DARK_BG)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Confirm",
            command=lambda: self._confirm_preview(cam_name, win),
            **BTN_ACCENT,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Back",
            command=lambda: self._close_preview(win),
            **BTN_STYLE,
        ).pack(side="left", padx=10)

    def _confirm_preview(self, cam_name: str, win: tk.Toplevel):
        self._result.cam_name = cam_name
        self._result.cam_index = self._camera_configs[cam_name]["index"]
        self._close_preview(win)
        self._result.quit = False
        print(f"[DEBUG] confirm_preview done, cam={cam_name} idx={self._result.cam_index}")
        self.event_generate("<<CameraSelected>>")

    def _close_preview(self, win: tk.Toplevel):
        print("[DEBUG] close_preview: stopping feed...")
        if hasattr(self, "_preview_feed"):
            self._preview_feed.stop_feed()
        print("[DEBUG] close_preview: stopping reader...")
        if self._preview_reader:
            self._preview_reader.stop()
            self._preview_reader = None
        print("[DEBUG] close_preview: reader stopped, destroying window")
        win.destroy()
        self._preview_win = None

    def _on_generate_board(self):
        self._result.generate_board = True
        self.event_generate("<<CameraSelected>>")

    def _on_quit(self):
        self._result.quit = True
        self.event_generate("<<CameraSelected>>")

    def refresh_calibrated(self, calibrated: set[str]):
        """Refresh the calibrated status"""
        self._calibrated = calibrated
        # Rebuild
        for w in self.winfo_children():
            w.destroy()
        self._build()


# ── 畫面 2：分辨率選擇 ──

class ResolutionSelectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._resolutions: list[tuple[int, int]] = []
        self._selected = tk.IntVar(value=0)
        self._list_frame: tk.Frame | None = None
        self._confirm_btn: tk.Button | None = None

        self._build()

    def _build(self):
        self._title_label = tk.Label(
            self, text="Select resolution",
            font=("Helvetica", 18, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        )
        self._title_label.pack(pady=(30, 10))

        self._status_label = tk.Label(
            self, text="Probing resolutions...",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_SECONDARY,
        )
        self._status_label.pack(pady=5)

        self._list_frame = tk.Frame(self, bg=DARK_BG)
        self._list_frame.pack(pady=15)

        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=20)

        self._confirm_btn = tk.Button(
            btn_frame, text="Confirm", state="disabled",
            command=self._on_confirm, **BTN_ACCENT,
        )
        self._confirm_btn.pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Back",
            command=self._on_back, **BTN_STYLE,
        ).pack(side="left", padx=10)

    def start_probe(self, cam_name: str, cam_index: int):
        """Start probing resolutions in background"""
        self._title_label.configure(text=f"Select resolution for {cam_name}:")
        self._status_label.configure(text="Probing resolutions...")
        self._confirm_btn.configure(state="disabled")

        # Clear old radio buttons
        for w in self._list_frame.winfo_children():
            w.destroy()

        root = self.winfo_toplevel()
        run_in_thread(
            lambda: probe_resolutions(cam_index),
            self._on_probed, root,
        )

    def _on_probed(self, res_list: list[tuple[int, int]]):
        self._resolutions = res_list
        if not res_list:
            self._status_label.configure(text="No resolutions detected. Using camera default.")
            return

        self._status_label.configure(text="")
        self._selected.set(0)

        for i, (w, h) in enumerate(res_list):
            tk.Radiobutton(
                self._list_frame, text=f"{w} x {h}",
                variable=self._selected, value=i,
                font=FONT_BODY, bg=DARK_BG, fg=TEXT_PRIMARY,
                selectcolor=PANEL_BG, activebackground=DARK_BG,
                activeforeground=TEXT_PRIMARY,
            ).pack(anchor="w", padx=40, pady=3)

        self._confirm_btn.configure(state="normal")

    def _on_confirm(self):
        if not self._resolutions:
            self._result.resolution = None
        else:
            idx = self._selected.get()
            self._result.resolution = self._resolutions[idx]
        self.event_generate("<<ResolutionSelected>>")

    def _on_back(self):
        self._result.resolution = None
        self._result.cam_name = None  # Signal to go back
        self.event_generate("<<ResolutionSelected>>")


# ── 畫面 3：模式選擇 ──

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

        auto_btn = tk.Button(
            btn_frame, text="Auto — countdown after detection",
            command=lambda: self._select("auto"),
            width=40, height=2, **BTN_ACCENT,
        )
        auto_btn.pack(pady=10)

        manual_btn = tk.Button(
            btn_frame, text="Manual — press button to capture",
            command=lambda: self._select("manual"),
            width=40, height=2, **BTN_STYLE,
        )
        manual_btn.pack(pady=10)

        tk.Button(
            self, text="Back", command=self._on_back, **BTN_STYLE,
        ).pack(pady=20)

    def _select(self, mode: str):
        self._result.mode = mode
        self.event_generate("<<ModeSelected>>")

    def _on_back(self):
        self._result.mode = None
        self.event_generate("<<ModeSelected>>")


# ── 畫面 4：採集 ──

class CollectionScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult,
                 board, dictionary, output_dir: str):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._board = board
        self._dictionary = dictionary
        self._output_dir = output_dir

        self._reader: CameraReader | None = None
        self._feed: CameraFeedWidget | None = None
        self._saved_paths: list[str] = []
        self._all_detected_corners: list[np.ndarray] = []

        # Auto mode state
        self._countdown_start: float | None = None
        self._cooldown_until: float = 0.0
        self._auto_paused = False
        self._auto_timer_id: str | None = None

        # Last detection (updated from overlay)
        self._last_detection = None

        self._build()

    def _build(self):
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: camera feed
        self._feed_frame = tk.Frame(main, bg=DARK_BG)
        self._feed_frame.pack(side="left", padx=(0, 5))

        # Right: info panel
        panel = tk.Frame(main, bg=PANEL_BG, width=240)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._cam_label = tk.Label(
            panel, text="", font=FONT_TITLE, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._cam_label.pack(pady=(15, 5))

        self._mode_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._mode_label.pack()

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        self._frames_label = tk.Label(
            panel, text="Frames: 0/30", font=FONT_BODY, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._frames_label.pack(pady=(5, 2))

        self._progress = ttk.Progressbar(panel, length=200, mode="determinate",
                                         maximum=TARGET_FRAMES)
        self._progress.pack(padx=15, pady=5)

        self._corners_label = tk.Label(
            panel, text="Corners: 0", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._corners_label.pack()

        self._coverage_label = tk.Label(
            panel, text="Coverage: 0%", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._coverage_label.pack()

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

        # Status message
        self._status_msg = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=ACCENT_GREEN,
        )
        self._status_msg.pack(pady=10)

    def start_collection(self):
        """Initialize and start the collection process"""
        cam_name = self._result.cam_name
        cam_index = self._result.cam_index
        resolution = self._result.resolution
        mode = self._result.mode

        os.makedirs(self._output_dir, exist_ok=True)
        self._saved_paths = []
        self._all_detected_corners = []
        self._countdown_start = None
        self._cooldown_until = 0.0
        self._auto_paused = False
        self._last_detection = None

        self._cam_label.configure(text=cam_name)
        mode_text = f"Mode: {mode.upper()}"
        self._mode_label.configure(text=mode_text)
        self._update_info()

        # Show/hide pause button for auto mode
        if mode == "auto":
            self._pause_btn.pack(fill="x", pady=3, before=self._finish_btn)
        else:
            self._pause_btn.pack_forget()

        # Create reader
        print(f"[DEBUG] start_collection: creating CameraReader idx={cam_index} res={resolution}")
        self._reader = CameraReader(cam_index, resolution=resolution)
        print(f"[DEBUG] start_collection: cap.isOpened={self._reader.cap.isOpened()}")
        self._reader.start()

        # Create feed widget
        if self._feed:
            self._feed.stop_feed()
            self._feed.destroy()

        self._feed = CameraFeedWidget(
            self._feed_frame, self._reader, (640, 480),
            overlay_fn=self._overlay,
        )
        self._feed.pack()
        self._feed.start_feed()

        # Start auto mode timer
        if mode == "auto":
            self._auto_tick()

    def stop_collection(self):
        if self._auto_timer_id:
            self.after_cancel(self._auto_timer_id)
            self._auto_timer_id = None
        if self._feed:
            self._feed.stop_feed()
        if self._reader:
            self._reader.stop()
            self._reader = None

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        """Overlay charuco detection on the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_charuco(gray, self._board, self._dictionary, refine_subpix=False)
        self._last_detection = detection

        display = frame.copy()
        if detection.aruco_corners or detection.charuco_corners is not None:
            display = draw_detection_overlay(display, detection)

        # Draw countdown if active
        if (self._result.mode == "auto" and not self._auto_paused
                and self._countdown_start is not None):
            now = time.time()
            remaining = COUNTDOWN_SECONDS - (now - self._countdown_start)
            if remaining > 0:
                self._draw_countdown_on_frame(display, remaining)

        return display

    def _draw_countdown_on_frame(self, display: np.ndarray, remaining: float):
        h, w = display.shape[:2]
        text = str(int(remaining) + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 5.0
        thickness = 8
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        overlay = display.copy()
        cv2.circle(overlay, (w // 2, h // 2), 100, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, text, (x, y), font, scale, (0, 255, 255), thickness)

    def _auto_tick(self):
        """Periodically check auto capture conditions"""
        if self._result.mode != "auto" or self._auto_paused:
            self._auto_timer_id = self.after(100, self._auto_tick)
            return

        now = time.time()
        detection = self._last_detection

        if now < self._cooldown_until:
            # Cooldown
            pass
        elif (detection is not None and detection.success
              and detection.num_corners >= MIN_CORNERS_DETECT):
            if self._countdown_start is None:
                self._countdown_start = now

            elapsed = now - self._countdown_start
            if elapsed >= COUNTDOWN_SECONDS:
                # Auto capture!
                self._do_capture()
                self._countdown_start = None
                self._cooldown_until = now + COOLDOWN_SECONDS
        else:
            self._countdown_start = None

        self._auto_timer_id = self.after(100, self._auto_tick)

    def _on_capture(self):
        detection = self._last_detection
        if detection is None or not detection.success:
            self._status_msg.configure(text="No valid detection!", fg=ACCENT_RED)
            return
        self._do_capture()

    def _do_capture(self):
        if self._reader is None or self._reader.frame is None:
            return
        frame = self._reader.frame.copy()
        cam_name = self._result.cam_name
        path = _save_frame(frame, cam_name, len(self._saved_paths), self._output_dir)
        self._saved_paths.append(path)

        detection = self._last_detection
        if detection and detection.charuco_corners is not None:
            self._all_detected_corners.append(detection.charuco_corners.copy())

        self._update_info()
        self._status_msg.configure(text=f"Captured frame {len(self._saved_paths)}!", fg=ACCENT_GREEN)

    def _on_delete_last(self):
        if not self._saved_paths:
            return
        removed = self._saved_paths.pop()
        if self._all_detected_corners:
            self._all_detected_corners.pop()
        try:
            os.remove(removed)
        except OSError:
            pass
        self._update_info()
        self._status_msg.configure(text=f"Deleted: {os.path.basename(removed)}", fg=ACCENT_YELLOW)

    def _on_pause(self):
        self._auto_paused = not self._auto_paused
        self._countdown_start = None
        if self._auto_paused:
            self._pause_btn.configure(text="Resume")
            self._mode_label.configure(text=f"Mode: AUTO (PAUSED)")
        else:
            self._pause_btn.configure(text="Pause")
            self._mode_label.configure(text="Mode: AUTO")

    def _on_finish(self):
        n = len(self._saved_paths)
        if n < MIN_FRAMES:
            if not messagebox.askyesno(
                "Confirm",
                f"Only {n}/{MIN_FRAMES} frames collected!\n"
                "Calibration may be poor or fail.\n\nQuit anyway?",
            ):
                return

        self.stop_collection()
        self._result.image_paths = list(self._saved_paths)
        self.winfo_toplevel().event_generate("<<CollectionDone>>")

    def _update_info(self):
        n = len(self._saved_paths)
        self._frames_label.configure(text=f"Frames: {n}/{TARGET_FRAMES}")
        self._progress["value"] = min(n, TARGET_FRAMES)

        detection = self._last_detection
        corners = detection.num_corners if detection else 0
        self._corners_label.configure(text=f"Corners: {corners}")

        if self._reader and self._reader.frame is not None:
            gray_shape = self._reader.frame.shape[:2]
        else:
            gray_shape = (480, 640)
        coverage = _estimate_coverage(self._all_detected_corners, gray_shape)
        self._coverage_label.configure(text=f"Coverage: {coverage:.0f}%")

        if n >= TARGET_FRAMES:
            self._status_msg.configure(text="Target reached! Press Finish.", fg=ACCENT_GREEN)
        elif n >= MIN_FRAMES:
            self._status_msg.configure(text=f"Min reached ({MIN_FRAMES}). Continue or Finish.",
                                       fg=ACCENT_YELLOW)


# ── 畫面 5：驗收 ──

class ValidationScreen(tk.Frame):
    def __init__(self, master, manager: ScreenManager, result: GUIResult,
                 board, dictionary, max_rms: float, reports_dir: str):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._result = result
        self._board = board
        self._dictionary = dictionary
        self._max_rms = max_rms
        self._reports_dir = reports_dir

        self._valid_paths: list[str] = []
        self._errors: list[float] = []
        self._sample_idx = 0
        self._bar_chart: np.ndarray | None = None
        self._passed = False

        # Photo references to prevent GC
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._undist_photo: ImageTk.PhotoImage | None = None
        self._chart_photo: ImageTk.PhotoImage | None = None

        self._build()

    def _build(self):
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: images + chart
        left = tk.Frame(main, bg=DARK_BG)
        left.pack(side="left", fill="both", expand=True)

        # Top: orig + undistorted
        img_row = tk.Frame(left, bg=DARK_BG)
        img_row.pack(pady=5)

        self._orig_label = tk.Label(img_row, bg=DARK_BG)
        self._orig_label.pack(side="left", padx=2)

        self._undist_label = tk.Label(img_row, bg=DARK_BG)
        self._undist_label.pack(side="left", padx=2)

        # Chart
        self._chart_label = tk.Label(left, bg=DARK_BG)
        self._chart_label.pack(pady=5)

        # Right: info panel
        panel = tk.Frame(main, bg=PANEL_BG, width=260)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._cam_val_label = tk.Label(
            panel, text="", font=FONT_TITLE, bg=PANEL_BG, fg=TEXT_PRIMARY,
        )
        self._cam_val_label.pack(pady=(15, 5))

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

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        self._image_idx_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._image_idx_label.pack()

        self._size_label = tk.Label(
            panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._size_label.pack()

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        # K matrix
        tk.Label(panel, text="K (intrinsics):", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w", padx=10)
        self._k_label = tk.Label(
            panel, text="", font=FONT_MONO, bg=PANEL_BG, fg=TEXT_SECONDARY,
            justify="left",
        )
        self._k_label.pack(anchor="w", padx=10)

        tk.Label(panel, text="D (distortion):", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w", padx=10, pady=(8, 0))
        self._d_label = tk.Label(
            panel, text="", font=FONT_MONO, bg=PANEL_BG, fg=TEXT_SECONDARY,
            justify="left",
        )
        self._d_label.pack(anchor="w", padx=10)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=10)

        # Nav buttons
        nav_frame = tk.Frame(panel, bg=PANEL_BG)
        nav_frame.pack(fill="x", padx=15)

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

    def load_results(self, calib_result: CalibResult, image_paths: list[str]):
        """Load calibration results and display"""
        self._result.calib_result = calib_result
        K, D = calib_result.K, calib_result.D
        cam_name = calib_result.cam_name

        # Compute per-frame errors
        per_frame = compute_per_frame_errors(
            image_paths, self._board, self._dictionary, K, D,
        )
        if not per_frame:
            messagebox.showwarning("Warning", "No valid frames for validation!")
            self._result.accepted = False
            self.winfo_toplevel().event_generate("<<ValidationDone>>")
            return

        self._errors = [e for _, e in per_frame]
        self._valid_paths = [p for p, _ in per_frame]
        self._passed = calib_result.rms <= self._max_rms
        self._sample_idx = 0

        # Draw bar chart
        self._bar_chart = _draw_bar_chart(
            self._errors, self._max_rms, cam_name, width=800, height=250,
        )

        # Save error report
        os.makedirs(self._reports_dir, exist_ok=True)
        report_path = os.path.join(self._reports_dir, f"{cam_name}_reprojection_errors.png")
        cv2.imwrite(report_path, self._bar_chart)

        # Update info panel
        self._cam_val_label.configure(text=cam_name)
        self._rms_label.configure(text=f"RMS: {calib_result.rms:.4f} px")
        self._pass_label.configure(
            text="PASS" if self._passed else "FAIL",
            fg=ACCENT_GREEN if self._passed else ACCENT_RED,
        )
        self._threshold_label.configure(text=f"Threshold: {self._max_rms:.2f} px")
        self._size_label.configure(
            text=f"Size: {calib_result.image_size[0]}x{calib_result.image_size[1]}",
        )

        # K matrix
        k_text = "\n".join(" ".join(f"{v:8.1f}" for v in row) for row in K)
        self._k_label.configure(text=k_text)

        # D vector
        d_text = " ".join(f"{v:.4f}" for v in D.flatten()[:5])
        self._d_label.configure(text=d_text)

        # Show chart
        self._chart_photo = cv_to_photoimage(self._bar_chart, (760, 230))
        self._chart_label.configure(image=self._chart_photo)

        # Show first image
        self._show_image(0)

    def _show_image(self, idx: int):
        if not self._valid_paths:
            return
        idx = idx % len(self._valid_paths)
        self._sample_idx = idx

        calib = self._result.calib_result
        K, D = calib.K, calib.D

        img = cv2.imread(self._valid_paths[idx])
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        display_size = (380, 285)

        # Original
        orig = cv2.resize(img, display_size)
        cv2.putText(orig, "ORIGINAL", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._orig_photo = cv_to_photoimage(orig)
        self._orig_label.configure(image=self._orig_photo)

        # Undistorted
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undist = cv2.undistort(img, K, D, None, new_K)
        undist_resized = cv2.resize(undist, display_size)
        cv2.putText(undist_resized, "UNDISTORTED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._undist_photo = cv_to_photoimage(undist_resized)
        self._undist_label.configure(image=self._undist_photo)

        self._image_idx_label.configure(
            text=f"Image: {idx + 1}/{len(self._valid_paths)}",
        )

    def _prev(self):
        self._show_image(self._sample_idx - 1)

    def _next(self):
        self._show_image(self._sample_idx + 1)

    def _on_accept(self):
        self._result.accepted = True
        self.winfo_toplevel().event_generate("<<ValidationDone>>")

    def _on_redo(self):
        self._result.accepted = False
        self.winfo_toplevel().event_generate("<<ValidationDone>>")


# ── 主流程 ──

def run_calibration_gui_tk(
    cameras_cfg: dict,
    charuco_cfg: dict,
    max_rms: float,
    intrinsics_path: str,
    reports_dir: str,
    board,
    dictionary,
    generate_board_fn,
    get_calibrated_fn,
    save_intrinsics_fn,
    root_dir: str,
) -> None:
    """啟動 tkinter 版標定 GUI 主流程"""

    root = tk.Tk()
    root.title("Phase 1 — Intrinsics Calibration")
    root.configure(bg=DARK_BG)
    root.geometry("1100x700")

    manager = ScreenManager(root)
    result = GUIResult()

    # Pre-create screens
    cam_screen = CameraSelectionScreen(
        root, manager, cameras_cfg,
        get_calibrated_fn(), result, intrinsics_path,
    )
    res_screen = ResolutionSelectionScreen(root, manager, result)
    mode_screen = ModeSelectionScreen(root, manager, result)

    raw_base_dir = os.path.join(root_dir, "phase1_intrinsics", "outputs", "raw_images")

    collection_screen: CollectionScreen | None = None
    validation_screen: ValidationScreen | None = None

    # ── Flow control ──

    def show_camera_selection():
        result.cam_name = None
        result.resolution = None
        result.mode = None
        result.generate_board = False
        result.quit = False
        cam_screen.refresh_calibrated(get_calibrated_fn())
        manager.show(cam_screen)

    def on_camera_selected(event=None):
        if result.quit:
            root.destroy()
            return
        if result.generate_board:
            generate_board_fn()
            result.generate_board = False
            return
        if result.cam_name is None:
            return
        # 跳過分辨率選擇，使用默認 640x480
        result.resolution = (640, 480)
        manager.show(mode_screen)

    def on_resolution_selected(event=None):
        if result.cam_name is None:
            show_camera_selection()
            return
        if result.resolution is None:
            result.resolution = (640, 480)
        manager.show(mode_screen)

    def on_mode_selected(event=None):
        if result.mode is None:
            show_camera_selection()
            return
        # Start collection
        nonlocal collection_screen
        output_dir = os.path.join(raw_base_dir, result.cam_name)
        collection_screen = CollectionScreen(
            root, manager, result, board, dictionary, output_dir,
        )
        manager.show(collection_screen)
        collection_screen.start_collection()

    def on_collection_done(event=None):
        nonlocal collection_screen
        if collection_screen:
            collection_screen.stop_collection()

        if not result.image_paths:
            show_camera_selection()
            return

        # Show calibrating message
        cal_frame = tk.Frame(root, bg=DARK_BG)
        tk.Label(
            cal_frame, text="Calibrating...",
            font=("Helvetica", 20, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(expand=True)
        manager.show(cal_frame)

        def _calibrate():
            return calibrate_camera(
                result.image_paths, board, dictionary, result.cam_name,
            )

        def _on_calibrated(calib_result):
            nonlocal validation_screen
            if calib_result is None:
                messagebox.showwarning("Calibration Failed",
                                       "Not enough valid frames for calibration.")
                show_camera_selection()
                return

            result.calib_result = calib_result
            validation_screen = ValidationScreen(
                root, manager, result, board, dictionary, max_rms, reports_dir,
            )
            manager.show(validation_screen)
            validation_screen.load_results(calib_result, result.image_paths)

        run_in_thread(_calibrate, _on_calibrated, root)

    def on_validation_done(event=None):
        if result.accepted and result.calib_result:
            save_intrinsics_fn(result.calib_result)
        show_camera_selection()

    # Bind events
    cam_screen.bind("<<CameraSelected>>", on_camera_selected)
    res_screen.bind("<<ResolutionSelected>>", on_resolution_selected)
    mode_screen.bind("<<ModeSelected>>", on_mode_selected)

    # We need to bind collection events dynamically since the screen is created later
    root.bind("<<CollectionDone>>", on_collection_done)
    root.bind("<<ValidationDone>>", on_validation_done)

    # Start
    show_camera_selection()
    root.mainloop()
