"""Phase 3 — Stereo Depth tkinter GUI

3 個畫面 + 畫面管理器：
  1. SetupScreen — 檢查標定、載入模型
  2. CaptureScreen — 擷取模式 + 即時模式切換
  3. ResultScreen — 擷取結果展示
"""

from __future__ import annotations

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

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
from phase1_intrinsics.src.intrinsics_io import get_calibrated_cameras
from phase2_extrinsics.src.extrinsics_io import get_calibrated_pairs
from phase3_stereo_depth.src.stereo_rectifier import StereoRectifier
from phase3_stereo_depth.src.stereo_inference import StereoInference
from phase3_stereo_depth.src.depth_converter import (
    disparity_to_depth, depth_to_pointcloud, depth_to_pointcloud_fast,
    save_pointcloud_ply,
)
from phase3_stereo_depth.src.depth_utils import (
    colorize_disparity, colorize_depth, draw_rectification_check,
    save_depth, save_disparity_vis, compute_depth_stats,
    measure_distance_3d, draw_measurement_overlay,
)


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


# ── 共享狀態 ──

class AppState:
    """在畫面間傳遞的共享狀態"""
    def __init__(self):
        self.rectifier: StereoRectifier | None = None
        self.inference: StereoInference | None = None
        # 最近一次擷取結果
        self.rect_left: np.ndarray | None = None
        self.rect_right: np.ndarray | None = None
        self.disparity: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.inference_time_ms: float = 0.0
        # 測距
        self.measure_a: tuple[int, int] | None = None  # (u, v) 原始分辨率
        self.measure_b: tuple[int, int] | None = None
        self.measure_dist: float | None = None


# ── 畫面 1：SetupScreen ──

class SetupScreen(tk.Frame):
    def __init__(
        self, master, manager: ScreenManager, state: AppState,
        cfg: dict, intrinsics_path: str, extrinsics_path: str,
        pair_name: str,
    ):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._state = state
        self._cfg = cfg
        self._intrinsics_path = intrinsics_path
        self._extrinsics_path = extrinsics_path
        self._pair_name = pair_name
        self._model_ready = False
        self._calib_ready = False
        self._build()

    def _build(self):
        tk.Label(
            self, text="Phase 3 — Stereo Depth",
            font=("Helvetica", 20, "bold"), bg=DARK_BG, fg=TEXT_PRIMARY,
        ).pack(pady=(30, 5))

        tk.Label(
            self, text="Setup & Model Loading",
            font=FONT_BODY, bg=DARK_BG, fg=TEXT_SECONDARY,
        ).pack(pady=(0, 20))

        # 標定狀態面板
        calib_frame = tk.LabelFrame(
            self, text="Calibration Status", font=FONT_BODY,
            bg=PANEL_BG, fg=TEXT_PRIMARY, padx=20, pady=10,
        )
        calib_frame.pack(padx=40, pady=5, fill="x")

        cam_left, cam_right = self._pair_name.split("_", 1)
        calibrated_cams = get_calibrated_cameras(self._intrinsics_path)
        calibrated_pairs = get_calibrated_pairs(self._extrinsics_path)

        checks = [
            (f"{cam_left} intrinsics", cam_left in calibrated_cams),
            (f"{cam_right} intrinsics", cam_right in calibrated_cams),
            (f"{self._pair_name} extrinsics", self._pair_name in calibrated_pairs),
        ]

        self._calib_ready = all(ok for _, ok in checks)

        for label, ok in checks:
            row = tk.Frame(calib_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=2)
            sym = "\u2713" if ok else "\u2717"
            color = ACCENT_GREEN if ok else ACCENT_RED
            tk.Label(
                row, text=f"  {sym}  {label}",
                font=FONT_BODY, bg=PANEL_BG, fg=color, anchor="w",
            ).pack(side="left")

        # 模型狀態面板
        model_frame = tk.LabelFrame(
            self, text="Model Status", font=FONT_BODY,
            bg=PANEL_BG, fg=TEXT_PRIMARY, padx=20, pady=10,
        )
        model_frame.pack(padx=40, pady=10, fill="x")

        self._model_status = tk.Label(
            model_frame, text="Not loaded", font=FONT_BODY,
            bg=PANEL_BG, fg=TEXT_DIM,
        )
        self._model_status.pack(pady=5)

        self._gpu_label = tk.Label(
            model_frame, text="", font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY,
        )
        self._gpu_label.pack()

        self._load_progress = ttk.Progressbar(
            model_frame, length=400, mode="indeterminate",
        )
        self._load_progress.pack(pady=5)

        # 按鈕列
        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=20)

        self._load_btn = tk.Button(
            btn_frame, text="Load Model", command=self._on_load_model,
            state="normal" if self._calib_ready else "disabled",
            **BTN_ACCENT,
        )
        self._load_btn.pack(side="left", padx=10)

        self._start_btn = tk.Button(
            btn_frame, text="Start", command=self._on_start,
            state="disabled", **BTN_ACCENT,
        )
        self._start_btn.pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Quit", command=self._on_quit, **BTN_DANGER,
        ).pack(side="left", padx=10)

        # GPU info
        self._update_gpu_info()

    def _update_gpu_info(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                self._gpu_label.configure(text=f"GPU: {name} ({mem:.0f} GB)")
            else:
                self._gpu_label.configure(text="GPU: Not available (CPU mode)")
        except ImportError:
            self._gpu_label.configure(text="PyTorch not installed")

    def _on_load_model(self):
        self._load_btn.configure(state="disabled")
        self._model_status.configure(text="Loading...", fg=ACCENT_YELLOW)
        self._load_progress.start(10)

        sd_cfg = self._cfg.get("stereo_depth", {})
        model_dir = sd_cfg.get("model_dir", "external/Fast-FoundationStereo/weights/fast_foundationstereo")
        max_disp = sd_cfg.get("max_disparity", 256)
        valid_iters = sd_cfg.get("valid_iters", 8)
        pad_multiple = sd_cfg.get("pad_multiple", 32)

        # 取得項目根目錄
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir_abs = os.path.join(root_dir, model_dir)

        def _load():
            # 先載入 rectifier
            rectifier = StereoRectifier(
                self._intrinsics_path, self._extrinsics_path, self._pair_name,
            )
            # 再載入推理模型
            inference = StereoInference(
                model_dir=model_dir_abs,
                max_disp=max_disp,
                valid_iters=valid_iters,
                pad_multiple=pad_multiple,
            )
            inference.warmup()
            return rectifier, inference

        def _on_done(result):
            self._load_progress.stop()
            if isinstance(result, Exception):
                self._model_status.configure(
                    text=f"Error: {result}", fg=ACCENT_RED,
                )
                self._load_btn.configure(state="normal")
                return
            rectifier, inference = result
            self._state.rectifier = rectifier
            self._state.inference = inference
            self._model_ready = True
            self._model_status.configure(text="Ready", fg=ACCENT_GREEN)
            self._start_btn.configure(state="normal")

        def _load_safe():
            try:
                return _load()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return e

        run_in_thread(_load_safe, _on_done, self.winfo_toplevel())

    def _on_start(self):
        self.event_generate("<<SetupDone>>")

    def _on_quit(self):
        self.winfo_toplevel().destroy()


# ── 畫面 2：CaptureScreen ──

class CaptureScreen(tk.Frame):
    def __init__(
        self, master, manager: ScreenManager, state: AppState,
        cfg: dict, cam_left_idx: int, cam_right_idx: int,
    ):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._state = state
        self._cfg = cfg
        self._cam_left_idx = cam_left_idx
        self._cam_right_idx = cam_right_idx

        self._reader_l: CameraReader | None = None
        self._reader_r: CameraReader | None = None
        self._feed_l: CameraFeedWidget | None = None
        self._feed_r: CameraFeedWidget | None = None

        # 即時模式
        self._live_mode = False
        self._live_running = False
        self._live_thread: threading.Thread | None = None
        self._live_disp_img: np.ndarray | None = None
        self._live_depth_img: np.ndarray | None = None
        self._live_fps = 0.0
        self._live_latency = 0.0
        self._live_mean_depth = 0.0
        self._live_max_depth = cfg.get("stereo_depth", {}).get("max_depth", 10.0)

        # 3D viewer
        self._view3d_running = False
        self._view3d_thread: threading.Thread | None = None

        # Photo references
        self._disp_photo = None
        self._depth_photo = None

        self._build()

    def _build(self):
        # ── 頂部列：模式切換（左）+ 設定（右）──
        top_bar = tk.Frame(self, bg=PANEL_BG)
        top_bar.pack(fill="x", padx=5, pady=5)

        # 左：模式切換
        self._capture_mode_btn = tk.Button(
            top_bar, text="Capture Mode", command=self._switch_capture_mode,
            **BTN_ACCENT,
        )
        self._capture_mode_btn.pack(side="left", padx=10, pady=5)

        self._live_mode_btn = tk.Button(
            top_bar, text="Live Mode", command=self._switch_live_mode,
            **BTN_STYLE,
        )
        self._live_mode_btn.pack(side="left", padx=10, pady=5)

        # 右：設定（iterations + max_depth）
        tk.Label(
            top_bar, text="Max depth(m):", font=FONT_SMALL,
            bg=PANEL_BG, fg=TEXT_SECONDARY,
        ).pack(side="right", padx=(4, 10), pady=5)
        self._max_depth_var = tk.DoubleVar(
            value=self._cfg.get("stereo_depth", {}).get("max_depth", 10.0)
        )
        tk.Entry(
            top_bar, textvariable=self._max_depth_var, width=5,
            bg="#3a3a3a", fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            font=FONT_SMALL,
        ).pack(side="right", pady=5)

        ttk.Separator(top_bar, orient="vertical").pack(side="right", fill="y", padx=8, pady=4)

        self._iters_var = tk.IntVar(value=self._cfg.get("stereo_depth", {}).get("valid_iters", 8))
        for v, label in reversed([(4, "4 快速"), (6, "6"), (8, "8 精確")]):
            tk.Radiobutton(
                top_bar, text=label, variable=self._iters_var, value=v,
                bg=PANEL_BG, fg=TEXT_PRIMARY, selectcolor=PANEL_BG,
                activebackground=PANEL_BG, activeforeground=TEXT_PRIMARY,
                font=FONT_SMALL,
            ).pack(side="right", padx=2, pady=5)
        tk.Label(
            top_bar, text="Iters:", font=FONT_SMALL,
            bg=PANEL_BG, fg=TEXT_SECONDARY,
        ).pack(side="right", padx=(4, 2), pady=5)

        # ── 主要區域（可拖動分隔線）──
        paned = tk.PanedWindow(
            self, orient="horizontal", bg="#555555",
            sashwidth=5, sashrelief="raised",
        )
        paned.pack(fill="both", expand=True, padx=5, pady=0)

        # 左側：影像區
        img_area = tk.Frame(paned, bg=DARK_BG)

        # 上排：相機畫面
        self._camera_row = tk.Frame(img_area, bg=DARK_BG)
        self._camera_row.pack(fill="x", pady=2)

        self._feed_frame_l = tk.Frame(self._camera_row, bg=DARK_BG)
        self._feed_frame_l.pack(side="left", padx=2)

        self._feed_frame_r = tk.Frame(self._camera_row, bg=DARK_BG)
        self._feed_frame_r.pack(side="left", padx=2)

        # 下排：即時模式的 disparity/depth 顯示
        self._result_row = tk.Frame(img_area, bg=DARK_BG)
        self._result_row.pack(fill="x", pady=2)

        self._disp_label = tk.Label(self._result_row, bg=DARK_BG, cursor="crosshair")
        self._disp_label.pack(side="left", padx=2)
        self._disp_label.bind("<Button-1>", self._on_image_click)

        self._depth_label = tk.Label(self._result_row, bg=DARK_BG, cursor="crosshair")
        self._depth_label.pack(side="left", padx=2)
        self._depth_label.bind("<Button-1>", self._on_image_click)

        # 即時模式顯示尺寸（用於座標映射）
        self._live_display_size = (380, 285)

        paned.add(img_area, stretch="always")

        # ── 右側面板：統計 + 按鈕 ──
        self._panel = tk.Frame(paned, bg=PANEL_BG)

        paned.add(self._panel, width=200, minsize=150, stretch="never")

        # 說明
        tk.Label(
            self._panel, text="Disparity\n= 左右像素偏移\n\nDepth\n= 實際距離 (公尺)",
            font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_DIM, justify="left",
        ).pack(padx=10, pady=(10, 5), anchor="w")

        ttk.Separator(self._panel, orient="horizontal").pack(fill="x", padx=10, pady=5)

        # 即時統計
        self._stats_label = tk.Label(
            self._panel, text="", font=FONT_MONO, bg=PANEL_BG,
            fg=TEXT_SECONDARY, justify="left",
        )
        self._stats_label.pack(padx=10, anchor="w")

        ttk.Separator(self._panel, orient="horizontal").pack(fill="x", padx=10, pady=5)

        # 按鈕 — 用固定順序：capture/screenshot/view3d 共用同一個位置，quit 永遠在底部
        self._btn_frame = tk.Frame(self._panel, bg=PANEL_BG)
        self._btn_frame.pack(fill="x", padx=10, pady=5)

        self._capture_btn = tk.Button(
            self._btn_frame, text="Capture & Infer",
            command=self._on_capture, **BTN_ACCENT,
        )
        self._capture_btn.pack(fill="x", pady=2)

        self._screenshot_btn = tk.Button(
            self._btn_frame, text="Save Screenshot",
            command=self._on_screenshot, **BTN_STYLE,
        )
        # 不 pack — live mode 時才顯示

        self._view3d_btn = tk.Button(
            self._btn_frame, text="View 3D",
            command=self._on_view_3d, **BTN_STYLE,
        )
        # 不 pack — live mode 時才顯示

        self._quit_btn = tk.Button(
            self._btn_frame, text="Quit", command=self._on_quit, **BTN_DANGER,
        )
        self._quit_btn.pack(fill="x", pady=2)

        # 狀態
        self._status_label = tk.Label(
            self._panel, text="", font=FONT_SMALL, bg=PANEL_BG, fg=ACCENT_GREEN,
        )
        self._status_label.pack(pady=5)

    def start_cameras(self):
        """啟動相機讀取"""
        self._reader_l = CameraReader(self._cam_left_idx)
        self._reader_l.start()
        self._reader_r = CameraReader(self._cam_right_idx)
        self._reader_r.start()

        feed_size = (380, 285)

        self._feed_l = CameraFeedWidget(
            self._feed_frame_l, self._reader_l, feed_size,
        )
        self._feed_l.pack()
        self._feed_l.start_feed()
        self._feed_l._label.configure(cursor="crosshair")
        self._feed_l._label.bind("<Button-1>", self._on_image_click)

        self._feed_r = CameraFeedWidget(
            self._feed_frame_r, self._reader_r, feed_size,
        )
        self._feed_r.pack()
        self._feed_r.start_feed()
        self._feed_r._label.configure(cursor="crosshair")
        self._feed_r._label.bind("<Button-1>", self._on_image_click)

    def stop_cameras(self):
        """停止相機和即時模式"""
        self._view3d_running = False
        self._stop_live_loop()
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

    def _switch_capture_mode(self):
        """切換到擷取模式"""
        self._stop_live_loop()
        self._live_mode = False
        self._capture_mode_btn.configure(**BTN_ACCENT)
        self._live_mode_btn.configure(**BTN_STYLE)
        # 重排按鈕：capture → quit
        self._screenshot_btn.pack_forget()
        self._view3d_btn.pack_forget()
        self._capture_btn.pack_forget()
        self._quit_btn.pack_forget()
        self._capture_btn.pack(fill="x", pady=2)
        self._quit_btn.pack(fill="x", pady=2)
        # 隱藏即時結果顯示
        self._disp_label.configure(image="")
        self._depth_label.configure(image="")
        self._stats_label.configure(text="")

    def _switch_live_mode(self):
        """切換到即時模式"""
        if self._state.inference is None or self._state.rectifier is None:
            self._status_label.configure(text="Model not loaded!", fg=ACCENT_RED)
            return
        self._live_mode = True
        self._capture_mode_btn.configure(**BTN_STYLE)
        self._live_mode_btn.configure(**BTN_ACCENT)
        # 重排按鈕：screenshot → view3d → quit
        self._capture_btn.pack_forget()
        self._screenshot_btn.pack_forget()
        self._view3d_btn.pack_forget()
        self._quit_btn.pack_forget()
        self._screenshot_btn.pack(fill="x", pady=2)
        self._view3d_btn.pack(fill="x", pady=2)
        self._quit_btn.pack(fill="x", pady=2)
        self._state.inference.set_fast_mode(True)
        self._live_max_depth = self._max_depth_var.get()
        self._start_live_loop()

    def _start_live_loop(self):
        """啟動即時推理迴圈"""
        if self._live_running:
            return
        self._live_running = True
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()
        self._update_live_display()

    def _stop_live_loop(self):
        """停止即時推理迴圈"""
        self._live_running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=3.0)
            self._live_thread = None
        if self._state.inference is not None:
            self._state.inference.set_fast_mode(False)

    def _live_loop(self):
        """背景線程：持續 grab → rectify → infer → colorize"""
        rectifier = self._state.rectifier
        inference = self._state.inference
        sd_cfg = self._cfg.get("stereo_depth", {})
        max_disp = sd_cfg.get("max_disparity", 256)

        frame_count = 0
        fps_start = time.time()

        while self._live_running:
            if self._reader_l is None or self._reader_r is None:
                time.sleep(0.01)
                continue
            frame_l = self._reader_l.frame
            frame_r = self._reader_r.frame
            if frame_l is None or frame_r is None:
                time.sleep(0.01)
                continue

            frame_l = frame_l.copy()
            frame_r = frame_r.copy()

            t0 = time.time()
            rect_l, rect_r = rectifier.rectify(frame_l, frame_r)
            disp = inference.predict_disparity(rect_l, rect_r)
            latency = (time.time() - t0) * 1000

            # 從主線程快取的值讀取（避免跨線程存取 tkinter 變數）
            max_depth = self._live_max_depth
            depth = disparity_to_depth(
                disp, rectifier.focal_length, rectifier.baseline,
                max_depth=max_depth,
            )

            self._live_disp_img = colorize_disparity(disp, max_disp)
            self._live_depth_img = colorize_depth(depth, max_depth)

            # 更新即時截圖用的數據
            self._state.rect_left = rect_l
            self._state.rect_right = rect_r
            self._state.disparity = disp
            self._state.depth = depth

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self._live_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            self._live_latency = latency
            valid = depth[depth > 0]
            self._live_mean_depth = float(np.mean(valid)) if len(valid) > 0 else 0.0

    def _on_image_click(self, event):
        """點擊影像：標記測量點 A/B，計算 3D 距離"""
        state = self._state
        if state.depth is None or state.rectifier is None:
            return

        # 顯示座標 → 原始分辨率座標
        dw, dh = self._live_display_size
        ow, oh = state.rectifier.image_size  # (640, 480)
        u = int(event.x * ow / dw)
        v = int(event.y * oh / dh)
        u = max(0, min(u, ow - 1))
        v = max(0, min(v, oh - 1))

        if state.measure_a is None or (state.measure_a is not None and state.measure_b is not None):
            # 第一點或重置
            state.measure_a = (u, v)
            state.measure_b = None
            state.measure_dist = None
        else:
            # 第二點
            state.measure_b = (u, v)
            state.measure_dist = measure_distance_3d(
                state.measure_a, state.measure_b,
                state.depth, state.rectifier.Q,
            )

    def _apply_measurement_overlay(self, img: np.ndarray) -> np.ndarray:
        """在影像上疊加測量標記"""
        state = self._state
        if state.measure_a is None:
            return img
        return draw_measurement_overlay(
            img, state.measure_a, state.measure_b,
            distance=state.measure_dist,
            depth=state.depth,
            Q=state.rectifier.Q if state.rectifier else None,
        )

    def _update_live_display(self):
        """定時更新即時模式的顯示（主線程）"""
        if not self._live_running:
            return

        display_size = self._live_display_size

        if self._live_disp_img is not None:
            disp_show = self._apply_measurement_overlay(self._live_disp_img)
            self._disp_photo = cv_to_photoimage(disp_show, display_size)
            self._disp_label.configure(image=self._disp_photo)

        if self._live_depth_img is not None:
            depth_show = self._apply_measurement_overlay(self._live_depth_img)
            self._depth_photo = cv_to_photoimage(depth_show, display_size)
            self._depth_label.configure(image=self._depth_photo)

        # 分辨率 + 測距資訊
        disp = self._state.disparity
        dep = self._state.depth
        if disp is not None and dep is not None:
            res = f"{disp.shape[1]}x{disp.shape[0]}"
        else:
            res = "--"

        measure_text = ""
        if self._state.measure_dist is not None:
            measure_text = f"\nDistance:   {self._state.measure_dist*100:.1f} cm"
        elif self._state.measure_a is not None:
            measure_text = "\nClick 2nd point..."

        self._stats_label.configure(
            text=f"Resolution: {res}\n"
                 f"FPS:        {self._live_fps:.1f}\n"
                 f"Latency:    {self._live_latency:.0f} ms\n"
                 f"Mean depth: {self._live_mean_depth:.2f} m"
                 f"{measure_text}",
        )

        # 同步 max_depth 設定（主線程安全讀取 tkinter 變數）
        self._live_max_depth = self._max_depth_var.get()

        self.after(50, self._update_live_display)

    def _on_capture(self):
        """擷取模式：抓取一幀並推理"""
        if self._state.inference is None or self._state.rectifier is None:
            self._status_label.configure(text="Model not loaded!", fg=ACCENT_RED)
            return
        if self._reader_l is None or self._reader_r is None:
            return
        frame_l = self._reader_l.frame
        frame_r = self._reader_r.frame
        if frame_l is None or frame_r is None:
            self._status_label.configure(text="No camera frames!", fg=ACCENT_RED)
            return

        frame_l = frame_l.copy()
        frame_r = frame_r.copy()

        # 設定迭代次數
        iters = self._iters_var.get()
        self._state.inference._valid_iters = iters
        self._state.inference._model.args.valid_iters = iters
        self._capture_btn.configure(state="disabled")
        self._status_label.configure(text="Rectifying + Inferring...", fg=ACCENT_YELLOW)

        rectifier = self._state.rectifier
        inference = self._state.inference

        def _infer():
            rect_l, rect_r = rectifier.rectify(frame_l, frame_r)
            t0 = time.time()
            disp = inference.predict_disparity(rect_l, rect_r)
            elapsed_ms = (time.time() - t0) * 1000
            max_depth = self._max_depth_var.get()
            depth = disparity_to_depth(
                disp, rectifier.focal_length, rectifier.baseline,
                max_depth=max_depth,
            )
            return rect_l, rect_r, disp, depth, elapsed_ms

        def _on_done(result):
            self._capture_btn.configure(state="normal")
            if isinstance(result, Exception):
                self._status_label.configure(text=f"Error: {result}", fg=ACCENT_RED)
                return
            rect_l, rect_r, disp, depth, elapsed_ms = result
            self._state.rect_left = rect_l
            self._state.rect_right = rect_r
            self._state.disparity = disp
            self._state.depth = depth
            self._state.inference_time_ms = elapsed_ms
            self._status_label.configure(
                text=f"Done ({elapsed_ms:.0f} ms)", fg=ACCENT_GREEN,
            )
            self.winfo_toplevel().event_generate("<<CaptureReady>>")

        def _infer_safe():
            try:
                return _infer()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return e

        run_in_thread(_infer_safe, _on_done, self.winfo_toplevel())

    def _on_screenshot(self):
        """即時模式截圖儲存"""
        # 快照當前幀（避免 live loop 覆寫）
        depth = self._state.depth
        disparity = self._state.disparity
        if depth is None or disparity is None:
            return
        depth = depth.copy()
        disparity = disparity.copy()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        depth_dir = os.path.join(root_dir, "phase3_stereo_depth", "outputs", "stereo_depth")
        disp_dir = os.path.join(root_dir, "phase3_stereo_depth", "outputs", "disparity")

        save_depth(depth, depth_dir, ts)
        max_disp = self._cfg.get("stereo_depth", {}).get("max_disparity", 256)
        disp_color = colorize_disparity(disparity, max_disp)
        save_disparity_vis(disp_color, disp_dir, ts)
        self._status_label.configure(text=f"Saved: {ts}", fg=ACCENT_GREEN)

    def _on_view_3d(self):
        """開啟即時更新的 3D 點雲視窗（背景線程）"""
        # 已開啟則忽略
        if self._view3d_running:
            self._status_label.configure(text="3D viewer already open", fg=ACCENT_YELLOW)
            return

        state = self._state
        if state.depth is None or state.rectifier is None:
            self._status_label.configure(text="No depth data yet", fg=ACCENT_RED)
            return

        try:
            import open3d as o3d
            _ = o3d.geometry.PointCloud()
        except ImportError:
            self._status_label.configure(text="open3d not installed", fg=ACCENT_RED)
            return

        self._view3d_running = True
        self._view3d_thread = threading.Thread(target=self._view3d_loop, daemon=True)
        self._view3d_thread.start()
        self._status_label.configure(text="3D viewer opened (real-time)", fg=ACCENT_GREEN)

    def _view3d_loop(self):
        """背景線程：Open3D 即時點雲視覺化，持續從 live loop 讀取最新深度

        使用 depth_to_pointcloud_fast（numpy 直算 + 2x 降採樣），
        跳過 cv2.reprojectImageTo3D 和 voxel_down_sample，
        每幀約 2ms（vs 原本 196ms）。
        """
        import open3d as o3d

        rectifier = self._state.rectifier
        Q = rectifier.Q

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Live 3D Point Cloud", width=1024, height=768)

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        pcd = o3d.geometry.PointCloud()
        geometry_added = False

        while self._view3d_running:
            depth = self._state.depth
            rect_left = self._state.rect_left
            if depth is None:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.03)
                continue

            # 快速路徑：numpy 直算 + 2x 降採樣 → ~2ms
            points, colors = depth_to_pointcloud_fast(
                depth, Q,
                color_image=rect_left,
                max_depth=self._live_max_depth,
                subsample=2,
            )

            if len(points) == 0:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.03)
                continue

            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            if not geometry_added:
                vis.add_geometry(pcd)
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, -1, 0])
                geometry_added = True
            else:
                vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

        vis.destroy_window()
        self._view3d_running = False

    def _on_quit(self):
        self.stop_cameras()
        self.winfo_toplevel().destroy()


# ── 畫面 3：ResultScreen ──

class ResultScreen(tk.Frame):
    def __init__(
        self, master, manager: ScreenManager, state: AppState, cfg: dict,
    ):
        super().__init__(master, bg=DARK_BG)
        self._manager = manager
        self._state = state
        self._cfg = cfg

        # Photo references
        self._rect_photo = None
        self._disp_photo = None
        self._depth_photo = None

        self._build()

    def _build(self):
        # 上排三張圖
        img_row = tk.Frame(self, bg=DARK_BG)
        img_row.pack(pady=5)

        self._rect_label = tk.Label(img_row, bg=DARK_BG, cursor="crosshair")
        self._rect_label.pack(side="left", padx=3)
        self._rect_label.bind("<Button-1>", self._on_image_click)

        self._disp_label = tk.Label(img_row, bg=DARK_BG, cursor="crosshair")
        self._disp_label.pack(side="left", padx=3)
        self._disp_label.bind("<Button-1>", self._on_image_click)

        self._depth_label = tk.Label(img_row, bg=DARK_BG, cursor="crosshair")
        self._depth_label.pack(side="left", padx=3)
        self._depth_label.bind("<Button-1>", self._on_image_click)

        self._result_display_size = (350, 262)

        # 圖片標題列
        title_row = tk.Frame(self, bg=DARK_BG)
        title_row.pack(fill="x", padx=10)
        titles = [
            "Rectified Left\n(校正後左圖)",
            "Disparity\n(視差圖：左右像素偏移量)",
            "Depth\n(深度圖：距離，公尺)",
        ]
        for t in titles:
            tk.Label(
                title_row, text=t, font=FONT_SMALL, bg=DARK_BG, fg=TEXT_SECONDARY,
                width=34, anchor="center", justify="center",
            ).pack(side="left", padx=3)

        # 下排統計面板
        stats_frame = tk.LabelFrame(
            self, text="Depth Statistics", font=FONT_BODY,
            bg=PANEL_BG, fg=TEXT_PRIMARY, padx=20, pady=10,
        )
        stats_frame.pack(padx=20, pady=10, fill="x")

        self._stats_text = tk.Label(
            stats_frame, text="", font=FONT_MONO, bg=PANEL_BG,
            fg=TEXT_SECONDARY, justify="left",
        )
        self._stats_text.pack(anchor="w")

        # 按鈕列
        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Save Depth", command=self._on_save_depth, **BTN_ACCENT,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="Save Point Cloud", command=self._on_save_ply, **BTN_STYLE,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="View 3D", command=self._on_view_3d, **BTN_STYLE,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="Recapture", command=self._on_recapture, **BTN_STYLE,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="Quit", command=self._on_quit, **BTN_DANGER,
        ).pack(side="left", padx=8)

        self._save_status = tk.Label(
            self, text="", font=FONT_SMALL, bg=DARK_BG, fg=ACCENT_GREEN,
        )
        self._save_status.pack(pady=5)

    def load_results(self):
        """從 AppState 載入並顯示結果"""
        state = self._state
        sd_cfg = self._cfg.get("stereo_depth", {})
        max_disp = sd_cfg.get("max_disparity", 256)
        max_depth = sd_cfg.get("max_depth", 10.0)

        display_size = (350, 262)

        # 校正左圖
        if state.rect_left is not None:
            self._rect_photo = cv_to_photoimage(state.rect_left, display_size)
            self._rect_label.configure(image=self._rect_photo)

        # 視差彩色圖
        if state.disparity is not None:
            disp_color = colorize_disparity(state.disparity, max_disp)
            self._disp_photo = cv_to_photoimage(disp_color, display_size)
            self._disp_label.configure(image=self._disp_photo)

        # 深度彩色圖
        if state.depth is not None:
            depth_color = colorize_depth(state.depth, max_depth)
            self._depth_photo = cv_to_photoimage(depth_color, display_size)
            self._depth_label.configure(image=self._depth_photo)

            # 統計
            H, W = state.depth.shape[:2]
            stats = compute_depth_stats(state.depth)
            text = (
                f"Resolution:   {W}x{H}\n"
                f"Min depth:    {stats['min']:.3f} m\n"
                f"Max depth:    {stats['max']:.3f} m\n"
                f"Mean depth:   {stats['mean']:.3f} m\n"
                f"Median depth: {stats['median']:.3f} m\n"
                f"Valid pixels: {stats['valid_pixels']}/{stats['total_pixels']}"
                f" ({stats['valid_ratio']:.1%})\n"
                f"Inference:    {state.inference_time_ms:.0f} ms"
            )
            self._stats_text.configure(text=text)

    def _on_image_click(self, event):
        """點擊影像：標記測量點 A/B，計算 3D 距離"""
        state = self._state
        if state.depth is None or state.rectifier is None:
            return

        dw, dh = self._result_display_size
        ow, oh = state.rectifier.image_size
        u = max(0, min(int(event.x * ow / dw), ow - 1))
        v = max(0, min(int(event.y * oh / dh), oh - 1))

        if state.measure_a is None or (state.measure_a is not None and state.measure_b is not None):
            state.measure_a = (u, v)
            state.measure_b = None
            state.measure_dist = None
        else:
            state.measure_b = (u, v)
            state.measure_dist = measure_distance_3d(
                state.measure_a, state.measure_b,
                state.depth, state.rectifier.Q,
            )

        self._redraw_with_overlay()

    def _redraw_with_overlay(self):
        """重繪三張圖並疊加測量標記"""
        state = self._state
        sd_cfg = self._cfg.get("stereo_depth", {})
        max_disp = sd_cfg.get("max_disparity", 256)
        max_depth = sd_cfg.get("max_depth", 10.0)
        display_size = self._result_display_size

        def _overlay(img):
            return draw_measurement_overlay(
                img, state.measure_a, state.measure_b,
                distance=state.measure_dist,
                depth=state.depth,
                Q=state.rectifier.Q if state.rectifier else None,
            )

        if state.rect_left is not None:
            self._rect_photo = cv_to_photoimage(_overlay(state.rect_left), display_size)
            self._rect_label.configure(image=self._rect_photo)

        if state.disparity is not None:
            disp_color = colorize_disparity(state.disparity, max_disp)
            self._disp_photo = cv_to_photoimage(_overlay(disp_color), display_size)
            self._disp_label.configure(image=self._disp_photo)

        if state.depth is not None:
            depth_color = colorize_depth(state.depth, max_depth)
            self._depth_photo = cv_to_photoimage(_overlay(depth_color), display_size)
            self._depth_label.configure(image=self._depth_photo)

        # 更新統計文字
        measure_text = ""
        if state.measure_dist is not None:
            measure_text = f"\nDistance:     {state.measure_dist*100:.1f} cm"
        elif state.measure_a is not None:
            measure_text = "\n(click 2nd point...)"
        self._save_status.configure(text=measure_text.strip(), fg=ACCENT_YELLOW)

    def _get_output_dirs(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        depth_dir = os.path.join(root_dir, "phase3_stereo_depth", "outputs", "stereo_depth")
        disp_dir = os.path.join(root_dir, "phase3_stereo_depth", "outputs", "disparity")
        return root_dir, depth_dir, disp_dir

    def _on_save_depth(self):
        state = self._state
        if state.depth is None or state.disparity is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir, depth_dir, disp_dir = self._get_output_dirs()

        save_depth(state.depth, depth_dir, ts)
        sd_cfg = self._cfg.get("stereo_depth", {})
        max_disp = sd_cfg.get("max_disparity", 256)
        disp_color = colorize_disparity(state.disparity, max_disp)
        save_disparity_vis(disp_color, disp_dir, ts)
        self._save_status.configure(text=f"Saved depth + disparity: {ts}", fg=ACCENT_GREEN)

    def _on_save_ply(self):
        state = self._state
        rectifier = state.rectifier
        if state.depth is None or rectifier is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir, depth_dir, _ = self._get_output_dirs()

        pc = depth_to_pointcloud(
            state.depth, rectifier.Q,
            color_image=state.rect_left,
            max_depth=self._cfg.get("stereo_depth", {}).get("max_depth", 10.0),
        )
        ply_path = os.path.join(depth_dir, f"pointcloud_{ts}.ply")
        save_pointcloud_ply(pc, ply_path)
        self._save_status.configure(
            text=f"Saved PLY ({len(pc.points)} pts): {ts}", fg=ACCENT_GREEN,
        )

    def _on_view_3d(self):
        """用 Open3D 開啟互動式 3D 點雲檢視器"""
        state = self._state
        rectifier = state.rectifier
        if state.depth is None or rectifier is None:
            return

        self._save_status.configure(text="Generating point cloud...", fg=ACCENT_YELLOW)
        self.update_idletasks()

        pc = depth_to_pointcloud(
            state.depth, rectifier.Q,
            color_image=state.rect_left,
            max_depth=self._cfg.get("stereo_depth", {}).get("max_depth", 10.0),
        )

        if pc.points.shape[0] == 0:
            self._save_status.configure(text="No valid points!", fg=ACCENT_RED)
            return

        try:
            import open3d as o3d
        except ImportError:
            self._save_status.configure(text="open3d not installed", fg=ACCENT_RED)
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pc.colors.astype(float) / 255.0)

        # 下採樣加速顯示
        if len(pc.points) > 100000:
            pcd = pcd.voxel_down_sample(voxel_size=0.005)

        self._save_status.configure(
            text=f"Opening 3D viewer ({len(pcd.points)} pts)...", fg=ACCENT_GREEN,
        )
        self.update_idletasks()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        vis.run()
        vis.destroy_window()

        self._save_status.configure(text="3D viewer closed", fg=TEXT_SECONDARY)

    def _on_recapture(self):
        self.winfo_toplevel().event_generate("<<Recapture>>")

    def _on_quit(self):
        self.winfo_toplevel().destroy()


# ── 主流程 ──

def run_stereo_depth_gui(
    cfg: dict,
    intrinsics_path: str,
    extrinsics_path: str,
    pair_name: str = "cam0_cam1",
) -> None:
    """啟動 Phase 3 Stereo Depth GUI"""

    cameras_cfg = cfg["cameras"]
    cam_left, cam_right = pair_name.split("_", 1)
    cam_left_idx = cameras_cfg[cam_left]["index"]
    cam_right_idx = cameras_cfg[cam_right]["index"]

    root = tk.Tk()
    root.title("Phase 3 — Stereo Depth")
    root.configure(bg=DARK_BG)
    root.geometry("1200x750")

    manager = ScreenManager(root)
    state = AppState()

    # 建立畫面
    setup_screen = SetupScreen(
        root, manager, state, cfg,
        intrinsics_path, extrinsics_path, pair_name,
    )

    capture_screen: CaptureScreen | None = None
    result_screen: ResultScreen | None = None

    def show_setup():
        manager.show(setup_screen)

    def on_setup_done(event=None):
        nonlocal capture_screen
        capture_screen = CaptureScreen(
            root, manager, state, cfg, cam_left_idx, cam_right_idx,
        )
        manager.show(capture_screen)
        capture_screen.start_cameras()

    def on_capture_ready(event=None):
        nonlocal result_screen
        result_screen = ResultScreen(root, manager, state, cfg)
        manager.show(result_screen)
        result_screen.load_results()

    def on_recapture(event=None):
        if capture_screen is not None:
            manager.show(capture_screen)

    def on_close():
        """確保退出時釋放相機資源"""
        if capture_screen is not None:
            capture_screen.stop_cameras()
        root.destroy()

    # 綁定事件
    setup_screen.bind("<<SetupDone>>", on_setup_done)
    root.bind("<<CaptureReady>>", on_capture_ready)
    root.bind("<<Recapture>>", on_recapture)
    root.protocol("WM_DELETE_WINDOW", on_close)

    show_setup()
    root.mainloop()
