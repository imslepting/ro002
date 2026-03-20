"""CapturePoint Skill 互動測試 GUI — 即時雙目深度 + 階段式 Pipeline

用法:
    conda run -n ro002 python phase5_vlm_planning/test_skill_capture_point.py

階段式流水線:
    1. 開啟雙目相機 → 即時顯示 RGB + 深度（可點擊量距）
    2. 按 [Capture] → 凍結當前幀
    3. 輸入 text → [Segment] → SAM3 mask 覆蓋顯示
    4. [Capture Point] → GraspGen 抓取結果顯示

狀態機:
    LOADING → IDLE → LIVE ⇄ CAPTURED → SEGMENTED → GRASPED
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import logging

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.camera_manager import CameraReader


# ── 工具函數 ──

def _load_intrinsics() -> dict:
    """載入 intrinsics.json，返回 {cam_name: K_matrix} dict"""
    path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for cam_name, cam_data in data.items():
        if "K" in cam_data:
            result[cam_name] = np.array(cam_data["K"], dtype=np.float64)
    return result


def _load_settings() -> dict:
    """載入 config/settings.yaml"""
    path = os.path.join(_ROOT, "config", "settings.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── 共享狀態容器 ──

class PipelineState:
    """Pipeline 各階段的共享數據"""

    def __init__(self):
        self.stage: str = "LOADING"

        # Phase3 模型
        self.rectifier = None   # StereoRectifier
        self.stereo_inf = None  # StereoInference

        # Phase5 模型
        self.sam3_skill = None
        self.capture_skill = None

        # 相機
        self.reader_l: CameraReader | None = None
        self.reader_r: CameraReader | None = None

        # 即時數據（bg thread 寫，main thread 讀）
        self.live_rect_left: np.ndarray | None = None
        self.live_depth: np.ndarray | None = None
        self.live_depth_color: np.ndarray | None = None
        self.live_fps: float = 0.0
        self.live_latency_ms: float = 0.0

        # 凍結數據
        self.frozen_rgb: np.ndarray | None = None
        self.frozen_depth: np.ndarray | None = None
        self.frozen_depth_color: np.ndarray | None = None
        self.K_rect: np.ndarray | None = None
        self.Q: np.ndarray | None = None

        # 量測
        self.measure_a: tuple[int, int] | None = None
        self.measure_b: tuple[int, int] | None = None
        self.measure_dist: float | None = None

        # SAM3 / Grasp
        self.mask: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.grasp_result = None

        # Workspace 可視化
        self.workspace_limits: dict = {}

        # 其他
        self.T_cam2arm: np.ndarray = np.eye(4)
        self.intrinsics: dict = {}


# ── 主 GUI 類 ──

class CapturePointPipelineGUI:
    """單視窗、階段式流水線 GUI"""

    _LIVE_DISPLAY_INTERVAL = 50  # ms

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CapturePoint Pipeline (Live Stereo + SAM3 + GraspGen)")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)

        self._state = PipelineState()
        self._state.intrinsics = _load_intrinsics()
        self._settings = _load_settings()
        self._stereo_cfg = self._settings.get("stereo_depth", {})
        cp_cfg = self._settings.get("skill_capture_point", {})
        self._state.workspace_limits = cp_cfg.get("workspace_limits", {
            "x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.6],
        })
        self._gripper_width = cp_cfg.get("gripper_width", 0.136)
        self._pre_grasp_distance = cp_cfg.get("pre_grasp_distance", 0.10)

        # 即時推理控制
        self._live_running = False
        self._live_thread: threading.Thread | None = None

        # 顯示用 PhotoImage 引用（防止 GC）
        self._photo_rgb: ImageTk.PhotoImage | None = None
        self._photo_depth: ImageTk.PhotoImage | None = None
        self._photo_result: ImageTk.PhotoImage | None = None

        # 顯示面板的 canvas 尺寸快取（用於座標映射）
        self._rgb_canvas_size: tuple[int, int] = (1, 1)
        self._depth_canvas_size: tuple[int, int] = (1, 1)
        # 原始圖像尺寸（用於座標映射）
        self._display_img_size: tuple[int, int] = (640, 480)

        self._build_ui()
        self._bind_events()
        self._update_button_states()
        self._load_models_async()

    # ══════════════════════════════════════════════════════════════
    # UI 構建
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── 頂部控制列 1: Live 控制 + 離線 fallback ──
        ctrl1 = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        ctrl1.pack(fill=tk.X)

        self._start_live_btn = ttk.Button(
            ctrl1, text="Start Live", command=self._start_live,
        )
        self._start_live_btn.pack(side=tk.LEFT)

        self._capture_btn = ttk.Button(
            ctrl1, text="Capture \u25a0", command=self._do_capture,
        )
        self._capture_btn.pack(side=tk.LEFT, padx=(4, 0))

        self._back_live_btn = ttk.Button(
            ctrl1, text="Back to Live \u21ba", command=self._back_to_live,
        )
        self._back_live_btn.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(ctrl1, text="Open RGB", command=self._open_rgb).pack(side=tk.LEFT)
        ttk.Button(ctrl1, text="Open Depth", command=self._open_depth).pack(
            side=tk.LEFT, padx=(4, 0),
        )

        ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # T_cam2arm
        self._t_label = tk.StringVar(value="T_cam2arm: Identity")
        ttk.Label(ctrl1, textvariable=self._t_label, foreground="gray").pack(side=tk.LEFT)
        ttk.Button(ctrl1, text="Load T...", command=self._load_t_cam2arm).pack(
            side=tk.LEFT, padx=4,
        )

        ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Camera K 選擇
        ttk.Label(ctrl1, text="Camera:").pack(side=tk.LEFT)
        self._cam_var = tk.StringVar()
        cam_names = list(self._state.intrinsics.keys()) if self._state.intrinsics else ["(no intrinsics)"]
        self._cam_combo = ttk.Combobox(
            ctrl1, textvariable=self._cam_var, values=cam_names,
            state="readonly", width=10,
        )
        self._cam_combo.pack(side=tk.LEFT, padx=4)
        if cam_names and cam_names[0] != "(no intrinsics)":
            self._cam_combo.current(0)

        # 狀態
        self._status_var = tk.StringVar(value="Loading models...")
        ttk.Label(ctrl1, textvariable=self._status_var, foreground="gray").pack(side=tk.RIGHT)

        # ── 頂部控制列 2: SAM3 + Capture Point ──
        ctrl2 = ttk.Frame(self.root, padding=(8, 2, 8, 4))
        ctrl2.pack(fill=tk.X)

        ttk.Label(ctrl2, text="SAM3 Prompt:").pack(side=tk.LEFT)
        self._prompt_var = tk.StringVar()
        self._prompt_entry = ttk.Entry(ctrl2, textvariable=self._prompt_var, width=24)
        self._prompt_entry.pack(side=tk.LEFT, padx=4)

        self._seg_btn = ttk.Button(ctrl2, text="Segment", command=self._run_segment)
        self._seg_btn.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(ctrl2, text="Load Mask .npy", command=self._open_mask).pack(
            side=tk.LEFT, padx=(4, 0),
        )

        ttk.Separator(ctrl2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._capture_point_btn = ttk.Button(
            ctrl2, text="Capture Point", command=self._run_capture_point,
        )
        self._capture_point_btn.pack(side=tk.LEFT, padx=4)

        self._save_btn = ttk.Button(ctrl2, text="Save Result...", command=self._save_result)
        self._save_btn.pack(side=tk.RIGHT)

        self._view3d_btn = ttk.Button(ctrl2, text="View 3D", command=self._view_3d_grasp)
        self._view3d_btn.pack(side=tk.RIGHT, padx=(0, 4))

        self._show_ws_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl2, text="Show Workspace", variable=self._show_ws_var,
            command=self._on_workspace_toggle,
        ).pack(side=tk.RIGHT, padx=(0, 8))

        # ── 三欄圖片顯示 ──
        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.LabelFrame(panes, text="RGB (Live/Frozen) + Mask", padding=4)
        panes.add(left, weight=1)
        self._rgb_canvas = tk.Canvas(left, bg="#2b2b2b")
        self._rgb_canvas.pack(fill=tk.BOTH, expand=True)

        mid = ttk.LabelFrame(panes, text="Depth (Live/Frozen)", padding=4)
        panes.add(mid, weight=1)
        self._depth_canvas = tk.Canvas(mid, bg="#2b2b2b")
        self._depth_canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(panes, text="Grasp Result", padding=4)
        panes.add(right, weight=1)
        self._result_canvas = tk.Canvas(right, bg="#2b2b2b")
        self._result_canvas.pack(fill=tk.BOTH, expand=True)

        # ── 底部資訊列 ──
        info = ttk.Frame(self.root, padding=4)
        info.pack(fill=tk.X)
        self._info_var = tk.StringVar()
        ttk.Label(
            info, textvariable=self._info_var, foreground="#555555",
            font=("monospace", 10),
        ).pack(side=tk.LEFT)

    def _bind_events(self):
        self._rgb_canvas.bind("<Button-1>", self._on_rgb_click)
        self._depth_canvas.bind("<Button-1>", self._on_depth_click)
        self._rgb_canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self._depth_canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self._result_canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

    def _on_canvas_resize(self):
        """畫布大小變更時重繪（僅非 LIVE 模式）"""
        if self._state.stage not in ("LIVE", "LOADING"):
            self._redraw_all()

    # ══════════════════════════════════════════════════════════════
    # 狀態機 & 按鈕管理
    # ══════════════════════════════════════════════════════════════

    def _set_stage(self, stage: str):
        self._state.stage = stage
        self._update_button_states()

    def _update_button_states(self):
        stage = self._state.stage
        # 按鈕狀態表
        btn_map = {
            "LOADING":   {"start": 0, "capture": 0, "back": 0, "seg": 0, "mask": 0, "cp": 0, "save": 0, "v3d": 0},
            "IDLE":      {"start": 1, "capture": 0, "back": 0, "seg": 0, "mask": 0, "cp": 0, "save": 0, "v3d": 0},
            "LIVE":      {"start": 0, "capture": 1, "back": 0, "seg": 0, "mask": 0, "cp": 0, "save": 0, "v3d": 0},
            "CAPTURED":  {"start": 0, "capture": 0, "back": 1, "seg": 1, "mask": 1, "cp": 0, "save": 0, "v3d": 0},
            "SEGMENTED": {"start": 0, "capture": 0, "back": 1, "seg": 1, "mask": 1, "cp": 1, "save": 0, "v3d": 0},
            "GRASPED":   {"start": 0, "capture": 0, "back": 1, "seg": 0, "mask": 0, "cp": 0, "save": 1, "v3d": 1},
        }
        m = btn_map.get(stage, btn_map["LOADING"])
        self._start_live_btn.config(state=tk.NORMAL if m["start"] else tk.DISABLED)
        self._capture_btn.config(state=tk.NORMAL if m["capture"] else tk.DISABLED)
        self._back_live_btn.config(state=tk.NORMAL if m["back"] else tk.DISABLED)
        self._seg_btn.config(state=tk.NORMAL if m["seg"] else tk.DISABLED)
        self._capture_point_btn.config(state=tk.NORMAL if m["cp"] else tk.DISABLED)
        self._save_btn.config(state=tk.NORMAL if m["save"] else tk.DISABLED)
        self._view3d_btn.config(state=tk.NORMAL if m["v3d"] else tk.DISABLED)

    # ══════════════════════════════════════════════════════════════
    # 模型載入
    # ══════════════════════════════════════════════════════════════

    def _load_models_async(self):
        def _load():
            errors = []

            # Phase3: StereoRectifier
            try:
                from phase3_stereo_depth.src.stereo_rectifier import StereoRectifier
                intrinsics_path = os.path.join(
                    _ROOT, "phase1_intrinsics", "outputs", "intrinsics.json",
                )
                extrinsics_path = os.path.join(
                    _ROOT, "phase2_extrinsics", "outputs", "extrinsics.json",
                )
                self._state.rectifier = StereoRectifier(
                    intrinsics_path, extrinsics_path, "cam0_cam1",
                )
                self._state.K_rect = self._state.rectifier._P1[:3, :3].copy()
                self._state.Q = self._state.rectifier.Q.copy()
            except Exception as exc:
                errors.append(f"Rectifier: {exc}")

            # Phase3: StereoInference
            try:
                from phase3_stereo_depth.src.stereo_inference import StereoInference
                model_dir = self._stereo_cfg.get(
                    "model_dir",
                    "external/Fast-FoundationStereo/weights/23-36-37/"
                    "model_best_bp2_serialize.pth",
                )
                if not os.path.isabs(model_dir):
                    model_dir = os.path.join(_ROOT, model_dir)
                self._state.stereo_inf = StereoInference(
                    model_dir=model_dir,
                    max_disp=self._stereo_cfg.get("max_disparity", 256),
                    valid_iters=self._stereo_cfg.get("valid_iters", 8),
                    pad_multiple=self._stereo_cfg.get("pad_multiple", 32),
                )
                # 即時模式用 fast_mode
                self._state.stereo_inf.set_fast_mode(True)
            except Exception as exc:
                errors.append(f"StereoInference: {exc}")

            # Phase5: SAM3
            try:
                from phase5_vlm_planning.skills.skill_sam3 import SAM3Skill
                self._state.sam3_skill = SAM3Skill()
            except Exception as exc:
                errors.append(f"SAM3: {exc}")

            # Phase5: CapturePoint
            try:
                from phase5_vlm_planning.skills.skill_capture_point import CapturePointSkill
                self._state.capture_skill = CapturePointSkill()
            except Exception as exc:
                errors.append(f"CapturePoint: {exc}")

            self.root.after(0, self._on_models_loaded, errors)

        threading.Thread(target=_load, daemon=True).start()

    def _on_models_loaded(self, errors: list[str]):
        self._set_stage("IDLE")
        if not errors:
            self._status_var.set("All models ready")
        else:
            msg = "; ".join(errors)
            self._status_var.set(f"Partial load: {msg}")
            print(f"[Pipeline] Model load errors: {msg}")

    # ══════════════════════════════════════════════════════════════
    # Live Loop — 即時雙目深度推理
    # ══════════════════════════════════════════════════════════════

    def _start_live(self):
        if self._state.stage != "IDLE":
            return
        if self._state.rectifier is None or self._state.stereo_inf is None:
            self._status_var.set("Phase3 models not loaded — cannot start live")
            return

        # 開啟相機
        cam_cfg = self._settings.get("cameras", {})
        cam_l_idx = cam_cfg.get("cam0", {}).get("index", 0)
        cam_r_idx = cam_cfg.get("cam1", {}).get("index", 1)

        self._state.reader_l = CameraReader(cam_l_idx)
        self._state.reader_r = CameraReader(cam_r_idx)
        self._state.reader_l.start()
        self._state.reader_r.start()

        # 啟動即時推理
        self._state.stereo_inf.set_fast_mode(True)
        self._live_running = True
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

        self._set_stage("LIVE")
        self._status_var.set("Live stereo depth running...")
        self._update_live_display()

    def _stop_live(self):
        """停止即時推理和相機"""
        self._live_running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=3.0)
            self._live_thread = None
        if self._state.reader_l is not None:
            self._state.reader_l.stop()
            self._state.reader_l = None
        if self._state.reader_r is not None:
            self._state.reader_r.stop()
            self._state.reader_r = None
        if self._state.stereo_inf is not None:
            self._state.stereo_inf.set_fast_mode(False)

    def _live_loop(self):
        """背景線程：持續 grab → rectify → infer → colorize"""
        from phase3_stereo_depth.src.depth_converter import disparity_to_depth
        from phase3_stereo_depth.src.depth_utils import colorize_depth

        rectifier = self._state.rectifier
        inference = self._state.stereo_inf
        sd_cfg = self._stereo_cfg

        frame_count = 0
        fps_start = time.time()

        while self._live_running:
            reader_l = self._state.reader_l
            reader_r = self._state.reader_r
            if reader_l is None or reader_r is None:
                time.sleep(0.01)
                continue
            frame_l = reader_l.frame
            frame_r = reader_r.frame
            if frame_l is None or frame_r is None:
                time.sleep(0.01)
                continue

            frame_l = frame_l.copy()
            frame_r = frame_r.copy()

            t0 = time.time()
            rect_l, rect_r = rectifier.rectify(frame_l, frame_r)
            disp = inference.predict_disparity(rect_l, rect_r)
            latency = (time.time() - t0) * 1000

            depth = disparity_to_depth(
                disp, rectifier.focal_length, rectifier.baseline,
                min_depth=sd_cfg.get("min_depth", 0.05),
                max_depth=sd_cfg.get("max_depth", 10.0),
            )
            depth_color = colorize_depth(depth, sd_cfg.get("max_depth", 10.0))

            # 寫入共享狀態
            self._state.live_rect_left = rect_l
            self._state.live_depth = depth
            self._state.live_depth_color = depth_color

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self._state.live_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            self._state.live_latency_ms = latency

    def _update_live_display(self):
        """定時更新即時模式的顯示（主線程）"""
        if not self._live_running:
            return

        state = self._state
        rgb_cw = self._rgb_canvas.winfo_width()
        rgb_ch = self._rgb_canvas.winfo_height()
        dep_cw = self._depth_canvas.winfo_width()
        dep_ch = self._depth_canvas.winfo_height()

        if state.live_rect_left is not None and rgb_cw > 2 and rgb_ch > 2:
            img = state.live_rect_left
            h, w = img.shape[:2]
            self._display_img_size = (w, h)
            # 套用量測 + workspace overlay
            display = self._apply_measurement_overlay(img, state.live_depth, state.Q)
            display = self._apply_workspace_overlay(display, state.live_depth, self._get_K())
            photo = self._fit_to_canvas(display, rgb_cw, rgb_ch)
            self._photo_rgb = photo
            self._rgb_canvas.delete("all")
            self._rgb_canvas.create_image(rgb_cw // 2, rgb_ch // 2, image=photo)
            self._rgb_canvas_size = (rgb_cw, rgb_ch)

        if state.live_depth_color is not None and dep_cw > 2 and dep_ch > 2:
            display = self._apply_measurement_overlay(
                state.live_depth_color, state.live_depth, state.Q,
            )
            display = self._apply_workspace_overlay(display, state.live_depth, self._get_K())
            photo = self._fit_to_canvas(display, dep_cw, dep_ch)
            self._photo_depth = photo
            self._depth_canvas.delete("all")
            self._depth_canvas.create_image(dep_cw // 2, dep_ch // 2, image=photo)
            self._depth_canvas_size = (dep_cw, dep_ch)

        # 底部狀態列
        depth_at_cursor = ""
        if state.measure_a is not None and state.live_depth is not None and state.Q is not None:
            from phase3_stereo_depth.src.depth_utils import pixel_to_3d
            pt = pixel_to_3d(state.measure_a[0], state.measure_a[1], state.live_depth, state.Q)
            if pt is not None:
                depth_at_cursor = f"Depth: {pt[2]:.2f}m"

        measure_text = ""
        if state.measure_dist is not None:
            measure_text = f"Dist: {state.measure_dist * 100:.1f}cm"
        elif state.measure_a is not None and state.measure_b is None:
            measure_text = "Click 2nd point..."

        parts = [
            f"FPS: {state.live_fps:.1f}",
            f"Latency: {state.live_latency_ms:.0f}ms",
        ]
        if depth_at_cursor:
            parts.append(depth_at_cursor)
        if measure_text:
            parts.append(measure_text)
        self._info_var.set(" | ".join(parts))

        self.root.after(self._LIVE_DISPLAY_INTERVAL, self._update_live_display)

    # ══════════════════════════════════════════════════════════════
    # Capture — 凍結當前幀
    # ══════════════════════════════════════════════════════════════

    def _do_capture(self):
        if self._state.stage != "LIVE":
            return
        state = self._state

        # 凍結
        state.frozen_rgb = state.live_rect_left.copy() if state.live_rect_left is not None else None
        state.frozen_depth = state.live_depth.copy() if state.live_depth is not None else None
        state.frozen_depth_color = state.live_depth_color.copy() if state.live_depth_color is not None else None

        if state.frozen_rgb is None:
            self._status_var.set("No live frame to capture")
            return

        # 停止即時推理（但不釋放相機，以便 Back to Live）
        self._live_running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=3.0)
            self._live_thread = None

        # 清除前一輪結果
        state.mask = None
        state.result_image = None
        state.grasp_result = None
        state.measure_a = None
        state.measure_b = None
        state.measure_dist = None

        h, w = state.frozen_rgb.shape[:2]
        self._display_img_size = (w, h)
        self._set_stage("CAPTURED")
        self._redraw_all()
        self._status_var.set(f"Captured: {w}x{h}")

    def _back_to_live(self):
        """回到即時模式"""
        stage = self._state.stage
        if stage not in ("CAPTURED", "SEGMENTED", "GRASPED"):
            return

        # 清除凍結數據和結果
        self._state.frozen_rgb = None
        self._state.frozen_depth = None
        self._state.frozen_depth_color = None
        self._state.mask = None
        self._state.result_image = None
        self._state.grasp_result = None
        self._state.measure_a = None
        self._state.measure_b = None
        self._state.measure_dist = None

        # 清除結果面板
        self._result_canvas.delete("all")
        self._photo_result = None

        # 如果相機仍在，重啟即時推理
        if self._state.reader_l is not None and self._state.reader_r is not None:
            self._state.stereo_inf.set_fast_mode(True)
            self._live_running = True
            self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
            self._live_thread.start()
            self._set_stage("LIVE")
            self._status_var.set("Back to live")
            self._update_live_display()
        else:
            # 相機已釋放，需要重新 Start Live
            self._set_stage("IDLE")
            self._status_var.set("Cameras released — press Start Live")

    # ══════════════════════════════════════════════════════════════
    # 離線 Fallback — Open RGB / Open Depth
    # ══════════════════════════════════════════════════════════════

    def _open_rgb(self):
        if self._state.stage == "LOADING" or self._state.stage == "LIVE":
            return
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status_var.set(f"Failed to load: {path}")
            return

        self._state.frozen_rgb = img
        self._state.mask = None
        self._state.result_image = None
        self._state.grasp_result = None
        self._state.measure_a = None
        self._state.measure_b = None
        self._state.measure_dist = None

        h, w = img.shape[:2]
        self._display_img_size = (w, h)

        # 如果有 depth 就進入 CAPTURED，否則也進入 CAPTURED（等 depth）
        if self._state.stage not in ("CAPTURED", "SEGMENTED", "GRASPED"):
            self._set_stage("CAPTURED")
        self._redraw_all()
        self._status_var.set(f"Loaded RGB: {os.path.basename(path)} ({w}x{h})")
        self._update_button_states()

    def _open_depth(self):
        if self._state.stage == "LOADING" or self._state.stage == "LIVE":
            return
        path = filedialog.askopenfilename(
            initialdir=os.path.join(_ROOT, "phase3_stereo_depth", "outputs", "stereo_depth"),
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            depth = np.load(path).astype(np.float32)
        except Exception as exc:
            self._status_var.set(f"Failed to load depth: {exc}")
            return

        self._state.frozen_depth = depth
        # 生成 depth colormap
        from phase3_stereo_depth.src.depth_utils import colorize_depth
        max_depth = self._stereo_cfg.get("max_depth", 10.0)
        self._state.frozen_depth_color = colorize_depth(depth, max_depth)

        h, w = depth.shape[:2]
        if self._state.stage not in ("CAPTURED", "SEGMENTED", "GRASPED"):
            self._set_stage("CAPTURED")
        self._redraw_all()
        self._status_var.set(f"Loaded Depth: {os.path.basename(path)} ({w}x{h})")
        self._update_button_states()

    # ══════════════════════════════════════════════════════════════
    # Mask 載入
    # ══════════════════════════════════════════════════════════════

    def _open_mask(self):
        path = filedialog.askopenfilename(
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self._state.mask = np.load(path).astype(bool)
            self._redraw_all()
            self._set_stage("SEGMENTED")
            self._status_var.set(f"Mask loaded: {self._state.mask.shape}")
        except Exception as exc:
            self._status_var.set(f"Failed to load mask: {exc}")

    # ══════════════════════════════════════════════════════════════
    # T_cam2arm 載入
    # ══════════════════════════════════════════════════════════════

    def _load_t_cam2arm(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("NumPy", "*.npy"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            if path.endswith(".npy"):
                self._state.T_cam2arm = np.load(path).astype(np.float64)
            else:
                with open(path) as f:
                    data = json.load(f)
                self._state.T_cam2arm = np.array(data, dtype=np.float64)
            self._t_label.set(f"T_cam2arm: {os.path.basename(path)}")
        except Exception as exc:
            self._status_var.set(f"Failed to load T: {exc}")

    # ══════════════════════════════════════════════════════════════
    # SAM3 分割
    # ══════════════════════════════════════════════════════════════

    def _run_segment(self):
        state = self._state
        if state.sam3_skill is None:
            self._status_var.set("SAM3 model not loaded")
            return
        if state.frozen_rgb is None:
            self._status_var.set("No image — capture or open an RGB first")
            return
        prompt = self._prompt_var.get().strip()
        if not prompt:
            self._status_var.set("Enter a SAM3 prompt first")
            return

        self._seg_btn.config(state=tk.DISABLED)
        self._status_var.set(f'Segmenting: "{prompt}" ...')

        def _infer():
            try:
                result = state.sam3_skill.segment(state.frozen_rgb, prompt)
                self.root.after(0, self._on_segment_done, result, None)
            except Exception as exc:
                self.root.after(0, self._on_segment_done, None, str(exc))

        threading.Thread(target=_infer, daemon=True).start()

    def _on_segment_done(self, result, err):
        if err:
            self._status_var.set(f"SAM3 error: {err}")
            self._update_button_states()
            return
        if len(result.masks) == 0:
            self._status_var.set("No masks found")
            self._state.mask = None
            self._update_button_states()
        else:
            self._state.mask = result.best_mask
            self._set_stage("SEGMENTED")
            self._status_var.set(
                f"SAM3: {len(result.masks)} mask(s), best={result.best_score:.3f}"
            )
        self._redraw_all()

    # ══════════════════════════════════════════════════════════════
    # CapturePoint (GraspGen) 推理
    # ══════════════════════════════════════════════════════════════

    def _get_K(self) -> np.ndarray | None:
        """取得 K 矩陣：校正後 K 優先，否則用下拉選的 intrinsics"""
        if self._state.K_rect is not None:
            return self._state.K_rect
        cam_name = self._cam_var.get()
        if cam_name in self._state.intrinsics:
            return self._state.intrinsics[cam_name]
        return None

    def _run_capture_point(self):
        state = self._state
        if state.capture_skill is None:
            self._status_var.set("CapturePoint model not loaded")
            return
        if state.frozen_rgb is None:
            self._status_var.set("No RGB image")
            return
        if state.frozen_depth is None:
            self._status_var.set("No depth map")
            return
        if state.mask is None:
            self._status_var.set("No mask — run SAM3 or load a mask first")
            return
        K = self._get_K()
        if K is None:
            self._status_var.set("No camera intrinsics selected")
            return

        # Debug: 檢查尺寸一致性
        rgb_h, rgb_w = state.frozen_rgb.shape[:2]
        dep_h, dep_w = state.frozen_depth.shape[:2]
        mask_h, mask_w = state.mask.shape[:2]
        print(f"[CapturePoint] RGB: {rgb_w}x{rgb_h}, Depth: {dep_w}x{dep_h}, "
              f"Mask: {mask_w}x{mask_h}")
        if (rgb_h, rgb_w) != (dep_h, dep_w):
            print(f"[CapturePoint] WARNING: RGB and Depth size mismatch!")
        if (rgb_h, rgb_w) != (mask_h, mask_w):
            print(f"[CapturePoint] WARNING: RGB and Mask size mismatch!")
        print(f"[CapturePoint] K:\n{K}")
        print(f"[CapturePoint] T_cam2arm:\n{state.T_cam2arm}")
        print(f"[CapturePoint] Mask pixels: {state.mask.sum()}, "
              f"Depth valid in mask: {(state.frozen_depth[state.mask] > 0).sum()}")

        self._capture_point_btn.config(state=tk.DISABLED)
        self._status_var.set("Running GraspGen ...")

        def _infer():
            try:
                result = state.capture_skill.capture(
                    state.frozen_rgb, state.frozen_depth, state.mask, K, state.T_cam2arm,
                )
                self.root.after(0, self._on_capture_point_done, result, None)
            except Exception as exc:
                self.root.after(0, self._on_capture_point_done, None, str(exc))

        threading.Thread(target=_infer, daemon=True).start()

    def _on_capture_point_done(self, result, err):
        if err:
            self._status_var.set(f"CapturePoint error: {err}")
            self._update_button_states()
            return

        # Debug: 印出 grasp 結果
        print(f"[CapturePoint] === RESULT ===")
        print(f"  num_candidates: {result.num_candidates}")
        print(f"  grasp_score: {result.grasp_score:.4f}")
        print(f"  grasp_pixel: {result.grasp_pixel}")
        print(f"  grasp_width: {result.grasp_width * 1000:.1f} mm")
        print(f"  cropped_cloud_size: {result.cropped_cloud_size}")
        print(f"  pose_arm:\n{result.pose_arm}")
        # 檢查 grasp_pixel 是否在 mask 內
        state = self._state
        gp = result.grasp_pixel
        if state.mask is not None and result.num_candidates > 0:
            h, w = state.mask.shape[:2]
            u, v = gp
            in_bounds = 0 <= u < w and 0 <= v < h
            in_mask = state.mask[v, u] if in_bounds else False
            print(f"  grasp_pixel in_bounds: {in_bounds}, in_mask: {in_mask}")
            if not in_mask:
                # 找 mask 的質心對比
                ys, xs = np.where(state.mask)
                if len(xs) > 0:
                    mask_cx, mask_cy = int(xs.mean()), int(ys.mean())
                    print(f"  mask centroid: ({mask_cx}, {mask_cy}), "
                          f"grasp pixel offset: ({u - mask_cx}, {v - mask_cy})")

        self._state.result_image = result.annotated_image
        self._state.grasp_result = result
        self._set_stage("GRASPED")
        self._redraw_all()

        if result.num_candidates == 0:
            self._status_var.set("No grasp candidates found")
            self._info_var.set("")
            return

        pos = result.pose_arm[:3, 3]
        self._status_var.set(
            f"Grasp found! Score: {result.grasp_score:.3f}, "
            f"Candidates: {result.num_candidates}"
        )
        self._info_var.set(
            f"Score: {result.grasp_score:.3f}  |  "
            f"Width: {result.grasp_width * 1000:.1f}mm  |  "
            f"Pos(arm): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  |  "
            f"Pixel: {result.grasp_pixel}  |  "
            f"Cloud: {result.cropped_cloud_size} pts"
        )

    # ══════════════════════════════════════════════════════════════
    # 點擊量測
    # ══════════════════════════════════════════════════════════════

    def _canvas_to_image_coords(self, event, canvas) -> tuple[int, int] | None:
        """顯示座標 → 原始圖像座標"""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return None

        iw, ih = self._display_img_size
        scale = min(cw / iw, ch / ih)
        disp_w = int(iw * scale)
        disp_h = int(ih * scale)

        # 圖像在 canvas 中居中
        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2

        x = event.x - ox
        y = event.y - oy
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None

        u = int(x * iw / disp_w)
        v = int(y * ih / disp_h)
        u = max(0, min(u, iw - 1))
        v = max(0, min(v, ih - 1))
        return (u, v)

    def _handle_click(self, event, canvas):
        """處理 RGB 或 Depth canvas 上的點擊"""
        state = self._state
        depth = state.live_depth if state.stage == "LIVE" else state.frozen_depth
        Q = state.Q
        if depth is None or Q is None:
            return

        pt = self._canvas_to_image_coords(event, canvas)
        if pt is None:
            return

        if state.measure_a is None or (state.measure_a is not None and state.measure_b is not None):
            # 第一點或重置
            state.measure_a = pt
            state.measure_b = None
            state.measure_dist = None
        else:
            # 第二點
            from phase3_stereo_depth.src.depth_utils import measure_distance_3d
            state.measure_b = pt
            state.measure_dist = measure_distance_3d(
                state.measure_a, state.measure_b, depth, Q,
            )

        # 非 LIVE 模式需要手動重繪
        if state.stage != "LIVE":
            self._redraw_all()
            # 更新底部狀態列
            if state.measure_dist is not None:
                self._info_var.set(f"Dist: {state.measure_dist * 100:.1f}cm")
            elif state.measure_a is not None and state.measure_b is None:
                self._info_var.set("Click 2nd point...")

    def _on_rgb_click(self, event):
        self._handle_click(event, self._rgb_canvas)

    def _on_depth_click(self, event):
        self._handle_click(event, self._depth_canvas)

    # ══════════════════════════════════════════════════════════════
    # 顯示工具
    # ══════════════════════════════════════════════════════════════

    def _fit_to_canvas(self, cv_img: np.ndarray, cw: int, ch: int) -> ImageTk.PhotoImage:
        """將 BGR 圖像 resize 到 canvas 尺寸並轉為 PhotoImage"""
        h, w = cv_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _apply_measurement_overlay(
        self, img: np.ndarray, depth: np.ndarray | None, Q: np.ndarray | None,
    ) -> np.ndarray:
        """在影像上疊加量測標記"""
        state = self._state
        if state.measure_a is None:
            return img
        from phase3_stereo_depth.src.depth_utils import draw_measurement_overlay
        return draw_measurement_overlay(
            img, state.measure_a, state.measure_b,
            distance=state.measure_dist,
            depth=depth, Q=Q,
        )

    def _on_workspace_toggle(self):
        """Checkbox 切換時重繪（非 LIVE 模式）"""
        if self._state.stage not in ("LIVE", "LOADING"):
            self._redraw_all()

    def _apply_workspace_overlay(
        self, img: np.ndarray, depth: np.ndarray | None, K: np.ndarray | None,
    ) -> np.ndarray:
        """在影像上將 workspace 外的像素變暗，workspace 內保持原色

        對每個有效 depth 像素，用 K 反投影到 3D → T_cam2arm → 檢查是否在 limits 內。
        workspace 外的區域半透明變暗（紅色色調）。
        """
        if not self._show_ws_var.get() or depth is None or K is None:
            return img
        state = self._state
        ws = state.workspace_limits
        if not ws:
            return img

        h, w = depth.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 建構每個像素的 3D 座標（相機座標系）
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        zs = depth.astype(np.float32)
        valid = zs > 0

        xs = np.where(valid, (us - cx) * zs / fx, 0)
        ys = np.where(valid, (vs - cy) * zs / fy, 0)

        # cam → arm
        T = state.T_cam2arm
        R = T[:3, :3]
        t = T[:3, 3]
        pts_cam = np.stack([xs, ys, zs], axis=-1)  # (H, W, 3)
        pts_arm = np.einsum("ij,hwj->hwi", R, pts_cam) + t

        # 檢查 workspace limits
        xl, xh = ws.get("x", [-0.5, 0.5])
        yl, yh = ws.get("y", [-0.5, 0.5])
        zl, zh = ws.get("z", [0.0, 0.6])
        in_ws = (
            valid
            & (pts_arm[..., 0] >= xl) & (pts_arm[..., 0] <= xh)
            & (pts_arm[..., 1] >= yl) & (pts_arm[..., 1] <= yh)
            & (pts_arm[..., 2] >= zl) & (pts_arm[..., 2] <= zh)
        )

        out = img.copy()
        # workspace 外 → 暗紅色調
        outside = valid & ~in_ws
        out[outside] = (out[outside] * 0.3 + np.array([0, 0, 60], dtype=np.uint8)).clip(
            0, 255,
        ).astype(np.uint8)
        # 無效深度 → 稍微變暗
        out[~valid] = (out[~valid] * 0.5).astype(np.uint8)

        # workspace 邊界上畫計數
        n_in = int(in_ws.sum())
        n_valid = int(valid.sum())
        cv2.putText(
            out, f"WS: {n_in}/{n_valid} px", (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
        )
        return out

    def _apply_mask_overlay(self, img: np.ndarray) -> np.ndarray:
        """在 RGB 圖上疊加綠色 mask"""
        state = self._state
        if state.mask is None:
            return img
        display = img.copy()
        overlay = display.copy()
        overlay[state.mask] = [0, 200, 0]
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        return display

    def _redraw_all(self):
        """根據當前狀態重繪所有面板"""
        state = self._state
        stage = state.stage

        # ── RGB 面板 ──
        self._rgb_canvas.delete("all")
        rgb_cw = self._rgb_canvas.winfo_width()
        rgb_ch = self._rgb_canvas.winfo_height()

        if stage in ("CAPTURED", "SEGMENTED", "GRASPED") and state.frozen_rgb is not None:
            display = state.frozen_rgb
            if stage in ("SEGMENTED", "GRASPED"):
                display = self._apply_mask_overlay(display)
            depth_for_measure = state.frozen_depth
            display = self._apply_measurement_overlay(display, depth_for_measure, state.Q)
            display = self._apply_workspace_overlay(display, state.frozen_depth, self._get_K())
            if rgb_cw > 2 and rgb_ch > 2:
                photo = self._fit_to_canvas(display, rgb_cw, rgb_ch)
                self._photo_rgb = photo
                self._rgb_canvas.create_image(rgb_cw // 2, rgb_ch // 2, image=photo)
                self._rgb_canvas_size = (rgb_cw, rgb_ch)
        elif stage == "IDLE":
            if rgb_cw > 2 and rgb_ch > 2:
                self._rgb_canvas.create_text(
                    rgb_cw // 2, rgb_ch // 2,
                    text="Press [Start Live] or [Open RGB]",
                    fill="#888888", font=("sans-serif", 13),
                )

        # ── Depth 面板 ──
        self._depth_canvas.delete("all")
        dep_cw = self._depth_canvas.winfo_width()
        dep_ch = self._depth_canvas.winfo_height()

        if stage in ("CAPTURED", "SEGMENTED", "GRASPED") and state.frozen_depth_color is not None:
            display = self._apply_measurement_overlay(
                state.frozen_depth_color, state.frozen_depth, state.Q,
            )
            display = self._apply_workspace_overlay(display, state.frozen_depth, self._get_K())
            if dep_cw > 2 and dep_ch > 2:
                photo = self._fit_to_canvas(display, dep_cw, dep_ch)
                self._photo_depth = photo
                self._depth_canvas.create_image(dep_cw // 2, dep_ch // 2, image=photo)
                self._depth_canvas_size = (dep_cw, dep_ch)

        # ── Result 面板 ──
        self._result_canvas.delete("all")
        res_cw = self._result_canvas.winfo_width()
        res_ch = self._result_canvas.winfo_height()

        if stage == "GRASPED" and state.result_image is not None:
            if res_cw > 2 and res_ch > 2:
                photo = self._fit_to_canvas(state.result_image, res_cw, res_ch)
                self._photo_result = photo
                self._result_canvas.create_image(res_cw // 2, res_ch // 2, image=photo)

    # ══════════════════════════════════════════════════════════════
    # 3D 可視化
    # ══════════════════════════════════════════════════════════════

    def _view_3d_grasp(self):
        """開啟 Open3D 視窗，顯示物體點雲 + grasp pose + 夾爪線框"""
        state = self._state
        if state.grasp_result is None or state.grasp_result.num_candidates == 0:
            self._status_var.set("No grasp result to visualize")
            return
        if state.frozen_depth is None or state.frozen_rgb is None:
            self._status_var.set("No depth/RGB data")
            return

        K = self._get_K()
        if K is None:
            self._status_var.set("No intrinsics for 3D view")
            return

        self._status_var.set("Opening 3D viewer...")

        def _run_viewer():
            try:
                import open3d as o3d
            except ImportError:
                self.root.after(0, self._status_var.set, "open3d not installed")
                return

            result = state.grasp_result
            pose_arm = result.pose_arm
            T_cam2arm = state.T_cam2arm

            # 1. 物體點雲（mask 區域）
            from phase5_vlm_planning.skills.skill_capture_point import pointcloud_cropper
            pts_cam, colors = pointcloud_cropper.mask_to_pointcloud(
                state.frozen_depth, K, state.mask, state.frozen_rgb,
            )
            pts_arm = pointcloud_cropper.transform_points(pts_cam, T_cam2arm)

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pts_arm.astype(np.float64))
            if colors is not None:
                pcd_obj.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            # 2. 場景點雲（非 mask 區域，半透明灰）
            scene_mask = ~state.mask & (state.frozen_depth > 0)
            pts_scene_cam, _ = pointcloud_cropper.mask_to_pointcloud(
                state.frozen_depth, K, scene_mask,
            )
            pts_scene_arm = pointcloud_cropper.transform_points(pts_scene_cam, T_cam2arm)
            # 降採樣場景
            if len(pts_scene_arm) > 20000:
                idx = np.random.choice(len(pts_scene_arm), 20000, replace=False)
                pts_scene_arm = pts_scene_arm[idx]

            pcd_scene = o3d.geometry.PointCloud()
            pcd_scene.points = o3d.utility.Vector3dVector(pts_scene_arm.astype(np.float64))
            pcd_scene.paint_uniform_color([0.5, 0.5, 0.5])

            # 3. 夾爪線框（用 control points 方式）
            gripper_width = self._gripper_width
            gripper_depth = 0.195  # robotiq_2f_140 depth
            gripper_lines = self._make_gripper_lineset(
                pose_arm, gripper_width, gripper_depth,
            )

            # 4. Pre-grasp → TCP approach 路徑
            tcp = pose_arm[:3, 3]
            approach_dir = pose_arm[:3, 2]  # z 軸 = approach direction
            pre_grasp_dist = self._pre_grasp_distance
            pre_grasp = tcp - approach_dir * pre_grasp_dist

            # pre-grasp → TCP 線段（紅色）
            approach_line = o3d.geometry.LineSet()
            approach_line.points = o3d.utility.Vector3dVector(
                np.array([pre_grasp, tcp], dtype=np.float64),
            )
            approach_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            approach_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])

            # 5. Pre-grasp 球（黃色）
            pre_grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            pre_grasp_sphere.translate(pre_grasp)
            pre_grasp_sphere.paint_uniform_color([1.0, 1.0, 0.0])

            # 6. TCP 球（紅色）
            tcp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            tcp_sphere.translate(tcp)
            tcp_sphere.paint_uniform_color([1.0, 0.0, 0.0])

            # 7. 接觸點球（綠色）
            from phase5_vlm_planning.skills.skill_capture_point.grasp_visualizer import (
                compute_contact_point,
            )
            contact = compute_contact_point(pose_arm, pts_arm)
            contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            contact_sphere.translate(contact)
            contact_sphere.paint_uniform_color([0.0, 1.0, 0.0])

            # 8. 座標軸
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # 9. Workspace box 線框
            ws_lines = self._make_workspace_lineset(state.workspace_limits)

            geometries = [
                pcd_scene, pcd_obj,
                gripper_lines, approach_line,
                pre_grasp_sphere, tcp_sphere, contact_sphere,
                axes,
            ]
            if ws_lines is not None:
                geometries.append(ws_lines)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Grasp 3D View", width=1280, height=960)
            opt = vis.get_render_option()
            opt.point_size = 3.0
            opt.background_color = np.array([0.1, 0.1, 0.1])

            for g in geometries:
                vis.add_geometry(g)

            # 相機朝向物體
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_lookat(contact.astype(np.float64))

            vis.run()
            vis.destroy_window()
            self.root.after(0, self._status_var.set, "3D viewer closed")

        threading.Thread(target=_run_viewer, daemon=True).start()

    @staticmethod
    def _make_gripper_lineset(
        pose: np.ndarray, width: float, depth: float,
    ):
        """用 grasp pose 建構夾爪線框

        GraspGen 慣例：z+ = approach, x = 夾爪張開方向
        """
        import open3d as o3d

        hw = width / 2
        # 夾爪控制點（gripper local frame）
        #   base (z=0) → finger tip (z=depth)
        #   左右 finger 在 x = ±hw
        pts_local = np.array([
            # 基座矩形
            [-hw * 0.3, 0, 0],  # 0: base left
            [hw * 0.3, 0, 0],   # 1: base right
            # 指根
            [-hw, 0, depth * 0.5],  # 2: left finger root
            [hw, 0, depth * 0.5],   # 3: right finger root
            # 指尖
            [-hw, 0, depth],  # 4: left finger tip
            [hw, 0, depth],   # 5: right finger tip
            # 頂部連接
            [0, 0, 0],  # 6: base center (TCP)
        ], dtype=np.float64)

        edges = [
            [0, 1],  # 基座
            [0, 2], [1, 3],  # 基座 → 指根
            [2, 4], [3, 5],  # 指根 → 指尖
            [6, 0], [6, 1],  # TCP → 基座
        ]

        # 轉到 world/arm 座標系
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = (R @ pts_local.T).T + t

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_world)
        ls.lines = o3d.utility.Vector2iVector(edges)
        # 亮綠色
        ls.colors = o3d.utility.Vector3dVector(
            [[0.0, 1.0, 0.0]] * len(edges),
        )
        return ls

    @staticmethod
    def _make_workspace_lineset(ws: dict):
        """Workspace bounding box 線框"""
        if not ws:
            return None
        import open3d as o3d

        xl, xh = ws.get("x", [-0.5, 0.5])
        yl, yh = ws.get("y", [-0.5, 0.5])
        zl, zh = ws.get("z", [0.0, 0.6])

        corners = np.array([
            [xl, yl, zl], [xh, yl, zl], [xh, yh, zl], [xl, yh, zl],
            [xl, yl, zh], [xh, yl, zh], [xh, yh, zh], [xl, yh, zh],
        ], dtype=np.float64)

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector(
            [[1.0, 0.6, 0.0]] * len(edges),  # 橘色
        )
        return ls

    # ══════════════════════════════════════════════════════════════
    # 保存結果
    # ══════════════════════════════════════════════════════════════

    def _save_result(self):
        if self._state.result_image is None:
            self._status_var.set("No result to save")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            cv2.imwrite(path, self._state.result_image)
            self._status_var.set(f"Saved: {path}")

    # ══════════════════════════════════════════════════════════════
    # 清理
    # ══════════════════════════════════════════════════════════════

    def destroy(self):
        self._stop_live()


def main():
    root = tk.Tk()
    gui = CapturePointPipelineGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.destroy(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
