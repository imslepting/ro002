"""ARM-VLM Agent 互動測試 GUI

VLM 作為決策者，通過 tool-use API 自主決定調用 SAM3、CapturePoint 等工具。
支持 OpenAI (GPT-4o) 和 Claude Code CLI 雙 Provider。

用法:
    conda run -n ro002 python phase5_vlm_planning/test_vlm_agent.py

狀態機:
    LOADING → IDLE → LIVE → AGENT_RUNNING → PLAN_READY
                      ↑                         │
                      └─────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger(__name__)

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.camera_manager import CameraReader


# ── 工具函數 ──

def _load_intrinsics() -> dict:
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
    path = os.path.join(_ROOT, "config", "settings.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── 共享狀態容器 ──

class PipelineState:
    def __init__(self):
        self.stage: str = "LOADING"

        # Phase3 模型
        self.rectifier = None
        self.stereo_inf = None

        # Phase5 模型
        self.sam3_skill = None
        self.capture_skill = None

        # VLM
        self.vlm_client = None

        # 相機
        self.reader_l: CameraReader | None = None
        self.reader_r: CameraReader | None = None

        # 即時數據
        self.live_rect_left: np.ndarray | None = None
        self.live_depth: np.ndarray | None = None
        self.live_depth_color: np.ndarray | None = None
        self.live_fps: float = 0.0
        self.live_latency_ms: float = 0.0

        # 校正數據
        self.K_rect: np.ndarray | None = None
        self.Q: np.ndarray | None = None

        # Agent 結果
        self.agent_result = None
        self.result_image: np.ndarray | None = None

        # 配置
        self.workspace_limits: dict = {}
        self.T_cam2arm: np.ndarray = np.eye(4)
        self.intrinsics: dict = {}


# ── 主 GUI 類 ──

class VLMAgentGUI:
    """ARM-VLM Agent GUI"""

    _LIVE_INTERVAL = 50  # ms

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ARM-VLM Agent")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)

        self._state = PipelineState()
        self._state.intrinsics = _load_intrinsics()
        self._settings = _load_settings()
        self._stereo_cfg = self._settings.get("stereo_depth", {})
        cp_cfg = self._settings.get("skill_capture_point", {})
        self._state.workspace_limits = cp_cfg.get("workspace_limits", {
            "x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 2.0],
        })
        self._gripper_width = cp_cfg.get("gripper_width", 0.136)
        self._pre_grasp_distance = cp_cfg.get("pre_grasp_distance", 0.10)

        # VLM config
        vlm_cfg = self._settings.get("vlm", {})
        self._vlm_provider = vlm_cfg.get("provider", "openai")
        self._vlm_openai_model = vlm_cfg.get("openai_model", "gpt-4o")
        self._vlm_claude_model = vlm_cfg.get("claude_model", "claude-sonnet-4-20250514")

        # Live loop
        self._live_running = False
        self._live_thread: threading.Thread | None = None

        # Agent loop
        self._agent_loop = None
        self._agent_thread: threading.Thread | None = None

        # 凍結快照（agent 用）
        self._snapshot_rgb: np.ndarray | None = None
        self._snapshot_depth: np.ndarray | None = None

        # PhotoImage 引用
        self._photo_rgb: ImageTk.PhotoImage | None = None
        self._photo_depth: ImageTk.PhotoImage | None = None
        self._photo_seg: ImageTk.PhotoImage | None = None
        self._photo_grasp: ImageTk.PhotoImage | None = None

        # 追蹤 executor 結果（用於更新 segment/grasp canvas）
        self._last_sam3_result = None
        self._last_capture_result = None

        # 深度點擊查詢狀態
        self._depth_markers: list[tuple[int, int, float]] = []  # (img_x, img_y, depth_val)

        # 顯示尺寸
        self._display_img_size: tuple[int, int] = (640, 480)

        self._build_ui()
        self._update_button_states()

        # 顯示佔位文字
        self.root.update_idletasks()
        self._show_placeholder("Loading models...\n\nPlease wait")

        self._load_models_async()

    # ══════════════════════════════════════════════════════════════
    # UI 構建
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── 頂部狀態列 ──
        top = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        top.pack(fill=tk.X)

        ttk.Label(top, text="ARM-VLM Agent", font=("Helvetica", 14, "bold")).pack(
            side=tk.LEFT,
        )
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._status_var = tk.StringVar(value="Loading models...")
        ttk.Label(top, textvariable=self._status_var, foreground="gray").pack(
            side=tk.LEFT,
        )

        self._fps_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._fps_var, foreground="#666").pack(
            side=tk.RIGHT,
        )

        # ── 主體：左右兩欄 ──
        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── 左欄 (45%) ──
        left = ttk.Frame(body, padding=4, width=500)
        body.add(left, weight=45)

        # 2x2 圖像網格
        img_grid = ttk.Frame(left)
        img_grid.pack(fill=tk.BOTH, expand=True)
        img_grid.columnconfigure(0, weight=1)
        img_grid.columnconfigure(1, weight=1)
        img_grid.rowconfigure(0, weight=1)
        img_grid.rowconfigure(1, weight=1)

        # RGB Canvas (0,0)
        rgb_frame = ttk.LabelFrame(img_grid, text="RGB", padding=2)
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))
        self._rgb_canvas = tk.Canvas(rgb_frame, bg="#2b2b2b")
        self._rgb_canvas.pack(fill=tk.BOTH, expand=True)

        # Depth Canvas (0,1) — 可點擊查詢深度值
        depth_frame = ttk.LabelFrame(img_grid, text="Depth", padding=2)
        depth_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=(0, 2))
        self._depth_canvas = tk.Canvas(depth_frame, bg="#2b2b2b")
        self._depth_canvas.pack(fill=tk.BOTH, expand=True)
        self._depth_info_var = tk.StringVar(value="Click depth to query")
        ttk.Label(depth_frame, textvariable=self._depth_info_var,
                  foreground="#888", font=("Consolas", 9)).pack(anchor=tk.W)
        self._depth_canvas.bind("<Button-1>", self._on_depth_click)
        self._depth_canvas.bind("<Button-3>", self._on_depth_clear)

        # Segment Canvas (1,0)
        seg_frame = ttk.LabelFrame(img_grid, text="Segment", padding=2)
        seg_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 2), pady=(2, 0))
        self._seg_canvas = tk.Canvas(seg_frame, bg="#2b2b2b")
        self._seg_canvas.pack(fill=tk.BOTH, expand=True)

        # Grasp Canvas (1,1)
        grasp_frame = ttk.LabelFrame(img_grid, text="Grasp", padding=2)
        grasp_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 0), pady=(2, 0))
        self._grasp_canvas = tk.Canvas(grasp_frame, bg="#2b2b2b")
        self._grasp_canvas.pack(fill=tk.BOTH, expand=True)

        # 左欄控制
        ctrl_left = ttk.Frame(left, padding=(0, 4, 0, 0))
        ctrl_left.pack(fill=tk.X)

        self._start_live_btn = ttk.Button(
            ctrl_left, text="Start Live", command=self._start_live,
        )
        self._start_live_btn.pack(side=tk.LEFT)

        self._stop_live_btn = ttk.Button(
            ctrl_left, text="Stop", command=self._stop_live_and_idle,
        )
        self._stop_live_btn.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(ctrl_left, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8,
        )

        # T_cam2arm
        self._t_label = tk.StringVar(value="T_cam2arm: config")
        ttk.Label(ctrl_left, textvariable=self._t_label, foreground="gray").pack(
            side=tk.LEFT,
        )
        self._load_t_btn = ttk.Button(
            ctrl_left, text="Load T...", command=self._load_t_cam2arm,
        )
        self._load_t_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl_left, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8,
        )

        # Camera K
        ttk.Label(ctrl_left, text="Cam:").pack(side=tk.LEFT)
        self._cam_var = tk.StringVar()
        cam_names = list(self._state.intrinsics.keys()) or ["(none)"]
        self._cam_combo = ttk.Combobox(
            ctrl_left, textvariable=self._cam_var, values=cam_names,
            state="readonly", width=8,
        )
        self._cam_combo.pack(side=tk.LEFT, padx=4)
        if cam_names and cam_names[0] != "(none)":
            self._cam_combo.current(0)

        ttk.Separator(ctrl_left, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8,
        )

        # Provider 選擇 + Load Agent
        ttk.Label(ctrl_left, text="Provider:").pack(side=tk.LEFT)
        self._provider_var = tk.StringVar(value=self._vlm_provider)
        self._provider_combo = ttk.Combobox(
            ctrl_left, textvariable=self._provider_var,
            values=["openai", "claude_code"],
            state="readonly", width=12,
        )
        self._provider_combo.pack(side=tk.LEFT, padx=4)

        self._load_agent_btn = ttk.Button(
            ctrl_left, text="Load Agent", command=self._load_agent,
        )
        self._load_agent_btn.pack(side=tk.LEFT, padx=(2, 0))

        self._agent_status_var = tk.StringVar(value="")
        ttk.Label(ctrl_left, textvariable=self._agent_status_var,
                  foreground="gray").pack(side=tk.LEFT, padx=4)

        # Agent 控制按鈕（放右邊）
        self._clear_btn = ttk.Button(
            ctrl_left, text="Clear", command=self._clear,
        )
        self._clear_btn.pack(side=tk.RIGHT)

        self._skip_btn = ttk.Button(
            ctrl_left, text="Skip Verify", command=self._skip_verify,
        )
        self._skip_btn.pack(side=tk.RIGHT, padx=(0, 4))

        self._cancel_btn = ttk.Button(
            ctrl_left, text="Cancel", command=self._cancel,
        )
        self._cancel_btn.pack(side=tk.RIGHT, padx=(0, 4))

        # ── 右欄 (55%) ──
        right = ttk.Frame(body, padding=4, width=600)
        body.add(right, weight=55)

        # Conversation Panel
        conv_frame = ttk.LabelFrame(right, text="Agent Conversation", padding=4)
        conv_frame.pack(fill=tk.BOTH, expand=True)

        self._conv_text = tk.Text(
            conv_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
            font=("Consolas", 10), relief=tk.FLAT,
            selectbackground="#264f78",
            width=60, height=20,
        )
        conv_scroll = ttk.Scrollbar(conv_frame, command=self._conv_text.yview)
        self._conv_text.config(yscrollcommand=conv_scroll.set)
        conv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._conv_text.pack(fill=tk.BOTH, expand=True)

        # 配置 tag 顏色
        self._conv_text.tag_configure("agent", foreground="#569cd6")
        self._conv_text.tag_configure("tool_call", foreground="#dcdcaa")
        self._conv_text.tag_configure("tool_result", foreground="#6a9955")
        self._conv_text.tag_configure("user", foreground="#ce9178")
        self._conv_text.tag_configure("error", foreground="#f44747")
        self._conv_text.tag_configure("info", foreground="#888888")

        # Result Summary
        summary_frame = ttk.LabelFrame(right, text="Result Summary", padding=4)
        summary_frame.pack(fill=tk.X, pady=(4, 0))

        self._summary_var = tk.StringVar(value="No result yet")
        ttk.Label(
            summary_frame, textvariable=self._summary_var,
            font=("Consolas", 10), wraplength=600, justify=tk.LEFT,
        ).pack(fill=tk.X)

        # Summary 按鈕
        summary_btns = ttk.Frame(summary_frame)
        summary_btns.pack(fill=tk.X, pady=(4, 0))

        self._save_plan_btn = ttk.Button(
            summary_btns, text="Save Plan", command=self._save_plan,
        )
        self._save_plan_btn.pack(side=tk.LEFT)

        self._view3d_btn = ttk.Button(
            summary_btns, text="View 3D", command=self._view_3d,
        )
        self._view3d_btn.pack(side=tk.LEFT, padx=(4, 0))

        # ── 底部：Task 輸入 ──
        task_frame = ttk.Frame(self.root, padding=(8, 4))
        task_frame.pack(fill=tk.X)

        ttk.Label(task_frame, text="Task:").pack(side=tk.LEFT)
        self._task_var = tk.StringVar()
        self._task_entry = ttk.Entry(
            task_frame, textvariable=self._task_var, font=("sans-serif", 12),
        )
        self._task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self._task_entry.bind("<Return>", lambda e: self._send_task())

        self._send_direct_btn = ttk.Button(
            task_frame, text="Run (Direct)",
            command=self._send_direct,
            style="Accent.TButton",
        )
        self._send_direct_btn.pack(side=tk.LEFT)

        self._send_btn = ttk.Button(
            task_frame, text="Send (VLM)", command=self._send_task,
        )
        self._send_btn.pack(side=tk.LEFT, padx=(4, 0))

        self._send_llm_btn = ttk.Button(
            task_frame, text="Send (LLM)",
            command=self._send_llm_fast,
        )
        self._send_llm_btn.pack(side=tk.LEFT, padx=(4, 0))

    # ══════════════════════════════════════════════════════════════
    # 狀態機 & 按鈕管理
    # ══════════════════════════════════════════════════════════════

    def _set_stage(self, stage: str):
        self._state.stage = stage
        self._update_button_states()

    def _update_button_states(self):
        stage = self._state.stage
        btn_map = {
            #                  start stop  send  cancel skip  save  v3d  clear loadT
            "LOADING":        (0,    0,    0,    0,     0,    0,    0,   0,    0),
            "IDLE":           (1,    0,    0,    0,     0,    0,    0,   0,    1),
            "LIVE":           (0,    1,    1,    0,     0,    0,    0,   1,    1),
            "AGENT_RUNNING":  (0,    0,    0,    1,     1,    0,    0,   0,    0),
            "PLAN_READY":     (0,    1,    1,    0,     0,    1,    1,   1,    1),
        }
        m = btn_map.get(stage, btn_map["LOADING"])
        self._start_live_btn.config(state=tk.NORMAL if m[0] else tk.DISABLED)
        self._stop_live_btn.config(state=tk.NORMAL if m[1] else tk.DISABLED)
        self._send_btn.config(state=tk.NORMAL if m[2] else tk.DISABLED)
        self._send_llm_btn.config(state=tk.NORMAL if m[2] else tk.DISABLED)
        self._send_direct_btn.config(state=tk.NORMAL if m[2] else tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL if m[3] else tk.DISABLED)
        self._skip_btn.config(state=tk.NORMAL if m[4] else tk.DISABLED)
        self._save_plan_btn.config(state=tk.NORMAL if m[5] else tk.DISABLED)
        self._view3d_btn.config(state=tk.NORMAL if m[6] else tk.DISABLED)
        self._clear_btn.config(state=tk.NORMAL if m[7] else tk.DISABLED)
        self._load_t_btn.config(state=tk.NORMAL if m[8] else tk.DISABLED)
        self._task_entry.config(state=tk.NORMAL if m[2] else tk.DISABLED)
        # Load Agent: 在非 LOADING 且非 AGENT_RUNNING 時可用
        self._load_agent_btn.config(
            state=tk.NORMAL if stage not in ("LOADING", "AGENT_RUNNING") else tk.DISABLED,
        )

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

            # T_cam2arm from config
            arm_cfg = self._settings.get("arm", {})
            t_data = arm_cfg.get("T_cam2arm")
            if t_data:
                self._state.T_cam2arm = np.array(t_data, dtype=np.float64)

            self.root.after(0, self._on_models_loaded, errors)

        threading.Thread(target=_load, daemon=True).start()

    def _show_placeholder(self, text: str = ""):
        """在所有 canvas 上顯示佔位文字"""
        canvases = [
            (self._rgb_canvas, text),
            (self._depth_canvas, text),
            (self._seg_canvas, "Segment"),
            (self._grasp_canvas, "Grasp"),
        ]
        for canvas, label in canvases:
            canvas.delete("all")
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw > 2 and ch > 2:
                canvas.create_text(
                    cw // 2, ch // 2, text=label,
                    fill="#888888", font=("sans-serif", 11),
                )

    def _on_models_loaded(self, errors: list[str]):
        self._set_stage("IDLE")
        self._show_placeholder("Press [Start Live]")
        if not errors:
            self._status_var.set("All models ready")
        else:
            msg = "; ".join(errors)
            self._status_var.set(f"Partial load: {msg}")
            log.warning(f"Model load errors: {msg}")

    # ══════════════════════════════════════════════════════════════
    # Live Loop
    # ══════════════════════════════════════════════════════════════

    def _start_live(self):
        if self._state.stage != "IDLE":
            return
        if self._state.rectifier is None or self._state.stereo_inf is None:
            self._status_var.set("Phase3 models not loaded")
            return

        cam_cfg = self._settings.get("cameras", {})
        cam_l_idx = cam_cfg.get("cam0", {}).get("index", 0)
        cam_r_idx = cam_cfg.get("cam1", {}).get("index", 1)

        self._state.reader_l = CameraReader(cam_l_idx)
        self._state.reader_r = CameraReader(cam_r_idx)
        self._state.reader_l.start()
        self._state.reader_r.start()

        self._state.stereo_inf.set_fast_mode(True)
        self._live_running = True
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

        self._set_stage("LIVE")
        self._status_var.set("Live stereo depth running...")
        self._update_live_display()

    def _live_loop(self):
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
            photo = self._fit_to_canvas(img, rgb_cw, rgb_ch)
            self._photo_rgb = photo
            self._rgb_canvas.delete("all")
            self._rgb_canvas.create_image(rgb_cw // 2, rgb_ch // 2, image=photo)

        if state.live_depth_color is not None and dep_cw > 2 and dep_ch > 2:
            photo = self._fit_to_canvas(state.live_depth_color, dep_cw, dep_ch)
            self._photo_depth = photo
            self._depth_canvas.delete("all")
            self._depth_canvas.create_image(dep_cw // 2, dep_ch // 2, image=photo)

        self._fps_var.set(
            f"FPS: {state.live_fps:.1f} | Latency: {state.live_latency_ms:.0f}ms"
        )
        self.root.after(self._LIVE_INTERVAL, self._update_live_display)

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

    def _stop_live_and_idle(self):
        self._stop_live()
        self._set_stage("IDLE")
        self._status_var.set("Stopped")

    # ══════════════════════════════════════════════════════════════
    # VLM Agent
    # ══════════════════════════════════════════════════════════════

    def _get_K(self) -> np.ndarray | None:
        if self._state.K_rect is not None:
            return self._state.K_rect
        cam_name = self._cam_var.get()
        if cam_name in self._state.intrinsics:
            return self._state.intrinsics[cam_name]
        return None

    def _create_vlm_client(self):
        from phase5_vlm_planning.src.vlm_client import create_vlm
        provider = self._provider_var.get()
        
        if provider == "openai":
            # 檢查是否使用 Azure OpenAI
            vlm_cfg = self._settings.get("vlm", {})
            azure_cfg = vlm_cfg.get("azure_openai", {})
            
            if azure_cfg.get("enabled", False):
                # 使用 Azure OpenAI
                endpoint = azure_cfg.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
                deployment = azure_cfg.get("deployment") or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                api_version = azure_cfg.get("api_version", "2024-12-01-preview")
                
                if not endpoint or not deployment:
                    raise ValueError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
                
                return create_vlm("openai", 
                                model=self._vlm_openai_model,
                                azure_endpoint=endpoint,
                                azure_deployment=deployment,
                                api_version=api_version)
            else:
                # 使用標準 OpenAI
                return create_vlm("openai", model=self._vlm_openai_model)
        elif provider == "claude_code":
            return create_vlm("claude_code", model=self._vlm_claude_model)
        else:
            return create_vlm(provider)

    def _load_agent(self):
        """預載 VLM client"""
        self._agent_status_var.set("Loading...")
        self._load_agent_btn.config(state=tk.DISABLED)

        def _do_load():
            try:
                vlm = self._create_vlm_client()
                self._state.vlm_client = vlm
                provider = self._provider_var.get()
                self.root.after(0, self._agent_status_var.set, f"Ready ({provider})")
            except Exception as e:
                self._state.vlm_client = None
                self.root.after(0, self._agent_status_var.set, f"Error: {e}")
            self.root.after(0, self._load_agent_btn.config, {"state": tk.NORMAL})

        threading.Thread(target=_do_load, daemon=True).start()

    def _send_task(self, use_images: bool = True):
        task_text = self._task_var.get().strip()
        if not task_text:
            self._status_var.set("Enter a task first")
            return

        state = self._state
        if state.sam3_skill is None or state.capture_skill is None:
            self._status_var.set("Models not loaded")
            return

        K = self._get_K()
        if K is None:
            self._status_var.set("No camera intrinsics")
            return

        # 凍結當前 live 幀
        if state.live_rect_left is not None:
            self._snapshot_rgb = state.live_rect_left.copy()
        if state.live_depth is not None:
            self._snapshot_depth = state.live_depth.copy()

        if self._snapshot_rgb is None:
            self._status_var.set("No scene image — start live first")
            return

        # 暫停 live loop
        self._live_running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=3.0)
            self._live_thread = None

        # 清空 conversation 和結果 canvas
        self._conv_text.config(state=tk.NORMAL)
        self._conv_text.delete("1.0", tk.END)
        self._conv_text.config(state=tk.DISABLED)
        self._summary_var.set("Agent running...")
        self._last_sam3_result = None
        self._last_capture_result = None
        self._seg_canvas.delete("all")
        self._grasp_canvas.delete("all")

        # 取得 VLM client（優先用預載的）
        if self._state.vlm_client is not None:
            vlm = self._state.vlm_client
        else:
            try:
                vlm = self._create_vlm_client()
            except Exception as e:
                self._status_var.set(f"VLM init error: {e}")
                self._resume_live()
                return

        # 創建 ToolExecutor
        from phase5_vlm_planning.src.agent_tools import ToolExecutor
        from phase5_vlm_planning.src.plan_serializer import PlanSerializer
        from phase5_vlm_planning.src.agent_loop import AgentLoop

        serializer = PlanSerializer(
            output_base=os.path.join(_ROOT, "phase5_vlm_planning", "outputs"),
        )
        executor = ToolExecutor(
            sam3_skill=state.sam3_skill,
            capture_skill=state.capture_skill,
            K_rect=K,
            T_cam2arm=state.T_cam2arm,
            plan_serializer=serializer,
        )
        executor.set_snapshot(self._snapshot_rgb, self._snapshot_depth)
        executor.set_no_images(not use_images)

        # 顯示快照在 RGB canvas
        self._show_image_on_canvas(
            self._rgb_canvas, self._snapshot_rgb, "_photo_rgb",
        )

        # 創建 AgentLoop
        self._agent_executor = executor
        agent = AgentLoop(
            vlm=vlm,
            tool_executor=executor,
            on_turn=lambda turn: self.root.after(0, self._on_turn, turn),
        )
        self._agent_loop = agent

        mode_label = "VLM" if use_images else "LLM (no images)"
        self._set_stage("AGENT_RUNNING")
        self._status_var.set(f"Agent running ({mode_label}): {task_text}")

        # 背景線程
        scene_img = self._snapshot_rgb if use_images else None

        def _run():
            result = agent.run(task_text, scene_img)
            self.root.after(0, self._on_agent_done, result)

        self._agent_thread = threading.Thread(target=_run, daemon=True)
        self._agent_thread.start()

    def _on_turn(self, turn):
        """每個 agent turn 的回調（主線程）"""
        from phase5_vlm_planning.src.agent_loop import AgentTurn

        self._conv_text.config(state=tk.NORMAL)

        if turn.role == "user":
            # 用戶消息
            for block in turn.content:
                if block.get("type") == "text":
                    self._append_conv(f"[User] {block['text']}\n", "user")
                elif block.get("type") == "image":
                    self._append_conv("[User] [場景圖片]\n", "user")

        elif turn.role == "assistant":
            for block in turn.content:
                if block.get("type") == "text":
                    text = block["text"].strip()
                    if text:
                        self._append_conv(f"[Agent] {text}\n", "agent")
                elif block.get("type") == "tool_use":
                    name = block["name"]
                    inp = block.get("input", {})
                    inp_str = ", ".join(f'{k}="{v}"' for k, v in inp.items())
                    self._append_conv(
                        f"  -> [{name}({inp_str})]\n", "tool_call",
                    )

        elif turn.role == "tool_result":
            for block in turn.content:
                if block.get("type") == "tool_result":
                    for sub in block.get("content", []):
                        if sub.get("type") == "text":
                            self._append_conv(
                                f"  <- {sub['text']}\n", "tool_result",
                            )
                        elif sub.get("type") == "image":
                            self._append_conv(
                                "  <- [圖片結果]\n", "tool_result",
                            )

            # 檢查 executor 狀態，更新 segment/grasp canvas
            if hasattr(self, "_agent_executor") and self._agent_executor:
                sam3 = self._agent_executor.sam3_result
                if sam3 is not None and sam3 is not self._last_sam3_result:
                    self._last_sam3_result = sam3
                    if sam3.annotated_image is not None:
                        self._show_image_on_canvas(
                            self._seg_canvas, sam3.annotated_image, "_photo_seg",
                        )
                capture = self._agent_executor.capture_result
                if capture is not None and capture is not self._last_capture_result:
                    self._last_capture_result = capture
                    if capture.annotated_image is not None:
                        self._show_image_on_canvas(
                            self._grasp_canvas, capture.annotated_image, "_photo_grasp",
                        )

        self._conv_text.config(state=tk.DISABLED)
        self._conv_text.see(tk.END)

    def _on_depth_click(self, event):
        """點擊深度圖查詢深度值"""
        # 優先用 snapshot（agent 運行時 live 停止），否則用 live
        depth = self._snapshot_depth if self._snapshot_depth is not None else self._state.live_depth
        if depth is None:
            return

        cw = self._depth_canvas.winfo_width()
        ch = self._depth_canvas.winfo_height()
        h, w = depth.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        # canvas 上圖片左上角
        x0 = (cw - nw) / 2
        y0 = (ch - nh) / 2

        img_x = int((event.x - x0) / scale)
        img_y = int((event.y - y0) / scale)

        if 0 <= img_x < w and 0 <= img_y < h:
            d = float(depth[img_y, img_x])
            if d > 0:
                self._depth_info_var.set(
                    f"({img_x}, {img_y}) = {d:.3f}m"
                )
                # 在 depth canvas 上畫十字標記
                self._depth_canvas.create_line(
                    event.x - 8, event.y, event.x + 8, event.y,
                    fill="#00ff00", width=1, tags="marker",
                )
                self._depth_canvas.create_line(
                    event.x, event.y - 8, event.x, event.y + 8,
                    fill="#00ff00", width=1, tags="marker",
                )
                self._depth_canvas.create_text(
                    event.x + 12, event.y - 12,
                    text=f"{d:.3f}m", fill="#00ff00",
                    font=("Consolas", 8), anchor=tk.SW, tags="marker",
                )
                self._depth_markers.append((img_x, img_y, d))

                # 如果有 >=2 個標記，畫連線並顯示距離
                if len(self._depth_markers) >= 2:
                    prev = self._depth_markers[-2]
                    curr = self._depth_markers[-1]
                    # 把上一個點也轉回 canvas 座標
                    px = prev[0] * scale + x0
                    py = prev[1] * scale + y0
                    self._depth_canvas.create_line(
                        px, py, event.x, event.y,
                        fill="#ffff00", width=1, dash=(4, 2), tags="marker",
                    )
                    mid_cx = (px + event.x) / 2
                    mid_cy = (py + event.y) / 2
                    delta_d = abs(curr[2] - prev[2])
                    self._depth_canvas.create_text(
                        mid_cx, mid_cy - 6,
                        text=f"Δ={delta_d:.3f}m", fill="#ffff00",
                        font=("Consolas", 8), tags="marker",
                    )
            else:
                self._depth_info_var.set(f"({img_x}, {img_y}) = N/A")

    def _on_depth_clear(self, event):
        """右鍵清除深度標記"""
        self._depth_canvas.delete("marker")
        self._depth_markers.clear()
        self._depth_info_var.set("Click depth to query")

    # ══════════════════════════════════════════════════════════════
    # Pipeline 共用前置
    # ══════════════════════════════════════════════════════════════

    def _prepare_pipeline(self, status_text: str) -> tuple | None:
        """共用管線前置：凍結快照、停止 live、清空 UI

        Returns (snapshot_rgb, snapshot_depth, K, T_cam2arm) or None.
        """
        state = self._state
        if state.sam3_skill is None or state.capture_skill is None:
            self._status_var.set("Models not loaded")
            return None

        K = self._get_K()
        if K is None:
            self._status_var.set("No camera intrinsics")
            return None

        if state.live_rect_left is not None:
            self._snapshot_rgb = state.live_rect_left.copy()
        if state.live_depth is not None:
            self._snapshot_depth = state.live_depth.copy()

        if self._snapshot_rgb is None:
            self._status_var.set("No scene image — start live first")
            return None

        self._live_running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=3.0)
            self._live_thread = None

        self._conv_text.config(state=tk.NORMAL)
        self._conv_text.delete("1.0", tk.END)
        self._conv_text.config(state=tk.DISABLED)
        self._summary_var.set(status_text)
        self._last_sam3_result = None
        self._last_capture_result = None
        self._seg_canvas.delete("all")
        self._grasp_canvas.delete("all")

        self._show_image_on_canvas(
            self._rgb_canvas, self._snapshot_rgb, "_photo_rgb",
        )
        self._set_stage("AGENT_RUNNING")

        return (
            self._snapshot_rgb.copy(),
            self._snapshot_depth.copy(),
            K,
            state.T_cam2arm.copy(),
        )

    def _run_grasp_save(self, snapshot_rgb, snapshot_depth, sam3_result,
                        K, T_cam2arm, scene_points_arm, task_text):
        """共用：grasp 計算 + save plan（在背景線程調用）

        Returns (capture_result, session_dir, plan_path, t_grasp) or None.
        """
        state = self._state
        self.root.after(0, self._append_conv_safe,
                        "  Computing grasp ...\n", "info")
        t0 = time.time()
        capture_result = state.capture_skill.capture(
            snapshot_rgb, snapshot_depth,
            sam3_result.best_mask, K, T_cam2arm,
            scene_points_arm=scene_points_arm,
        )
        t_grasp = time.time() - t0

        if capture_result.num_candidates > 0:
            pos = capture_result.pose_arm[:3, 3]
            self.root.after(0, self._append_conv_safe,
                            f"  Grasp: score={capture_result.grasp_score:.3f}, "
                            f"width={capture_result.grasp_width * 1000:.1f}mm, "
                            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                            f"({t_grasp:.1f}s)\n", "tool_result")
        else:
            self.root.after(0, self._append_conv_safe,
                            f"  Grasp: no candidates ({t_grasp:.1f}s)\n", "error")

        if capture_result.annotated_image is not None:
            self.root.after(0, self._show_image_on_canvas,
                            self._grasp_canvas,
                            capture_result.annotated_image, "_photo_grasp")

        if capture_result.num_candidates == 0:
            return None

        from phase5_vlm_planning.src.plan_serializer import PlanSerializer
        ser = PlanSerializer(
            output_base=os.path.join(_ROOT, "phase5_vlm_planning", "outputs"),
        )
        session_dir = ser.save_session(
            task_text=task_text, rgb=snapshot_rgb, depth=snapshot_depth,
            sam3_result=sam3_result, capture_result=capture_result,
        )
        plan_path = ser.save_plan(
            task_text=task_text, capture_result=capture_result,
            session_dir=session_dir,
        )

        return capture_result, session_dir, plan_path, t_grasp

    # ══════════════════════════════════════════════════════════════
    # Direct Pipeline（跳過 VLM，直接執行 segment → grasp → save）
    # ══════════════════════════════════════════════════════════════

    def _send_direct(self):
        """直接管線：跳過 VLM，segment → grasp → save（目標 <10s）"""
        task_text = self._task_var.get().strip()
        if not task_text:
            self._status_var.set("Enter object description (English)")
            return

        prep = self._prepare_pipeline("Direct pipeline running...")
        if prep is None:
            return
        snapshot_rgb, snapshot_depth, K, T_cam2arm = prep
        self._status_var.set(f"Direct: {task_text}")

        def _run():
            t_start = time.time()
            seg_result = self._run_segment_with_precompute(
                [task_text], snapshot_rgb, snapshot_depth, K, T_cam2arm,
                label="Direct",
            )
            if seg_result is None:
                return
            sam3_result, scene_points_arm, t_seg = seg_result

            grasp_result = self._run_grasp_save(
                snapshot_rgb, snapshot_depth, sam3_result,
                K, T_cam2arm, scene_points_arm, task_text,
            )
            if grasp_result is None:
                self.root.after(0, self._on_direct_error, "No grasp candidates")
                return
            capture_result, session_dir, plan_path, t_grasp = grasp_result

            t_total = time.time() - t_start
            self.root.after(0, self._append_conv_safe,
                            f"\n{'=' * 40}\n"
                            f"Done in {t_total:.1f}s "
                            f"(seg={t_seg:.1f}s, grasp={t_grasp:.1f}s)\n"
                            f"Plan: {plan_path}\n", "info")
            self._finish_pipeline(task_text, sam3_result, capture_result,
                                  session_dir, t_total)

        threading.Thread(target=_run, daemon=True).start()

    # ══════════════════════════════════════════════════════════════
    # LLM Fast Pipeline（單次 LLM 提取 → 並行 SAM3 編碼 → Direct）
    # ══════════════════════════════════════════════════════════════

    def _send_llm_fast(self):
        """LLM 快速模式：單次 LLM 提取描述 + 並行 SAM3 ViT + Direct Pipeline

        時序：
          ┌── LLM 提取 ("紅杯" → ["red cup",...])  ~1s (OpenAI) / ~5s (Claude)
          ├── SAM3 encode_image (ViT 前向)          ~2-4s (並行)
          └── 預計算場景點雲                          ~0.01s (並行)
          ↓ 全部完成
          SAM3 segment (decoder only, ViT 已快取)   ~0.5s
          GraspGen + save                           ~0.5s
          ─────────────────────────────────────────
          Total: max(LLM, SAM3_ViT) + 1s ≈ 3-5s
        """
        task_text = self._task_var.get().strip()
        if not task_text:
            self._status_var.set("Enter a task")
            return

        prep = self._prepare_pipeline("LLM fast pipeline running...")
        if prep is None:
            return
        snapshot_rgb, snapshot_depth, K, T_cam2arm = prep
        self._status_var.set(f"LLM fast: {task_text}")
        state = self._state

        def _run():
            import concurrent.futures
            from phase5_vlm_planning.skills.skill_capture_point import pointcloud_cropper

            t_start = time.time()

            # ── 三路並行 ──
            def _precompute_scene_pc():
                valid = snapshot_depth > 0
                pts_cam, _ = pointcloud_cropper.mask_to_pointcloud(
                    snapshot_depth, K, valid)
                pts_arm = pointcloud_cropper.transform_points(pts_cam, T_cam2arm)
                return pts_arm, valid

            self.root.after(0, self._append_conv_safe,
                            f"[LLM Fast] \"{task_text}\"\n"
                            f"  Parallel: LLM extract + SAM3 encode + scene PC\n",
                            "info")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                future_desc = pool.submit(self._extract_descriptions, task_text)
                future_encode = pool.submit(
                    state.sam3_skill.encode_image, snapshot_rgb)
                future_pc = pool.submit(_precompute_scene_pc)

                descriptions = future_desc.result()
                future_encode.result()  # SAM3 ViT 已快取
                all_points_arm, valid_mask = future_pc.result()

            t_parallel = time.time() - t_start
            self.root.after(0, self._append_conv_safe,
                            f"  Descriptions: {descriptions} ({t_parallel:.1f}s)\n",
                            "tool_result")

            # ── Segment（ViT 已快取，只跑 decoder ~0.5s）── 多描述自動重試
            sam3_result = None
            t0 = time.time()
            for desc in descriptions:
                self.root.after(0, self._append_conv_safe,
                                f"  Segment: \"{desc}\" ...\n", "info")
                result = state.sam3_skill.segment(snapshot_rgb, desc)
                if len(result.masks) > 0:
                    sam3_result = result
                    break
                self.root.after(0, self._append_conv_safe,
                                f"    No match\n", "error")
            t_seg = time.time() - t0

            if sam3_result is None:
                self.root.after(0, self._on_direct_error,
                                f"No object found: {descriptions}")
                return

            self.root.after(0, self._append_conv_safe,
                            f"  Segment OK: score={sam3_result.best_score:.3f} "
                            f"({t_seg:.1f}s)\n", "tool_result")
            if sam3_result.annotated_image is not None:
                self.root.after(0, self._show_image_on_canvas,
                                self._seg_canvas, sam3_result.annotated_image,
                                "_photo_seg")

            # ── 場景/物體分離 ──
            mask = sam3_result.best_mask
            vs, us = np.where(valid_mask)
            scene_points_arm = all_points_arm[~mask[vs, us]]
            if len(scene_points_arm) > 8192:
                idx = np.random.choice(len(scene_points_arm), 8192, replace=False)
                scene_points_arm = scene_points_arm[idx]

            # ── Grasp + Save ──
            grasp_result = self._run_grasp_save(
                snapshot_rgb, snapshot_depth, sam3_result,
                K, T_cam2arm, scene_points_arm, task_text,
            )
            if grasp_result is None:
                self.root.after(0, self._on_direct_error, "No grasp candidates")
                return
            capture_result, session_dir, plan_path, t_grasp = grasp_result

            t_total = time.time() - t_start
            self.root.after(0, self._append_conv_safe,
                            f"\n{'=' * 40}\n"
                            f"Done in {t_total:.1f}s (parallel={t_parallel:.1f}s, "
                            f"seg={t_seg:.1f}s, grasp={t_grasp:.1f}s)\n"
                            f"Plan: {plan_path}\n", "info")
            self._finish_pipeline(task_text, sam3_result, capture_result,
                                  session_dir, t_total)

        threading.Thread(target=_run, daemon=True).start()

    def _extract_descriptions(self, task_text: str) -> list[str]:
        """單次 LLM 調用：提取英文物件描述（多候選）

        優先用 OpenAI gpt-4o-mini（~0.5-1s），否則 Claude CLI（~5-10s）。
        """
        prompt = (
            "From this robotic arm task, extract the target object. "
            "Return ONLY a JSON array of 2-3 English visual descriptions, "
            "most specific first. No explanation.\n"
            f"Task: \"{task_text}\"\n"
            'Example: ["red cup", "red mug", "cup"]'
        )

        # 1) OpenAI gpt-4o-mini — 最快 (~0.5-1s)
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            parsed = self._parse_json_array(text)
            if parsed:
                log.info(f"OpenAI extraction: {parsed}")
                return parsed
        except Exception as e:
            log.info(f"OpenAI extraction unavailable: {e}")

        # 2) Claude CLI — 慢但可用 (~5-10s)
        try:
            import subprocess as sp
            result = sp.run(
                ["claude", "-p", "--output-format", "json",
                 "--max-turns", "1", "--tools", ""],
                input=prompt,
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                response = json.loads(result.stdout.strip())
                text = str(response.get("result", "")) if isinstance(
                    response, dict) else str(response)
                parsed = self._parse_json_array(text)
                if parsed:
                    log.info(f"Claude extraction: {parsed}")
                    return parsed
        except Exception as e:
            log.info(f"Claude extraction failed: {e}")

        # 3) Fallback: 原文當描述
        log.info(f"Extraction fallback: using task text as-is")
        return [task_text]

    @staticmethod
    def _parse_json_array(text: str) -> list[str] | None:
        """從 LLM 回應中解析 JSON 陣列"""
        import re
        text = text.strip()
        # 處理 markdown code block
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    return [str(d) for d in parsed[:5]]
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _run_segment_with_precompute(self, descriptions, snapshot_rgb,
                                     snapshot_depth, K, T_cam2arm, label=""):
        """共用：並行預計算場景 PC + SAM3 分割（多描述自動重試）

        Returns (sam3_result, scene_points_arm, t_seg) or None.
        """
        import concurrent.futures
        from phase5_vlm_planning.skills.skill_capture_point import pointcloud_cropper
        state = self._state

        def _precompute():
            valid = snapshot_depth > 0
            pts_cam, _ = pointcloud_cropper.mask_to_pointcloud(
                snapshot_depth, K, valid)
            pts_arm = pointcloud_cropper.transform_points(pts_cam, T_cam2arm)
            return pts_arm, valid

        tag = f"[{label}] " if label else ""
        self.root.after(0, self._append_conv_safe,
                        f"{tag}Segmenting: {descriptions[0]} ...\n", "info")

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future_pc = pool.submit(_precompute)
            # 多描述自動重試
            sam3_result = None
            for desc in descriptions:
                result = state.sam3_skill.segment(snapshot_rgb, desc)
                if len(result.masks) > 0:
                    sam3_result = result
                    break
            all_points_arm, valid_mask = future_pc.result()
        t_seg = time.time() - t0

        if sam3_result is None or len(sam3_result.masks) == 0:
            self.root.after(0, self._on_direct_error,
                            f"No object found for {descriptions}")
            return None

        self.root.after(0, self._append_conv_safe,
                        f"  Segment: score={sam3_result.best_score:.3f} "
                        f"({t_seg:.1f}s)\n", "tool_result")
        if sam3_result.annotated_image is not None:
            self.root.after(0, self._show_image_on_canvas,
                            self._seg_canvas, sam3_result.annotated_image,
                            "_photo_seg")

        # 分離場景/物體點雲
        mask = sam3_result.best_mask
        vs, us = np.where(valid_mask)
        scene_points_arm = all_points_arm[~mask[vs, us]]
        if len(scene_points_arm) > 8192:
            idx = np.random.choice(len(scene_points_arm), 8192, replace=False)
            scene_points_arm = scene_points_arm[idx]

        return sam3_result, scene_points_arm, t_seg

    def _finish_pipeline(self, task_text, sam3_result, capture_result,
                         session_dir, t_total):
        """共用：構建 AgentResult 並觸發完成回調"""
        from phase5_vlm_planning.src.agent_loop import AgentResult
        result = AgentResult(
            success=True,
            task_text=task_text,
            final_message=f"Pipeline: {t_total:.1f}s",
            sam3_result=sam3_result,
            capture_result=capture_result,
            session_dir=session_dir,
            total_tokens=0,
        )
        self.root.after(0, self._on_agent_done, result)

    def _append_conv_safe(self, text: str, tag: str = ""):
        """線程安全 conversation 追加（在主線程調用）"""
        self._conv_text.config(state=tk.NORMAL)
        self._append_conv(text, tag)
        self._conv_text.config(state=tk.DISABLED)
        self._conv_text.see(tk.END)

    def _on_direct_error(self, msg: str):
        """直接管線錯誤"""
        self._append_conv_safe(f"\n[Error] {msg}\n", "error")
        self._summary_var.set(f"Failed: {msg}")
        self._status_var.set(f"Direct failed: {msg}")
        self._resume_live()

    def _on_agent_done(self, result):
        """Agent 完成回調"""
        from phase5_vlm_planning.src.agent_loop import AgentResult

        self._state.agent_result = result

        if result.success:
            self._set_stage("PLAN_READY")
            self._status_var.set("Agent completed")

            # 更新 summary
            parts = [f"Task: {result.task_text}"]
            if result.capture_result and result.capture_result.num_candidates > 0:
                cr = result.capture_result
                pos = cr.pose_arm[:3, 3]
                parts.append(
                    f"Score: {cr.grasp_score:.3f}  Width: {cr.grasp_width*1000:.1f}mm"
                )
                parts.append(
                    f"Pos(arm): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                )
            parts.append(f"Tokens: {result.total_tokens:,}")
            if result.session_dir:
                parts.append(f"Session: {result.session_dir}")
            self._summary_var.set("\n".join(parts))

            # 最終回覆
            self._conv_text.config(state=tk.NORMAL)
            self._append_conv(f"\n{'='*50}\n", "info")
            self._append_conv(f"Tokens used: {result.total_tokens:,}\n", "info")
            self._conv_text.config(state=tk.DISABLED)

            # 顯示最後的 annotated image 到 Grasp canvas
            if result.capture_result and result.capture_result.annotated_image is not None:
                self._state.result_image = result.capture_result.annotated_image
                self._show_image_on_canvas(
                    self._grasp_canvas,
                    result.capture_result.annotated_image,
                    "_photo_grasp",
                )
            # 顯示 segment 結果到 Segment canvas
            if result.sam3_result and result.sam3_result.annotated_image is not None:
                self._show_image_on_canvas(
                    self._seg_canvas,
                    result.sam3_result.annotated_image,
                    "_photo_seg",
                )

        else:
            # 失敗
            self._conv_text.config(state=tk.NORMAL)
            self._append_conv(
                f"\n[Error] {result.error_message}\n", "error",
            )
            self._conv_text.config(state=tk.DISABLED)
            self._summary_var.set(f"Failed: {result.error_message}")
            self._status_var.set(f"Agent failed: {result.error_message}")
            self._resume_live()

    def _cancel(self):
        if self._agent_loop:
            self._agent_loop.cancel()
            self._status_var.set("Cancelling...")

    def _skip_verify(self):
        if hasattr(self, "_agent_executor") and self._agent_executor:
            self._agent_executor.set_skip_verify(True)
            self._conv_text.config(state=tk.NORMAL)
            self._append_conv("[Skip] 用戶跳過驗證\n", "info")
            self._conv_text.config(state=tk.DISABLED)

    def _clear(self):
        """清空 conversation 和結果"""
        self._conv_text.config(state=tk.NORMAL)
        self._conv_text.delete("1.0", tk.END)
        self._conv_text.config(state=tk.DISABLED)
        self._summary_var.set("No result yet")
        self._state.agent_result = None
        self._state.result_image = None
        self._last_sam3_result = None
        self._last_capture_result = None
        # 清空 segment / grasp canvas
        self._seg_canvas.delete("all")
        self._grasp_canvas.delete("all")
        # 清空深度標記
        self._depth_canvas.delete("marker")
        self._depth_markers.clear()
        self._depth_info_var.set("Click depth to query")

    def _resume_live(self):
        """恢復 live 模式"""
        if (self._state.reader_l is not None and self._state.reader_r is not None
                and self._state.stereo_inf is not None):
            self._state.stereo_inf.set_fast_mode(True)
            self._live_running = True
            self._live_thread = threading.Thread(
                target=self._live_loop, daemon=True,
            )
            self._live_thread.start()
            self._set_stage("LIVE")
            self._update_live_display()
        else:
            self._set_stage("IDLE")

    # ══════════════════════════════════════════════════════════════
    # Save Plan & View 3D
    # ══════════════════════════════════════════════════════════════

    def _save_plan(self):
        result = self._state.agent_result
        if result is None or result.capture_result is None:
            self._status_var.set("No result to save")
            return

        if result.session_dir:
            self._status_var.set(f"Plan already saved at: {result.session_dir}")
            return

        # 手動保存
        from phase5_vlm_planning.src.plan_serializer import PlanSerializer
        ser = PlanSerializer(
            output_base=os.path.join(_ROOT, "phase5_vlm_planning", "outputs"),
        )
        session_dir = ser.save_session(
            task_text=result.task_text,
            rgb=self._snapshot_rgb,
            depth=self._snapshot_depth,
            sam3_result=result.sam3_result,
            capture_result=result.capture_result,
        )
        plan_path = ser.save_plan(
            task_text=result.task_text,
            capture_result=result.capture_result,
            session_dir=session_dir,
        )
        self._status_var.set(f"Plan saved: {plan_path}")

    def _view_3d(self):
        state = self._state
        result = state.agent_result
        if result is None or result.capture_result is None:
            self._status_var.set("No grasp result to visualize")
            return
        if result.capture_result.num_candidates == 0:
            self._status_var.set("No grasp candidates")
            return
        if self._snapshot_depth is None or self._snapshot_rgb is None:
            self._status_var.set("No depth/RGB data")
            return

        K = self._get_K()
        if K is None:
            self._status_var.set("No intrinsics")
            return

        self._status_var.set("Opening 3D viewer...")
        capture_result = result.capture_result
        sam3_result = result.sam3_result

        def _run_viewer():
            try:
                import open3d as o3d
            except ImportError:
                self.root.after(0, self._status_var.set, "open3d not installed")
                return

            pose_arm = capture_result.pose_arm
            T_cam2arm = state.T_cam2arm
            mask = sam3_result.best_mask if sam3_result else None

            if mask is None:
                self.root.after(0, self._status_var.set, "No mask for 3D view")
                return

            from phase5_vlm_planning.skills.skill_capture_point import pointcloud_cropper

            # 物體點雲
            pts_cam, colors = pointcloud_cropper.mask_to_pointcloud(
                self._snapshot_depth, K, mask, self._snapshot_rgb,
            )
            pts_arm = pointcloud_cropper.transform_points(pts_cam, T_cam2arm)

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pts_arm.astype(np.float64))
            if colors is not None:
                pcd_obj.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            # 場景點雲
            scene_mask = ~mask & (self._snapshot_depth > 0)
            pts_scene_cam, _ = pointcloud_cropper.mask_to_pointcloud(
                self._snapshot_depth, K, scene_mask,
            )
            pts_scene_arm = pointcloud_cropper.transform_points(pts_scene_cam, T_cam2arm)
            if len(pts_scene_arm) > 20000:
                idx = np.random.choice(len(pts_scene_arm), 20000, replace=False)
                pts_scene_arm = pts_scene_arm[idx]

            pcd_scene = o3d.geometry.PointCloud()
            pcd_scene.points = o3d.utility.Vector3dVector(
                pts_scene_arm.astype(np.float64),
            )
            pcd_scene.paint_uniform_color([0.5, 0.5, 0.5])

            # 夾爪線框
            gripper_lines = self._make_gripper_lineset(
                pose_arm, self._gripper_width, 0.195,
            )

            # Approach 路徑
            tcp = pose_arm[:3, 3]
            approach_dir = pose_arm[:3, 2]
            pre_grasp = tcp - approach_dir * self._pre_grasp_distance

            approach_line = o3d.geometry.LineSet()
            approach_line.points = o3d.utility.Vector3dVector(
                np.array([pre_grasp, tcp], dtype=np.float64),
            )
            approach_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            approach_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])

            pre_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            pre_sphere.translate(pre_grasp)
            pre_sphere.paint_uniform_color([1.0, 1.0, 0.0])

            tcp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            tcp_sphere.translate(tcp)
            tcp_sphere.paint_uniform_color([1.0, 0.0, 0.0])

            from phase5_vlm_planning.skills.skill_capture_point.grasp_visualizer import (
                compute_contact_point,
            )
            contact = compute_contact_point(pose_arm, pts_arm)
            contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            contact_sphere.translate(contact)
            contact_sphere.paint_uniform_color([0.0, 1.0, 0.0])

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            ws_lines = self._make_workspace_lineset(state.workspace_limits)

            geometries = [
                pcd_scene, pcd_obj,
                gripper_lines, approach_line,
                pre_sphere, tcp_sphere, contact_sphere,
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

            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_lookat(contact.astype(np.float64))

            vis.run()
            vis.destroy_window()
            self.root.after(0, self._status_var.set, "3D viewer closed")

        threading.Thread(target=_run_viewer, daemon=True).start()

    @staticmethod
    def _make_gripper_lineset(pose: np.ndarray, width: float, depth: float):
        import open3d as o3d

        hw = width / 2
        pts_local = np.array([
            [-hw * 0.3, 0, 0],
            [hw * 0.3, 0, 0],
            [-hw, 0, depth * 0.5],
            [hw, 0, depth * 0.5],
            [-hw, 0, depth],
            [hw, 0, depth],
            [0, 0, 0],
        ], dtype=np.float64)

        edges = [
            [0, 1], [0, 2], [1, 3],
            [2, 4], [3, 5], [6, 0], [6, 1],
        ]

        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = (R @ pts_local.T).T + t

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_world)
        ls.lines = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(edges))
        return ls

    @staticmethod
    def _make_workspace_lineset(ws: dict):
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
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.6, 0.0]] * len(edges))
        return ls

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
            self._t_label.set(f"T: {os.path.basename(path)}")
        except Exception as exc:
            self._status_var.set(f"Failed to load T: {exc}")

    # ══════════════════════════════════════════════════════════════
    # 顯示工具
    # ══════════════════════════════════════════════════════════════

    def _fit_to_canvas(self, cv_img: np.ndarray, cw: int, ch: int) -> ImageTk.PhotoImage:
        h, w = cv_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _show_image_on_canvas(self, canvas: tk.Canvas, img: np.ndarray,
                              photo_attr: str):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        photo = self._fit_to_canvas(img, cw, ch)
        setattr(self, photo_attr, photo)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo)

    def _append_conv(self, text: str, tag: str = ""):
        """追加文字到 conversation panel"""
        if tag:
            self._conv_text.insert(tk.END, text, tag)
        else:
            self._conv_text.insert(tk.END, text)

    # ══════════════════════════════════════════════════════════════
    # 清理
    # ══════════════════════════════════════════════════════════════

    def destroy(self):
        self._live_running = False
        if self._agent_loop:
            self._agent_loop.cancel()
        self._stop_live()


def main():
    root = tk.Tk()
    gui = VLMAgentGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.destroy(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
