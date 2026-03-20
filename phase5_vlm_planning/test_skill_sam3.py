"""SAM3 Skill 互動測試 GUI

用法:
    conda run -n ro002 python phase5_vlm_planning/test_skill_sam3.py

功能:
    - 拖放 / 貼上 / 瀏覽 載入圖片
    - 從相機即時預覽並拍攝
    - 輸入文字 prompt 描述目標物件
    - 點擊 Segment 執行推理
    - 顯示標注結果（mask + bbox + score）
    - 右鍵保存結果圖
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageTk

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _scan_cameras(max_index: int = 8) -> list[int]:
    """快速掃描可用相機索引"""
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
        else:
            cap.release()
    return available


class _CameraWindow:
    """相機即時預覽子視窗，點擊 Capture 拍攝並傳回主視窗"""

    _PREVIEW_INTERVAL = 33  # ms (~30fps)

    def __init__(self, parent: tk.Tk, on_capture):
        self._on_capture = on_capture
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._photo = None  # prevent GC

        self.win = tk.Toplevel(parent)
        self.win.title("Camera Capture")
        self.win.geometry("680x560")
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        # 控制列
        ctrl = ttk.Frame(self.win, padding=6)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Camera:").pack(side=tk.LEFT)
        self._cam_var = tk.StringVar()
        self._cam_combo = ttk.Combobox(
            ctrl, textvariable=self._cam_var,
            state="readonly", width=18,
        )
        self._cam_combo.pack(side=tk.LEFT, padx=4)
        self._cam_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_camera())

        self._capture_btn = ttk.Button(ctrl, text="Capture", command=self._do_capture)
        self._capture_btn.pack(side=tk.LEFT, padx=8)
        self._capture_btn.config(state=tk.DISABLED)

        self._status_var = tk.StringVar(value="Scanning cameras...")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="gray").pack(side=tk.RIGHT)

        # 預覽畫布
        self._canvas = tk.Canvas(self.win, bg="#1e1e1e")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 背景掃描相機
        threading.Thread(target=self._scan_and_open, daemon=True).start()

    def is_alive(self) -> bool:
        try:
            return self.win.winfo_exists()
        except tk.TclError:
            return False

    def _scan_and_open(self):
        cams = _scan_cameras()
        if not cams:
            self.win.after(0, self._status_var.set, "No cameras found")
            return
        labels = [f"/dev/video{i}  (index {i})" for i in cams]
        self.win.after(0, self._on_scan_done, cams, labels)

    def _on_scan_done(self, indices: list[int], labels: list[str]):
        self._cam_indices = indices
        self._cam_combo["values"] = labels
        self._cam_combo.current(0)
        self._switch_camera()

    def _switch_camera(self):
        self._stop_preview()
        sel = self._cam_combo.current()
        if sel < 0:
            return
        idx = self._cam_indices[sel]
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            self._status_var.set(f"Failed to open camera {idx}")
            self._capture_btn.config(state=tk.DISABLED)
            return
        self._running = True
        self._capture_btn.config(state=tk.NORMAL)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._status_var.set(f"Camera {idx}: {w}x{h}")
        self._tick()

    def _tick(self):
        if not self._running or self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._last_frame = frame
            self._show_frame(frame)
        if self._running:
            self.win.after(self._PREVIEW_INTERVAL, self._tick)

    def _show_frame(self, frame: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _do_capture(self):
        if hasattr(self, "_last_frame") and self._last_frame is not None:
            self._on_capture(self._last_frame)
            self._status_var.set("Captured!")

    def _stop_preview(self):
        self._running = False
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def _close(self):
        self._stop_preview()
        self.win.destroy()


class SAM3TestGUI:
    """SAM3 互動測試介面"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SAM3 Skill Test")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)

        self._skill = None  # lazy load
        self._cv_image: np.ndarray | None = None  # 原圖 BGR
        self._result_image: np.ndarray | None = None  # 標注圖 BGR
        self._loading_model = False
        self._cam_window: _CameraWindow | None = None

        self._build_ui()
        self._bind_events()

        # 啟動時背景載入模型
        self._load_model_async()

    # ── UI 建構 ──

    def _build_ui(self):
        # 頂部控制列
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="Open Image...", command=self._open_file).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Paste (Ctrl+V)", command=self._paste_image).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(ctrl, text="Camera...", command=self._open_camera_window).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(ctrl, text="Prompt:").pack(side=tk.LEFT)
        self._prompt_var = tk.StringVar()
        self._prompt_entry = ttk.Entry(ctrl, textvariable=self._prompt_var, width=30)
        self._prompt_entry.pack(side=tk.LEFT, padx=4)

        self._seg_btn = ttk.Button(ctrl, text="Segment", command=self._run_segment)
        self._seg_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(ctrl, text="Save Result...", command=self._save_result).pack(side=tk.LEFT)

        # 狀態列
        self._status_var = tk.StringVar(value="Loading SAM3 model...")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="gray").pack(side=tk.RIGHT)

        # 圖片顯示區（左原圖、右結果）
        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 左側：原圖
        left = ttk.LabelFrame(panes, text="Input Image", padding=4)
        panes.add(left, weight=1)
        self._input_canvas = tk.Canvas(left, bg="#2b2b2b")
        self._input_canvas.pack(fill=tk.BOTH, expand=True)

        # 右側：結果
        right = ttk.LabelFrame(panes, text="Segmentation Result", padding=4)
        panes.add(right, weight=1)
        self._result_canvas = tk.Canvas(right, bg="#2b2b2b")
        self._result_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部資訊列
        info = ttk.Frame(self.root, padding=4)
        info.pack(fill=tk.X)
        self._info_var = tk.StringVar()
        ttk.Label(info, textvariable=self._info_var, foreground="#555555").pack(side=tk.LEFT)

    def _bind_events(self):
        self.root.bind("<Control-v>", lambda e: self._paste_image())
        self.root.bind("<Return>", lambda e: self._run_segment())
        self._input_canvas.bind("<Configure>", lambda e: self._redraw_input())
        self._result_canvas.bind("<Configure>", lambda e: self._redraw_result())

    # ── 模型載入 ──

    def _load_model_async(self):
        self._loading_model = True
        self._seg_btn.config(state=tk.DISABLED)

        def _load():
            from phase5_vlm_planning.skills.skill_sam3 import SAM3Skill
            try:
                self._skill = SAM3Skill()
                self.root.after(0, self._on_model_loaded, True)
            except Exception as exc:
                self.root.after(0, self._on_model_loaded, False, str(exc))

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _on_model_loaded(self, ok: bool, err: str = ""):
        self._loading_model = False
        if ok:
            self._status_var.set("Model ready")
            self._seg_btn.config(state=tk.NORMAL)
        else:
            self._status_var.set(f"Model load failed: {err}")

    # ── 相機拍攝 ──

    def _open_camera_window(self):
        """開啟相機預覽視窗"""
        if self._cam_window is not None and self._cam_window.is_alive():
            self._cam_window.win.lift()
            return
        self._cam_window = _CameraWindow(self.root, self._on_camera_capture)

    def _on_camera_capture(self, frame: np.ndarray):
        """相機拍攝回調"""
        self._cv_image = frame.copy()
        self._result_image = None
        self._redraw_input()
        self._clear_result()
        h, w = frame.shape[:2]
        self._status_var.set(f"Captured from camera: {w}x{h}")

    # ── 圖片載入 ──

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._load_image_from_path(path)

    def _paste_image(self):
        """從剪貼簿貼上圖片"""
        try:
            # 嘗試用 xclip 讀取剪貼簿圖片（Linux）
            import subprocess
            proc = subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                capture_output=True, timeout=3,
            )
            if proc.returncode == 0 and proc.stdout:
                pil = Image.open(BytesIO(proc.stdout))
                rgb = np.array(pil.convert("RGB"))
                self._cv_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._result_image = None
                self._redraw_input()
                self._clear_result()
                h, w = self._cv_image.shape[:2]
                self._status_var.set(f"Pasted image: {w}x{h}")
                return
        except Exception:
            pass

        # Fallback: PIL/Tk 剪貼簿
        try:
            pil = ImageTk.getimage(self.root.clipboard_get())
            rgb = np.array(pil.convert("RGB"))
            self._cv_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._result_image = None
            self._redraw_input()
            self._clear_result()
            return
        except Exception:
            pass

        self._status_var.set("No image in clipboard")

    def _load_image_from_path(self, path: str):
        img = cv2.imread(path)
        if img is None:
            self._status_var.set(f"Failed to load: {path}")
            return
        self._cv_image = img
        self._result_image = None
        self._redraw_input()
        self._clear_result()
        h, w = img.shape[:2]
        self._status_var.set(f"Loaded: {os.path.basename(path)} ({w}x{h})")

    # ── 推理 ──

    def _run_segment(self):
        if self._skill is None:
            self._status_var.set("Model not loaded yet")
            return
        if self._cv_image is None:
            self._status_var.set("No image loaded")
            return
        prompt = self._prompt_var.get().strip()
        if not prompt:
            self._status_var.set("Enter a text prompt first")
            return

        self._seg_btn.config(state=tk.DISABLED)
        self._status_var.set(f"Segmenting: \"{prompt}\" ...")
        self.root.update_idletasks()

        def _infer():
            try:
                result = self._skill.segment(self._cv_image, prompt)
                self.root.after(0, self._on_segment_done, result, None)
            except Exception as exc:
                self.root.after(0, self._on_segment_done, None, str(exc))

        threading.Thread(target=_infer, daemon=True).start()

    def _on_segment_done(self, result, err: str | None):
        self._seg_btn.config(state=tk.NORMAL)
        if err:
            self._status_var.set(f"Error: {err}")
            return

        self._result_image = result.annotated_image
        self._redraw_result()

        n = len(result.masks)
        if n == 0:
            self._status_var.set(f"No masks found for \"{result.object_description}\"")
            self._info_var.set("")
        else:
            score_strs = [f"#{i}={s:.2f}" for i, s in enumerate(result.scores)]
            self._status_var.set(f"Found {n} mask(s), best score: {result.best_score:.3f}")
            self._info_var.set(f"Scores: {', '.join(score_strs)}")

    # ── 顯示 ──

    def _cv_to_tk(self, cv_img: np.ndarray, canvas: tk.Canvas) -> ImageTk.PhotoImage | None:
        """將 BGR 圖片縮放適配 canvas 尺寸，轉為 PhotoImage"""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return None
        h, w = cv_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _redraw_input(self):
        self._input_canvas.delete("all")
        if self._cv_image is None:
            self._input_canvas.create_text(
                self._input_canvas.winfo_width() // 2,
                self._input_canvas.winfo_height() // 2,
                text="Open or paste an image",
                fill="#888888", font=("sans-serif", 14),
            )
            return
        photo = self._cv_to_tk(self._cv_image, self._input_canvas)
        if photo:
            self._input_photo = photo  # prevent GC
            cw = self._input_canvas.winfo_width()
            ch = self._input_canvas.winfo_height()
            self._input_canvas.create_image(cw // 2, ch // 2, image=photo)

    def _redraw_result(self):
        self._result_canvas.delete("all")
        if self._result_image is None:
            return
        photo = self._cv_to_tk(self._result_image, self._result_canvas)
        if photo:
            self._result_photo = photo  # prevent GC
            cw = self._result_canvas.winfo_width()
            ch = self._result_canvas.winfo_height()
            self._result_canvas.create_image(cw // 2, ch // 2, image=photo)

    def _clear_result(self):
        self._result_canvas.delete("all")
        self._info_var.set("")

    # ── 保存 ──

    def _save_result(self):
        if self._result_image is None:
            self._status_var.set("No result to save")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            cv2.imwrite(path, self._result_image)
            self._status_var.set(f"Saved: {path}")


def main():
    root = tk.Tk()
    SAM3TestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
