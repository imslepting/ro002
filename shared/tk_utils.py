"""共用 tkinter 工具 — 影像轉換、CameraFeedWidget、樣式常量、分辨率探測"""

from __future__ import annotations

import threading
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from shared.camera_manager import CameraReader, open_camera, release_camera


# ── 樣式常量 ──

DARK_BG = "#1e1e1e"
PANEL_BG = "#2d2d2d"
ACCENT_GREEN = "#4ec34e"
ACCENT_RED = "#dc3545"
ACCENT_YELLOW = "#e0c846"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#aaaaaa"
TEXT_DIM = "#777777"

FONT_TITLE = ("Helvetica", 16, "bold")
FONT_BODY = ("Helvetica", 12)
FONT_SMALL = ("Helvetica", 10)
FONT_MONO = ("Courier", 10)

BTN_STYLE = {
    "bg": "#3a3a3a",
    "fg": TEXT_PRIMARY,
    "activebackground": "#505050",
    "activeforeground": "#ffffff",
    "relief": "flat",
    "padx": 12,
    "pady": 6,
    "font": ("Helvetica", 11),
    "cursor": "hand2",
}

BTN_ACCENT = {
    **BTN_STYLE,
    "bg": "#2a6e2a",
    "activebackground": "#3a8e3a",
}

BTN_DANGER = {
    **BTN_STYLE,
    "bg": "#6e2a2a",
    "activebackground": "#8e3a3a",
}


# ── 影像轉換 ──

def cv_to_photoimage(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> ImageTk.PhotoImage:
    """將 OpenCV BGR 幀轉為 tkinter PhotoImage

    Args:
        frame_bgr: BGR numpy array
        target_size: (width, height) 如果需要縮放
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if target_size is not None:
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ── 分辨率探測 ──

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


def probe_resolutions(cam_index: int) -> list[tuple[int, int]]:
    """探測相機實際支持的分辨率

    用單一 VideoCapture 依序 set → read → 取 frame.shape，
    避免反覆開關設備導致 USB 相機不穩定。
    """
    cap = open_camera(cam_index)
    if not cap.isOpened():
        return []

    supported: list[tuple[int, int]] = []
    for w, h in _COMMON_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # 讀兩幀：第一幀可能還是舊分辨率，第二幀才是新的
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


# ── CameraFeedWidget ──

class CameraFeedWidget(tk.Frame):
    """在 tkinter Frame 中即時顯示 CameraReader 的畫面

    用 after(33ms) 定時從 CameraReader 讀取幀並顯示。
    overlay_fn(frame_bgr) -> frame_bgr 可選疊加函數。
    """

    def __init__(
        self,
        master,
        reader: CameraReader | None,
        display_size: tuple[int, int],
        overlay_fn=None,
        **kwargs,
    ):
        super().__init__(master, bg=DARK_BG, **kwargs)
        self._reader = reader
        self._display_size = display_size  # (w, h)
        self._overlay_fn = overlay_fn
        self._photo: ImageTk.PhotoImage | None = None
        self._running = False
        self._after_id: str | None = None

        self._label = tk.Label(self, bg=DARK_BG)
        self._label.pack()

        # Show placeholder
        self._show_placeholder("NO SIGNAL")

    def _show_placeholder(self, text: str) -> None:
        w, h = self._display_size
        img = np.zeros((h, w, 3), dtype=np.uint8) + 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(img, text, (x, y), font, 0.8, (0, 0, 180), 2)
        self._photo = cv_to_photoimage(img)
        self._label.configure(image=self._photo)

    def start_feed(self) -> None:
        self._running = True
        self._update()

    def stop_feed(self) -> None:
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def set_reader(self, new_reader: CameraReader | None) -> None:
        self._reader = new_reader

    def set_overlay_fn(self, fn) -> None:
        self._overlay_fn = fn

    def _update(self) -> None:
        if not self._running:
            return
        reader = self._reader
        if reader is not None and reader.frame is not None:
            frame = reader.frame.copy()
            try:
                if self._overlay_fn is not None:
                    frame = self._overlay_fn(frame)
                self._photo = cv_to_photoimage(frame, self._display_size)
                self._label.configure(image=self._photo)
            except Exception as e:
                import traceback
                traceback.print_exc()
        self._after_id = self.after(33, self._update)


# ── 工具函數 ──

def run_in_thread(target, callback=None, root=None):
    """在背景線程中執行 target()，完成後用 root.after() 回調"""
    def _worker():
        result = target()
        if callback and root:
            root.after(0, lambda: callback(result))
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
