"""共用相機讀取器 — 在獨立線程中持續讀取單台相機的最新幀"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# USB 相機在 release() 後設備節點不會立即可用，需要等待
_REOPEN_DELAY = 0.3    # 秒，release 後等待時間
_OPEN_RETRIES = 5      # 重試次數
_RETRY_INTERVAL = 0.5  # 秒，每次重試間隔


def release_camera(cap: cv2.VideoCapture | None) -> None:
    """釋放相機並等待設備節點就緒"""
    if cap is not None and cap.isOpened():
        cap.release()
    time.sleep(_REOPEN_DELAY)


def open_camera(
    cam_index: int,
    resolution: tuple[int, int] | None = None,
    retries: int = _OPEN_RETRIES,
) -> cv2.VideoCapture:
    """帶重試的相機開啟，解決 USB 設備釋放後短暫不可用的問題"""
    for attempt in range(retries):
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            if resolution is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            return cap
        cap.release()
        if attempt < retries - 1:
            time.sleep(_RETRY_INTERVAL)
    # 返回未開啟的 cap，讓調用方處理
    return cv2.VideoCapture(cam_index)


class CameraReader:
    """在獨立線程中持續讀取單台相機的最新幀"""

    def __init__(self, cam_index: int, resolution: tuple[int, int] | None = None):
        self.cam_index = cam_index
        self.resolution = resolution
        self.cap = open_camera(cam_index, resolution)
        self.frame: np.ndarray | None = None
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.cap.isOpened():
            return
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame = frame

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        release_camera(self.cap)
