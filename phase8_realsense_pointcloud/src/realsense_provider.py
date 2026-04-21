"""RealSense frame provider for live RGB-D streaming."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class RealSenseProvider:
    """Read aligned color/depth frames from a RealSense camera."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 20,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._intrinsics: CameraIntrinsics | None = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "pyrealsense2 is required. Install it in ro002 environment first.",
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(self.warmup_frames):
            self._pipeline.wait_for_frames()

        frames = self._align.process(self._pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to fetch initial color frame from RealSense")
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        self._intrinsics = CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._rs = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("RealSense is not started")
        return self._intrinsics

    def get_frames(self, timeout_ms: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned (color_bgr, depth_m)."""
        if self._pipeline is None or self._align is None:
            raise RuntimeError("RealSense is not started")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to fetch synchronized frames")

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        color = np.asanyarray(color_frame.get_data())
        if color.ndim != 3 or color.shape[2] != 3:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        return color, depth
