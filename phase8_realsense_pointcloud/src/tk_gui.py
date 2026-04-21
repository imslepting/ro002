"""Tk GUI for RealSense live RGB-D and point cloud preview."""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np

from shared.tk_utils import BTN_ACCENT, BTN_DANGER, DARK_BG, cv_to_photoimage

from phase8_realsense_pointcloud.src.pointcloud_builder import (
    build_pointcloud,
    colorize_depth,
)
from phase8_realsense_pointcloud.src.realsense_provider import RealSenseProvider


class _CloudViewer:
    """Background Open3D viewer with non-blocking updates."""

    def __init__(self):
        self._o3d = None
        self._vis = None
        self._pcd = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._view_initialized = False
        self._geometry_added = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            import open3d as o3d
        except Exception:
            self._running = False
            return

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window("Phase 8 Point Cloud", width=980, height=720)
        if not ok:
            self._running = False
            return

        opt = self._vis.get_render_option()
        if opt is None:
            self._vis.destroy_window()
            self._running = False
            return
        # Match phase3 style: dark background + larger points for visibility.
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        self._pcd = o3d.geometry.PointCloud()

        while self._running:
            with self._lock:
                pending = self._pending
                self._pending = None

            if pending is not None:
                pts, cols = pending
                if pts.shape[0] == 0:
                    self._vis.poll_events()
                    self._vis.update_renderer()
                    time.sleep(0.01)
                    continue

                self._pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                self._pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                if not self._geometry_added:
                    self._vis.add_geometry(self._pcd)
                    self._geometry_added = True
                else:
                    self._vis.update_geometry(self._pcd)

                if not self._view_initialized and pts.shape[0] > 0:
                    # Fit camera to first valid cloud for RealSense coordinates.
                    ctr = self._vis.get_view_control()
                    if ctr is not None:
                        center = pts.mean(axis=0)
                        ctr.set_lookat(center.tolist())
                        ctr.set_front([0.0, 0.0, 1.0])
                        ctr.set_up([0.0, -1.0, 0.0])
                        ctr.set_zoom(0.45)
                    self._view_initialized = True

            self._vis.poll_events()
            self._vis.update_renderer()
            time.sleep(0.01)

        self._vis.destroy_window()

    def update(self, points: np.ndarray, colors: np.ndarray) -> None:
        if not self._running:
            return
        with self._lock:
            self._pending = (points.copy(), colors.copy())

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._view_initialized = False
        self._geometry_added = False


class RealsensePointCloudGUI:
    _UI_INTERVAL_MS = 50

    def __init__(self, root: tk.Tk, cfg: dict, root_dir: str):
        self.root = root
        self.root_dir = root_dir
        self.cfg = cfg
        self.p8_cfg = cfg.get("phase8_realsense", {})

        self.provider = RealSenseProvider(
            width=int(self.p8_cfg.get("width", 640)),
            height=int(self.p8_cfg.get("height", 480)),
            fps=int(self.p8_cfg.get("fps", 30)),
            warmup_frames=int(self.p8_cfg.get("warmup_frames", 20)),
        )
        self.viewer = _CloudViewer()
        self._viewer_enabled = False

        self._running = False
        self._loop_thread: threading.Thread | None = None

        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_points: np.ndarray | None = None
        self._latest_colors: np.ndarray | None = None

        self._fps = 0.0
        self._latency_ms = 0.0
        self._status = tk.StringVar(value="Idle")
        self._metrics = tk.StringVar(value="FPS: 0.0 | Latency: 0 ms | Points: 0 | ValidDepth: 0.0%")
        self._viewer_btn_text = tk.StringVar(value="Open Point Cloud")

        self._photo_rgb = None
        self._photo_depth = None
        self._last_fallback = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.title("Phase 8 - RealSense Live Point Cloud")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1420x860")

        top = tk.Frame(self.root, bg=DARK_BG)
        top.pack(fill="x", padx=12, pady=8)
        tk.Button(top, text="Start", command=self.start, **BTN_ACCENT).pack(side="left", padx=4)
        tk.Button(top, text="Stop", command=self.stop, **BTN_DANGER).pack(side="left", padx=4)
        tk.Button(
            top,
            textvariable=self._viewer_btn_text,
            command=self.toggle_pointcloud_viewer,
            **BTN_ACCENT,
        ).pack(side="left", padx=4)
        tk.Button(top, text="Save Snapshot", command=self.save_snapshot, **BTN_ACCENT).pack(side="left", padx=4)

        tk.Label(top, textvariable=self._status, bg=DARK_BG, fg="#dddddd", font=("Helvetica", 11)).pack(
            side="left", padx=14,
        )
        tk.Label(top, textvariable=self._metrics, bg=DARK_BG, fg="#8fd18f", font=("Helvetica", 10)).pack(
            side="left", padx=14,
        )

        content = tk.Frame(self.root, bg=DARK_BG)
        content.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(content, bg="#262626")
        right = tk.Frame(content, bg="#262626")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        tk.Label(left, text="Color", bg="#262626", fg="#ffffff", font=("Helvetica", 12, "bold")).pack(pady=6)
        self.rgb_canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.rgb_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(right, text="Depth", bg="#262626", fg="#ffffff", font=("Helvetica", 12, "bold")).pack(pady=6)
        self.depth_canvas = tk.Canvas(right, bg="#111111", highlightthickness=0)
        self.depth_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        hint = "3D point cloud opens in separate Open3D window"
        tk.Label(self.root, text=hint, bg=DARK_BG, fg="#9c9c9c", font=("Helvetica", 10)).pack(pady=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self) -> None:
        if self._running:
            return
        try:
            self.provider.start()
        except Exception as exc:
            messagebox.showerror("Phase 8", f"Failed to start RealSense: {exc}")
            return

        self._running = True
        self._status.set("Live")
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()
        self._refresh_ui()

    def stop(self) -> None:
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None
        self.viewer.stop()
        self._viewer_enabled = False
        self._viewer_btn_text.set("Open Point Cloud")
        self.provider.stop()
        self._status.set("Stopped")

    def toggle_pointcloud_viewer(self) -> None:
        if self._viewer_enabled:
            self.viewer.stop()
            self._viewer_enabled = False
            self._viewer_btn_text.set("Open Point Cloud")
            if self._running:
                self._status.set("Live")
            else:
                self._status.set("Viewer closed")
            return

        self.viewer.start()
        self._viewer_enabled = True
        self._viewer_btn_text.set("Close Point Cloud")
        if self._running:
            self._status.set("Live + 3D")
        else:
            self._status.set("3D viewer opened (camera not started)")

    def _loop(self) -> None:
        min_depth = float(self.p8_cfg.get("min_depth", 0.15))
        max_depth = float(self.p8_cfg.get("max_depth", 3.5))
        stride = int(self.p8_cfg.get("cloud_stride", 2))

        intr = self.provider.intrinsics
        frame_count = 0
        fps_t0 = time.time()

        while self._running:
            t0 = time.time()
            try:
                color, depth = self.provider.get_frames(timeout_ms=1000)
            except Exception:
                time.sleep(0.05)
                continue

            valid_depth = depth > 0
            valid_ratio = float(valid_depth.mean())

            points, colors = build_pointcloud(
                depth,
                color,
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                min_depth=min_depth,
                max_depth=max_depth,
                stride=stride,
            )

            # If configured depth range yields zero points but depth is valid,
            # retry with a wider range to avoid black/empty viewer.
            fallback_used = False
            if points.shape[0] == 0 and valid_ratio > 0.01:
                depth_valid_vals = depth[valid_depth]
                auto_min = max(0.03, float(depth_valid_vals.min()) - 0.05)
                auto_max = min(10.0, float(depth_valid_vals.max()) + 0.10)
                points, colors = build_pointcloud(
                    depth,
                    color,
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx,
                    cy=intr.cy,
                    min_depth=auto_min,
                    max_depth=auto_max,
                    stride=stride,
                )
                fallback_used = points.shape[0] > 0

            self._latest_color = color
            self._latest_depth = depth
            self._latest_points = points
            self._latest_colors = colors
            self._last_fallback = fallback_used

            if self._viewer_enabled:
                if points.shape[0] > 0:
                    self.viewer.update(points, colors)

            frame_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_t0 = time.time()
            self._latency_ms = (time.time() - t0) * 1000.0

    def _fit(self, img_bgr: np.ndarray, canvas: tk.Canvas):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 2 or ch <= 2:
            return None
        h, w = img_bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        return cv_to_photoimage(img_bgr, (nw, nh))

    def _refresh_ui(self) -> None:
        if not self._running:
            return

        if self._latest_color is not None:
            p = self._fit(self._latest_color, self.rgb_canvas)
            if p is not None:
                self._photo_rgb = p
                self.rgb_canvas.delete("all")
                self.rgb_canvas.create_image(
                    self.rgb_canvas.winfo_width() // 2,
                    self.rgb_canvas.winfo_height() // 2,
                    image=p,
                )

        if self._latest_depth is not None:
            max_d = float(self.p8_cfg.get("max_depth", 1.8))
            d_vis = colorize_depth(self._latest_depth, max_depth=max_d)
            p = self._fit(d_vis, self.depth_canvas)
            if p is not None:
                self._photo_depth = p
                self.depth_canvas.delete("all")
                self.depth_canvas.create_image(
                    self.depth_canvas.winfo_width() // 2,
                    self.depth_canvas.winfo_height() // 2,
                    image=p,
                )

        pts = 0 if self._latest_points is None else int(self._latest_points.shape[0])
        valid_ratio = 0.0
        if self._latest_depth is not None:
            valid_ratio = float((self._latest_depth > 0).mean())
        self._metrics.set(
            f"FPS: {self._fps:.1f} | Latency: {self._latency_ms:.0f} ms | "
            f"Points: {pts} | ValidDepth: {valid_ratio*100:.1f}%"
        )

        if self._running and pts == 0:
            self._status.set("Live (no valid points in current depth range)")
        elif self._running and self._viewer_enabled and self._last_fallback:
            self._status.set("Live + 3D (auto depth-range fallback)")
        elif self._running and self._viewer_enabled:
            self._status.set("Live + 3D")
        elif self._running:
            self._status.set("Live")

        self.root.after(self._UI_INTERVAL_MS, self._refresh_ui)

    def save_snapshot(self) -> None:
        if self._latest_color is None or self._latest_depth is None:
            self._status.set("No frame to save")
            return
        out_dir = os.path.join(self.root_dir, "phase8_realsense_pointcloud", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        color_path = os.path.join(out_dir, f"color_{ts}.png")
        depth_path = os.path.join(out_dir, f"depth_{ts}.npy")
        cv2.imwrite(color_path, self._latest_color)
        np.save(depth_path, self._latest_depth)

        if self._latest_points is not None and self._latest_points.size > 0:
            try:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self._latest_points.astype(np.float64))
                if self._latest_colors is not None and self._latest_colors.size > 0:
                    pcd.colors = o3d.utility.Vector3dVector(self._latest_colors.astype(np.float64))
                ply_path = os.path.join(out_dir, f"cloud_{ts}.ply")
                o3d.io.write_point_cloud(ply_path, pcd)
                self._status.set(f"Saved: {os.path.basename(color_path)}, {os.path.basename(depth_path)}, {os.path.basename(ply_path)}")
            except Exception as exc:
                self._status.set(f"Saved RGB/Depth only, PLY failed: {exc}")
        else:
            self._status.set(f"Saved: {os.path.basename(color_path)}, {os.path.basename(depth_path)}")

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()


def run_realsense_gui(cfg: dict, root_dir: str) -> None:
    root = tk.Tk()
    RealsensePointCloudGUI(root, cfg, root_dir)
    root.mainloop()
