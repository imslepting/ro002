"""Tk GUI for phase7 arm ICP with live RealSense cloud and virtual arm overlay."""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np
import yaml

from shared.tk_utils import BTN_ACCENT, BTN_DANGER, DARK_BG, cv_to_photoimage

try:
    from phase7_arm_icp.src.icp_calibrator import (
        _matrix_to_euler_xyz_deg,
        calibrate_cam_to_arm,
        euler_xyz_deg_to_matrix,
        load_model_from_mesh_dir,
        save_result_json,
    )
except Exception:
    from src.icp_calibrator import (
        _matrix_to_euler_xyz_deg,
        calibrate_cam_to_arm,
        euler_xyz_deg_to_matrix,
        load_model_from_mesh_dir,
        save_result_json,
    )
from phase8_realsense_pointcloud.src.pointcloud_builder import build_pointcloud, colorize_depth
from phase8_realsense_pointcloud.src.realsense_provider import RealSenseProvider


def _save_temp_cloud_npz(root_dir: str, points: np.ndarray, colors: np.ndarray) -> str:
    out_dir = os.path.join(root_dir, "phase7_arm_icp", "outputs", "cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "live_capture.npz")
    np.savez(out_path, points=points.astype(np.float32), colors=colors.astype(np.float32))
    return out_path


def _update_settings_t_cam_to_arm(settings_path: str, t_cam_to_arm: list[list[float]]) -> None:
    with open(settings_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("arm", {})
    cfg["arm"]["T_cam2arm"] = t_cam_to_arm

    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _safe_t_cam2arm(cfg: dict) -> np.ndarray:
    arm = cfg.get("arm", {})
    t = np.asarray(arm.get("T_cam2arm"), dtype=np.float64) if arm.get("T_cam2arm") is not None else None
    if t is None or t.shape != (4, 4):
        return np.eye(4, dtype=np.float64)
    return t


def _create_coordinate_frame_geometry(t_arm_to_cam: np.ndarray, axis_length: float = 0.2):
    """Create coordinate frame geometry (origin point + 3 axis lines) from transformation matrix.
    
    Args:
        t_arm_to_cam: 4x4 transformation matrix from arm frame to camera frame
        axis_length: length of each axis line (m)
    
    Returns:
        tuple of (origin_sphere, axis_lineset) geometries
    """
    try:
        import open3d as o3d
    except Exception:
        return None, None

    origin = t_arm_to_cam[:3, 3]
    rotation = t_arm_to_cam[:3, :3]

    # Create origin point cloud (small sphere)
    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(np.array([origin], dtype=np.float64))
    origin_pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 1.0]], dtype=np.float64))

    # Create axis lines
    axes_points = np.array([
        origin,  # 0: origin
        origin + rotation[:, 0] * axis_length,  # 1: X axis endpoint (red)
        origin + rotation[:, 1] * axis_length,  # 2: Y axis endpoint (green)
        origin + rotation[:, 2] * axis_length,  # 3: Z axis endpoint (blue)
    ], dtype=np.float64)

    lines = np.array([
        [0, 1],  # X axis
        [0, 2],  # Y axis
        [0, 3],  # Z axis
    ], dtype=np.int32)

    colors = np.array([
        [1.0, 0.0, 0.0],  # X: red
        [0.0, 1.0, 0.0],  # Y: green
        [0.0, 0.0, 1.0],  # Z: blue
    ], dtype=np.float64)

    axis_lineset = o3d.geometry.LineSet()
    axis_lineset.points = o3d.utility.Vector3dVector(axes_points)
    axis_lineset.lines = o3d.utility.Vector2iVector(lines)
    axis_lineset.colors = o3d.utility.Vector3dVector(colors)

    return origin_pcd, axis_lineset


class _OverlayCloudViewer:
    """Open3D viewer for live cloud + arm coordinate frame overlay."""

    def __init__(self):
        self._vis = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pending_transform: np.ndarray | None = None
        self._geometry_added = False
        self._view_initialized = False

        self._live_pcd = None
        self._arm_origin = None  # Origin point
        self._arm_axis = None    # Axis lines
        
        # Store latest transform for smooth updates
        self._current_t_arm_to_cam = np.eye(4, dtype=np.float64)

    def set_model_points(self, points: np.ndarray) -> None:
        """Dummy method for compatibility (not needed for coordinate frame visualization)."""
        pass

    def set_t_arm_to_cam(self, t_arm_to_cam: np.ndarray) -> None:
        """Update the arm coordinate frame without changing the live cloud."""
        if not self._running:
            return
        with self._lock:
            self._pending_transform = np.asarray(t_arm_to_cam, dtype=np.float64).copy()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._geometry_added = False
        self._view_initialized = False

    def update(self, live_points: np.ndarray, live_colors: np.ndarray, t_arm_to_cam: np.ndarray) -> None:
        if not self._running:
            return
        with self._lock:
            self._pending = (
                np.asarray(live_points, dtype=np.float64).copy(),
                np.asarray(live_colors, dtype=np.float64).copy(),
                np.asarray(t_arm_to_cam, dtype=np.float64).copy(),
            )

    def _loop(self) -> None:
        try:
            import open3d as o3d
        except Exception:
            self._running = False
            return

        vis = o3d.visualization.Visualizer()
        if not vis.create_window("Phase 7 Arm ICP Live", width=1100, height=760):
            self._running = False
            return
        self._vis = vis

        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.array([0.1, 0.1, 0.1])
<<<<<<< HEAD
            opt.point_size = 20.0
            opt.line_width = 30.0
=======
            opt.point_size = 2.0
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

        self._live_pcd = o3d.geometry.PointCloud()

        while self._running:
            with self._lock:
                pending = self._pending
                self._pending = None
                pending_transform = self._pending_transform
                self._pending_transform = None

            # Update transform if new one is provided (from slider)
            # slider-based transform takes priority
            if pending_transform is not None:
                self._current_t_arm_to_cam = pending_transform

            # Update point cloud if new one is provided (from camera/compute)
            if pending is not None:
                live_points, live_colors, t_from_pending = pending
                if live_points.shape[0] > 0:
                    # If no slider transform was applied this frame, use the compute thread's transform
                    if pending_transform is None:
                        self._current_t_arm_to_cam = t_from_pending
                    
                    self._live_pcd.points = o3d.utility.Vector3dVector(live_points)
                    self._live_pcd.colors = o3d.utility.Vector3dVector(live_colors)

                    if not self._geometry_added:
                        vis.add_geometry(self._live_pcd)
                        self._geometry_added = True
                    else:
                        # Update existing geometry with new points
                        vis.update_geometry(self._live_pcd)

            # Always update arm coordinate frame (regardless of cloud update)
            if self._geometry_added:
                origin, axis = _create_coordinate_frame_geometry(self._current_t_arm_to_cam, axis_length=0.2)
                
                if origin is not None:
                    if self._arm_origin is None:
                        vis.add_geometry(origin)
                        self._arm_origin = origin
                    else:
                        self._arm_origin.points = origin.points
                        self._arm_origin.colors = origin.colors
                        vis.update_geometry(self._arm_origin)
                
                if axis is not None:
                    if self._arm_axis is None:
                        vis.add_geometry(axis)
                        self._arm_axis = axis
                    else:
                        self._arm_axis.points = axis.points
                        self._arm_axis.lines = axis.lines
                        self._arm_axis.colors = axis.colors
                        vis.update_geometry(self._arm_axis)

            if not self._view_initialized and self._live_pcd.has_points():
                live_points = np.asarray(self._live_pcd.points)
                ctr = vis.get_view_control()
                if ctr is not None:
                    center = live_points.mean(axis=0)
                    ctr.set_lookat(center.tolist())
                    ctr.set_front([0.0, 0.0, 1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.45)
                self._view_initialized = True

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

        vis.destroy_window()


class Phase7ArmIcpGUI:
    _UI_INTERVAL_MS = 50

    def __init__(self, root: tk.Tk, cfg: dict, root_dir: str, settings_path: str):
        self.root = root
        self.cfg = cfg
        self.root_dir = root_dir
        self.settings_path = settings_path

        self.p8_cfg = cfg.get("phase8_realsense", {})
        self.p7_cfg = cfg.get("phase7_arm_icp", {})

        self.provider = RealSenseProvider(
            width=int(self.p8_cfg.get("width", 640)),
            height=int(self.p8_cfg.get("height", 480)),
            fps=int(self.p8_cfg.get("fps", 30)),
            warmup_frames=int(self.p8_cfg.get("warmup_frames", 20)),
        )

        self.viewer = _OverlayCloudViewer()
        self._viewer_enabled = False
        self._running = False
        self._solving = False

        self._loop_thread: threading.Thread | None = None

        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_points: np.ndarray | None = None
        self._latest_colors: np.ndarray | None = None
        self._last_fallback = False

        self._fps = 0.0
        self._latency_ms = 0.0

        self._status = tk.StringVar(value="Idle")
        self._metrics = tk.StringVar(value="FPS: 0.0 | Latency: 0 ms | Points: 0 | ValidDepth: 0.0%")
        self._icp_metrics = tk.StringVar(value="ICP: not solved")
        self._viewer_btn_text = tk.StringVar(value="Open Point Cloud")
        self._solve_btn_text = tk.StringVar(value="cpi solve")

        # Manual adjustment sliders
        self._tx_var = tk.DoubleVar()
        self._ty_var = tk.DoubleVar()
        self._tz_var = tk.DoubleVar()
        self._rx_var = tk.DoubleVar()
        self._ry_var = tk.DoubleVar()
        self._rz_var = tk.DoubleVar()
<<<<<<< HEAD
        self._prev_tx = 0.0
        self._prev_ty = 0.0
        self._prev_tz = 0.0
        self._syncing_sliders = False
=======
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

        self._photo_rgb = None
        self._photo_depth = None

        self._mesh_dir = self._resolve_mesh_dir()
        self._voxel_size = float(self.p7_cfg.get("voxel_size", 0.01))
        self._points_per_mesh = int(self.p7_cfg.get("points_per_mesh", 8000))
        self._max_corr = float(self.p7_cfg.get("max_correspondence_distance", 0.04))
        self._max_iters = int(self.p7_cfg.get("max_iterations", 80))
        self._point_to_plane = bool(self.p7_cfg.get("point_to_plane", True))
        self._auto_update_settings = bool(self.p7_cfg.get("auto_update_settings", True))

        self._t_cam_to_arm = _safe_t_cam2arm(cfg)
        self._t_arm_to_cam = np.linalg.inv(self._t_cam_to_arm)
<<<<<<< HEAD
        self._ctrl_t_arm_to_cam = self._t_arm_to_cam.copy()
        self._arm_model_points = self._load_arm_model_points()
        self.viewer.set_model_points(self._arm_model_points)

        # Initialize slider values from control pose (arm->cam in camera frame)
        self._tx_var.set(float(self._ctrl_t_arm_to_cam[0, 3]))
        self._ty_var.set(float(self._ctrl_t_arm_to_cam[1, 3]))
        self._tz_var.set(float(self._ctrl_t_arm_to_cam[2, 3]))
        rx, ry, rz = _matrix_to_euler_xyz_deg(self._ctrl_t_arm_to_cam)
        self._rx_var.set(rx)
        self._ry_var.set(ry)
        self._rz_var.set(rz)
        self._prev_tx = self._tx_var.get()
        self._prev_ty = self._ty_var.get()
        self._prev_tz = self._tz_var.get()
=======
        self._arm_model_points = self._load_arm_model_points()
        self.viewer.set_model_points(self._arm_model_points)

        # Initialize slider values from current T_cam2arm
        self._tx_var.set(float(self._t_cam_to_arm[0, 3]))
        self._ty_var.set(float(self._t_cam_to_arm[1, 3]))
        self._tz_var.set(float(self._t_cam_to_arm[2, 3]))
        rx, ry, rz = _matrix_to_euler_xyz_deg(self._t_cam_to_arm)
        self._rx_var.set(rx)
        self._ry_var.set(ry)
        self._rz_var.set(rz)
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

        self._build_ui()

    def _resolve_mesh_dir(self) -> str:
        mesh_dir = self.p7_cfg.get("mesh_dir", "assets/ra605_710")
        if not os.path.isabs(mesh_dir):
            return os.path.join(self.root_dir, mesh_dir)
        return mesh_dir

    def _load_arm_model_points(self) -> np.ndarray:
        try:
            model = load_model_from_mesh_dir(
                self._mesh_dir,
                points_per_mesh=self._points_per_mesh,
                voxel_size=self._voxel_size,
            )
            return np.asarray(model.points, dtype=np.float64)
        except Exception as exc:
            self._status.set(f"Model load failed: {exc}")
            return np.zeros((0, 3), dtype=np.float64)

    def _build_ui(self) -> None:
        self.root.title("Phase 7 - Arm ICP GUI")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1460x900")

        top = tk.Frame(self.root, bg=DARK_BG)
        top.pack(fill="x", padx=12, pady=8)

        tk.Button(top, text="Start", command=self.start, **BTN_ACCENT).pack(side="left", padx=4)
        tk.Button(top, text="Stop", command=self.stop, **BTN_DANGER).pack(side="left", padx=4)
        tk.Button(top, textvariable=self._viewer_btn_text, command=self.toggle_viewer, **BTN_ACCENT).pack(side="left", padx=4)
        tk.Button(top, text="Save Snapshot", command=self.save_snapshot, **BTN_ACCENT).pack(side="left", padx=4)
        tk.Button(top, textvariable=self._solve_btn_text, command=self.on_cpi_solve, **BTN_ACCENT).pack(side="left", padx=10)

        tk.Label(top, textvariable=self._status, bg=DARK_BG, fg="#dddddd", font=("Helvetica", 11)).pack(side="left", padx=12)
        tk.Label(top, textvariable=self._metrics, bg=DARK_BG, fg="#8fd18f", font=("Helvetica", 10)).pack(side="left", padx=10)

        tk.Label(self.root, textvariable=self._icp_metrics, bg=DARK_BG, fg="#f5c06a", font=("Helvetica", 11)).pack(
            fill="x", padx=14, pady=(0, 8)
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

        hint = "Virtual arm overlay uses arm.T_cam2arm from settings.yaml. Click cpi solve to run ICP."
        tk.Label(self.root, text=hint, bg=DARK_BG, fg="#9c9c9c", font=("Helvetica", 10)).pack(pady=(0, 8))

        # Manual adjustment controls
        controls = tk.Frame(self.root, bg=DARK_BG)
        controls.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(controls, text="Manual Arm Pose Adjustment", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 12, "bold")).pack(pady=(0, 8))

        # Translation sliders
        trans_frame = tk.Frame(controls, bg=DARK_BG)
        trans_frame.pack(fill="x", pady=(0, 8))
        tk.Label(trans_frame, text="Translation (m):", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 11)).pack(side="left", padx=(0, 10))

        tk.Label(trans_frame, text="Tx", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(trans_frame, from_=-2.0, to=2.0, resolution=0.01, orient="horizontal", variable=self._tx_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

        tk.Label(trans_frame, text="Ty", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(trans_frame, from_=-2.0, to=2.0, resolution=0.01, orient="horizontal", variable=self._ty_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

        tk.Label(trans_frame, text="Tz", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(trans_frame, from_=-2.0, to=2.0, resolution=0.01, orient="horizontal", variable=self._tz_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

        # Rotation sliders
        rot_frame = tk.Frame(controls, bg=DARK_BG)
        rot_frame.pack(fill="x", pady=(0, 8))
        tk.Label(rot_frame, text="Rotation (deg):", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 11)).pack(side="left", padx=(0, 10))

        tk.Label(rot_frame, text="Rx", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(rot_frame, from_=-180.0, to=180.0, resolution=1.0, orient="horizontal", variable=self._rx_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

        tk.Label(rot_frame, text="Ry", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(rot_frame, from_=-180.0, to=180.0, resolution=1.0, orient="horizontal", variable=self._ry_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

        tk.Label(rot_frame, text="Rz", bg=DARK_BG, fg="#ffffff", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        tk.Scale(rot_frame, from_=-180.0, to=180.0, resolution=1.0, orient="horizontal", variable=self._rz_var, command=self._on_slider_change, bg=DARK_BG, fg="#ffffff", highlightthickness=0).pack(side="left", padx=(0, 10))

<<<<<<< HEAD
        tk.Button(controls, text="Save", command=self._save_current_pose, bg="#2d7f2d", fg="#ffffff", font=("Helvetica", 11, "bold"), relief="raised", bd=2, padx=12, pady=6).pack(side="right", padx=(10, 0))
        tk.Button(controls, text="Apply Pose", command=self._commit_ctrl_pose, **BTN_ACCENT).pack(side="right", padx=(10, 0))
=======
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761
        tk.Button(controls, text="Reset to Settings", command=self._reset_to_settings, **BTN_ACCENT).pack(side="right", padx=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_slider_change(self, _event=None) -> None:
<<<<<<< HEAD
        if self._syncing_sliders:
            return

=======
        """Update T_cam_to_arm from slider values and refresh viewer."""
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761
        tx = self._tx_var.get()
        ty = self._ty_var.get()
        tz = self._tz_var.get()
        rx = self._rx_var.get()
        ry = self._ry_var.get()
        rz = self._rz_var.get()

<<<<<<< HEAD
        # Build rotation matrix from slider Euler (degrees)
        r = euler_xyz_deg_to_matrix(rx, ry, rz)

        # Translation sliders are applied as local-axis increments so motion follows
        # the currently rotated arm frame (not fixed camera world axes).
        d_local = np.array(
            [tx - self._prev_tx, ty - self._prev_ty, tz - self._prev_tz],
            dtype=np.float64,
        )
        self._prev_tx = tx
        self._prev_ty = ty
        self._prev_tz = tz

        T_arm_to_cam = self._ctrl_t_arm_to_cam.copy()
        T_arm_to_cam[:3, :3] = r
        T_arm_to_cam[:3, 3] = T_arm_to_cam[:3, 3] + (r @ d_local)

        # Control pose only: do not overwrite committed calibration until user applies.
        self._ctrl_t_arm_to_cam = T_arm_to_cam

        # Update viewer (viewer expects arm->cam)
        self.viewer.set_t_arm_to_cam(self._ctrl_t_arm_to_cam)

    def _save_current_pose(self) -> None:
        """Save current control pose to settings.yaml without changing committed state."""
        # Compute T_cam2arm from current control pose and save
        t_cam_to_arm = np.linalg.inv(self._ctrl_t_arm_to_cam)
        _update_settings_t_cam_to_arm(self.settings_path, t_cam_to_arm.tolist())
        self._status.set("Current pose saved to settings.yaml")

    def _commit_ctrl_pose(self) -> None:
        """Commit current control pose and convert it to T_cam2arm."""
        self._t_arm_to_cam = self._ctrl_t_arm_to_cam.copy()
        self._t_cam_to_arm = np.linalg.inv(self._t_arm_to_cam)
        if self._auto_update_settings:
            _update_settings_t_cam_to_arm(self.settings_path, self._t_cam_to_arm.tolist())
            self._status.set("Pose applied and saved to settings")
        else:
            self._status.set("Pose applied")
=======
        # Update transformation matrix
        r = euler_xyz_deg_to_matrix(rx, ry, rz)
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = r
        t[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

        self._t_cam_to_arm = t
        self._t_arm_to_cam = np.linalg.inv(t)

        # Update viewer with new transform
        self.viewer.set_t_arm_to_cam(self._t_arm_to_cam)
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

    def _reset_to_settings(self) -> None:
        """Reset sliders to values from settings.yaml."""
        # Load config from settings file
        settings_path = self.settings_path
        if not os.path.isabs(settings_path):
            settings_path = os.path.join(self.root_dir, settings_path)
        
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        t_cam_to_arm = _safe_t_cam2arm(cfg)
<<<<<<< HEAD
        self._t_cam_to_arm = t_cam_to_arm
        self._t_arm_to_cam = np.linalg.inv(t_cam_to_arm)
        self._ctrl_t_arm_to_cam = self._t_arm_to_cam.copy()

        self._syncing_sliders = True
        self._tx_var.set(float(self._ctrl_t_arm_to_cam[0, 3]))
        self._ty_var.set(float(self._ctrl_t_arm_to_cam[1, 3]))
        self._tz_var.set(float(self._ctrl_t_arm_to_cam[2, 3]))
        rx, ry, rz = _matrix_to_euler_xyz_deg(self._ctrl_t_arm_to_cam)
        self._rx_var.set(rx)
        self._ry_var.set(ry)
        self._rz_var.set(rz)
        self._syncing_sliders = False

        self._prev_tx = self._tx_var.get()
        self._prev_ty = self._ty_var.get()
        self._prev_tz = self._tz_var.get()
        self.viewer.set_t_arm_to_cam(self._ctrl_t_arm_to_cam)
=======

        self._tx_var.set(float(t_cam_to_arm[0, 3]))
        self._ty_var.set(float(t_cam_to_arm[1, 3]))
        self._tz_var.set(float(t_cam_to_arm[2, 3]))
        rx, ry, rz = _matrix_to_euler_xyz_deg(t_cam_to_arm)
        self._rx_var.set(rx)
        self._ry_var.set(ry)
        self._rz_var.set(rz)

        self._t_cam_to_arm = t_cam_to_arm
        self._t_arm_to_cam = np.linalg.inv(t_cam_to_arm)
        self.viewer.set_t_arm_to_cam(self._t_arm_to_cam)
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

    def start(self) -> None:
        if self._running:
            return
        try:
            self.provider.start()
        except Exception as exc:
            messagebox.showerror("Phase 7", f"Failed to start RealSense: {exc}")
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
        self.provider.stop()
        self.viewer.stop()
        self._viewer_enabled = False
        self._viewer_btn_text.set("Open Point Cloud")
        if not self._solving:
            self._status.set("Stopped")

    def toggle_viewer(self) -> None:
        if self._viewer_enabled:
            self.viewer.stop()
            self._viewer_enabled = False
            self._viewer_btn_text.set("Open Point Cloud")
            if self._running:
                self._status.set("Live")
            return

        self.viewer.start()
        self._viewer_enabled = True
        self._viewer_btn_text.set("Close Point Cloud")
        if self._running:
            self._status.set("Live + 3D")
        else:
            self._status.set("3D viewer opened (camera not started)")

    def _loop(self) -> None:
        intr = self.provider.intrinsics
        min_depth = float(self.p8_cfg.get("min_depth", 0.15))
        max_depth = float(self.p8_cfg.get("max_depth", 5.0))
        stride = int(self.p8_cfg.get("cloud_stride", 2))

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

            fallback_used = False
            if points.shape[0] == 0 and valid_ratio > 0.01:
                depth_valid_vals = depth[valid_depth]
                auto_min = max(0.03, float(depth_valid_vals.min()) - 0.05)
                auto_max = min(10.0, float(depth_valid_vals.max()) + 0.1)
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

            if self._viewer_enabled and points.shape[0] > 0:
<<<<<<< HEAD
                self.viewer.update(points, colors, self._ctrl_t_arm_to_cam)
=======
                self.viewer.update(points, colors, self._t_arm_to_cam)
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

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
            max_d = float(self.p8_cfg.get("max_depth", 5.0))
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

        out_dir = os.path.join(self.root_dir, "phase7_arm_icp", "outputs", "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        color_path = os.path.join(out_dir, f"color_{ts}.png")
        depth_path = os.path.join(out_dir, f"depth_{ts}.npy")
        cv2.imwrite(color_path, self._latest_color)
        np.save(depth_path, self._latest_depth)

        if self._latest_points is not None and self._latest_points.size > 0:
            npz_path = os.path.join(out_dir, f"cloud_{ts}.npz")
            np.savez(npz_path, points=self._latest_points.astype(np.float32), colors=self._latest_colors.astype(np.float32))
            self._status.set(
                f"Saved: {os.path.basename(color_path)}, {os.path.basename(depth_path)}, {os.path.basename(npz_path)}"
            )
        else:
            self._status.set(f"Saved: {os.path.basename(color_path)}, {os.path.basename(depth_path)}")

    def on_cpi_solve(self) -> None:
        if self._solving:
            return
        if self._latest_points is None or self._latest_points.shape[0] == 0:
            self._status.set("No live cloud to solve")
            return

        self._solving = True
        self._solve_btn_text.set("solving...")
        self._status.set("Running cpi solve (ICP)...")
        worker = threading.Thread(target=self._solve_worker, daemon=True)
        worker.start()

    def _solve_worker(self) -> None:
        try:
            points = np.asarray(self._latest_points, dtype=np.float64)
            colors = np.asarray(self._latest_colors, dtype=np.float64)
            cloud_path = _save_temp_cloud_npz(self.root_dir, points, colors)

<<<<<<< HEAD
            init_rxyz = _matrix_to_euler_xyz_deg(self._ctrl_t_arm_to_cam)
            init_txyz = tuple(float(v) for v in self._ctrl_t_arm_to_cam[:3, 3])
=======
            init_rxyz = _matrix_to_euler_xyz_deg(self._t_arm_to_cam)
            init_txyz = tuple(float(v) for v in self._t_arm_to_cam[:3, 3])
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

            result = calibrate_cam_to_arm(
                mesh_dir=self._mesh_dir,
                target_cloud_path=cloud_path,
                init_translation_xyz=init_txyz,
                init_rotation_xyz_deg=init_rxyz,
                voxel_size=self._voxel_size,
                points_per_mesh=self._points_per_mesh,
                crop_min_xyz=None,
                crop_max_xyz=None,
                max_correspondence_distance=self._max_corr,
                max_iterations=self._max_iters,
                point_to_plane=self._point_to_plane,
            )

            result_json = self.p7_cfg.get("result_json", "phase7_arm_icp/outputs/arm_icp_result.json")
            if not os.path.isabs(result_json):
                result_json = os.path.join(self.root_dir, result_json)
            save_result_json(result_json, result)

            if self._auto_update_settings:
                _update_settings_t_cam_to_arm(self.settings_path, result.t_cam_to_arm.tolist())

            self.root.after(0, lambda: self._on_solve_done(result, result_json, None))
        except Exception as exc:
            self.root.after(0, lambda: self._on_solve_done(None, None, exc))

    def _on_solve_done(self, result, result_json: str | None, err: Exception | None) -> None:
        self._solving = False
        self._solve_btn_text.set("cpi solve")

        if err is not None:
            self._status.set(f"cpi solve failed: {err}")
            return

        self._t_cam_to_arm = np.asarray(result.t_cam_to_arm, dtype=np.float64)
        self._t_arm_to_cam = np.asarray(result.t_arm_to_cam, dtype=np.float64)
<<<<<<< HEAD
        self._ctrl_t_arm_to_cam = self._t_arm_to_cam.copy()

        self._tx_var.set(float(self._ctrl_t_arm_to_cam[0, 3]))
        self._ty_var.set(float(self._ctrl_t_arm_to_cam[1, 3]))
        self._tz_var.set(float(self._ctrl_t_arm_to_cam[2, 3]))
        rx, ry, rz = _matrix_to_euler_xyz_deg(self._ctrl_t_arm_to_cam)
        self._rx_var.set(rx)
        self._ry_var.set(ry)
        self._rz_var.set(rz)

        if self._viewer_enabled and self._latest_points is not None and self._latest_points.shape[0] > 0:
            self.viewer.update(self._latest_points, self._latest_colors, self._ctrl_t_arm_to_cam)
=======

        if self._viewer_enabled and self._latest_points is not None and self._latest_points.shape[0] > 0:
            self.viewer.update(self._latest_points, self._latest_colors, self._t_arm_to_cam)
>>>>>>> e93f5c8bf53df2355df05c465428c6fae9f2e761

        self._icp_metrics.set(f"ICP: fitness={result.icp.fitness:.6f} | rmse={result.icp.inlier_rmse:.6f}")

        if self._auto_update_settings:
            self._status.set(f"cpi solve done, saved + updated settings ({os.path.basename(result_json)})")
        else:
            self._status.set(f"cpi solve done, saved result ({os.path.basename(result_json)})")

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()


def _matrix_to_euler_xyz_deg(mat: np.ndarray) -> tuple[float, float, float]:
    # Assumes mat is 4x4 transformation matrix, returns intrinsic XYZ Euler in degrees
    r = mat[:3, :3]
    # Prevent numerical issues
    sy = -r[0, 2]
    ay = np.arcsin(np.clip(sy, -1.0, 1.0))
    cy = np.cos(ay)
    if abs(cy) > 1e-6:
        ax = np.arctan2(r[1, 2], r[2, 2])
        az = np.arctan2(r[0, 1], r[0, 0])
    else:
        # Gimbal lock
        ax = np.arctan2(-r[2, 1], r[1, 1])
        az = 0.0
    return tuple(np.rad2deg([ax, ay, az]))


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert Euler angles in degrees to rotation matrix (intrinsic XYZ convention)."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    r = np.array([
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
    ], dtype=np.float64)
    return r


def run_phase7_arm_icp_gui(cfg: dict, root_dir: str, settings_path: str) -> None:
    root = tk.Tk()
    Phase7ArmIcpGUI(root, cfg, root_dir=root_dir, settings_path=settings_path)
    root.mainloop()
