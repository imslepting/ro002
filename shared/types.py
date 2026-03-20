"""RO002 共用數據類型定義"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Phase 0：硬體診斷 ──

@dataclass
class DetectedDevice:
    """掃描到的視頻設備"""
    index: int
    is_open: bool
    backend_name: str
    device_path: str  # e.g. "/dev/video0"


@dataclass
class CameraTestResult:
    """單台相機的測試結果"""
    cam_index: int
    resolution: tuple[int, int]       # (width, height)
    fps_reported: float
    fps_measured: float
    frame_success_rate: float         # 0.0 ~ 1.0
    mean_brightness: float            # 0 ~ 255
    sharpness_score: float            # Laplacian 方差
    is_color: bool
    status: str = "OK"               # "OK" / "WARNING" / "ERROR"
    warnings: list[str] = field(default_factory=list)


# ── Phase 1-2：標定結果 ──

@dataclass
class CalibResult:
    """單台相機內參標定結果"""
    cam_name: str
    K: np.ndarray              # 3x3 內參矩陣
    D: np.ndarray              # 畸變係數
    image_size: tuple[int, int]
    rms: float


@dataclass
class ExtrinsicsResult:
    """相機外參（相對於世界座標系 cam0）"""
    cam_name: str
    T_cam_to_world: np.ndarray  # 4x4 齊次變換矩陣
    rms: float


# ── Phase 3：重建結果 ──

@dataclass
class PointCloudResult:
    """融合後的度量點雲"""
    points: np.ndarray          # (N, 3) xyz
    colors: Optional[np.ndarray] = None  # (N, 3) rgb, 0-255
    timestamp: str = ""


# ── Phase 4：執行計劃 ──

@dataclass
class PlanStep:
    """單一執行步驟"""
    action: str                 # e.g. "move_to", "grasp", "release"
    target: Optional[np.ndarray] = None  # (x, y, z) 世界座標
    params: dict = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """VLM 生成的執行計劃"""
    task_description: str
    steps: list[PlanStep] = field(default_factory=list)
    timestamp: str = ""


# ── Phase 5：VLM Planning ──

@dataclass
class SAM3Result:
    """SAM3 分割結果"""
    masks: list[np.ndarray]           # list of (H,W) bool masks
    scores: list[float]               # 每個 mask 的置信度
    boxes: list[np.ndarray]           # 每個 mask 的 bbox [x1,y1,x2,y2]
    best_mask: np.ndarray             # 最高分 mask (H,W) bool
    best_score: float                 # 最高分
    annotated_image: np.ndarray       # 標注後的圖像 BGR uint8
    object_description: str           # 輸入的物件描述


@dataclass
class TrajectoryWaypoint:
    """單一軌跡路點"""
    joints: list[float]           # 關節角度（弧度），xArm7 = 7 個
    timestamp: float = 0.0       # 從軌跡起始的秒數


@dataclass
class TrajectoryPlanningResult:
    """cuRobo 軌跡規劃結果"""
    waypoints: list[TrajectoryWaypoint]
    estimated_duration_sec: float
    num_waypoints: int
    success: bool
    planning_time_sec: float
    start_joints: list[float]
    goal_pose_arm: np.ndarray         # 4×4 目標位姿
    collision_voxel_count: int
    error_message: str = ""


@dataclass
class CapturePointResult:
    """GraspGen 抓取位姿結果"""
    pose_arm: np.ndarray              # 4×4 arm 座標系下的抓取位姿
    grasp_width: float                # 夾爪張開寬度（公尺）
    grasp_score: float                # GraspGen 置信度
    grasp_pixel: tuple[int, int]      # (u, v) 像素座標（VLM 可視化用）
    num_candidates: int               # 候選抓取總數
    annotated_image: np.ndarray       # 標注後的圖像 BGR uint8
    cropped_cloud_size: int           # 裁剪後點雲大小（debug 用）
