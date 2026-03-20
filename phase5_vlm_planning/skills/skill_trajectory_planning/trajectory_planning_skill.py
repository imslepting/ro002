"""TrajectoryPlanning 技能：使用 cuRobo GPU 加速運動規劃

數據流：
    當前關節狀態 + GraspGen 目標位姿 + 場景碰撞點雲
    → cuRobo MotionGen（無碰撞軌跡規劃）
    → TrajectoryPlanningResult（waypoints + 持續時間）
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml

from shared.types import TrajectoryPlanningResult, TrajectoryWaypoint

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("skill_trajectory_planning", {})


class TrajectoryPlanningSkill:
    """cuRobo 軌跡規劃技能封裝"""

    def __init__(self, device: str = "cuda"):
        self._cfg = _load_config()
        self._device = device

        robot_config = self._cfg.get("robot_config", "config/robot_curobo.yml")
        if not Path(robot_config).is_absolute():
            robot_config = str(_PROJECT_ROOT / robot_config)

        self._collision_from_pointcloud = self._cfg.get("collision_from_pointcloud", True)
        self._voxel_size = self._cfg.get("voxel_size", 0.02)
        self._interpolation_steps = self._cfg.get("interpolation_steps", 100)

        logger.info("Loading TrajectoryPlanningSkill on %s ...", device)

        from .curobo_wrapper import CuroboPlanner

        self._planner = CuroboPlanner(
            robot_config_path=robot_config,
            device=device,
            interpolation_steps=self._interpolation_steps,
            voxel_size=self._voxel_size,
        )

        logger.info("TrajectoryPlanningSkill ready.")

    def plan(
        self,
        start_joints: list[float] | np.ndarray,
        goal_pose_arm: np.ndarray,
        scene_points_arm: np.ndarray | None = None,
    ) -> TrajectoryPlanningResult:
        """規劃從當前關節狀態到目標位姿的無碰撞軌跡

        Parameters
        ----------
        start_joints : 7-DoF 關節角度（弧度）
        goal_pose_arm : 4×4 SE(3) 目標位姿（arm 座標系）
        scene_points_arm : (N, 3) 場景碰撞點雲（arm 座標系），可選

        Returns
        -------
        TrajectoryPlanningResult
        """
        start_joints = list(np.asarray(start_joints, dtype=float).ravel())
        goal_pose_arm = np.asarray(goal_pose_arm, dtype=np.float64).reshape(4, 4)

        t0 = time.time()

        # 1. 更新碰撞世界
        collision_voxel_count = 0
        if self._collision_from_pointcloud and scene_points_arm is not None:
            collision_voxel_count = self._planner.update_world_voxels(scene_points_arm)

        # 2. 規劃
        result = self._planner.plan_single(start_joints, goal_pose_arm)
        planning_time = time.time() - t0

        # 3. 轉換為 TrajectoryPlanningResult
        if result.success:
            waypoints = []
            n = len(result.joint_positions)
            dt = result.estimated_duration / max(n, 1)
            for i, jp in enumerate(result.joint_positions):
                waypoints.append(TrajectoryWaypoint(
                    joints=jp.tolist(),
                    timestamp=i * dt,
                ))

            return TrajectoryPlanningResult(
                waypoints=waypoints,
                estimated_duration_sec=result.estimated_duration,
                num_waypoints=n,
                success=True,
                planning_time_sec=planning_time,
                start_joints=start_joints,
                goal_pose_arm=goal_pose_arm,
                collision_voxel_count=collision_voxel_count,
            )
        else:
            return TrajectoryPlanningResult(
                waypoints=[],
                estimated_duration_sec=0.0,
                num_waypoints=0,
                success=False,
                planning_time_sec=planning_time,
                start_joints=start_joints,
                goal_pose_arm=goal_pose_arm,
                collision_voxel_count=collision_voxel_count,
                error_message=result.error_message,
            )

    def get_fk(self, joints: list[float]) -> np.ndarray:
        """Forward Kinematics → 4×4 EE pose（供可視化用）"""
        return self._planner.forward_kinematics(joints)

    def warmup(self):
        """空跑一次暖機"""
        logger.info("TrajectoryPlanning warmup ...")
        dummy_result = self.plan(
            start_joints=[0.0] * 7,
            goal_pose_arm=np.eye(4),
        )
        logger.info(
            "TrajectoryPlanning warmup done (success=%s).",
            dummy_result.success,
        )
