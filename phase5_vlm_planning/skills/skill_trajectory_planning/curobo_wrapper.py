"""cuRobo API 封裝層 — 隔離 cuRobo 依賴，方便版本遷移

用法：
    planner = CuroboPlanner("config/robot_curobo.yml", "cuda", 100, 0.02)
    planner.update_world_voxels(scene_points)
    result = planner.plan_single(start_joints, goal_pose_4x4)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """cuRobo 規劃結果"""
    success: bool
    joint_positions: np.ndarray   # (T, num_joints)
    estimated_duration: float
    error_message: str = ""


class CuroboPlanner:
    """cuRobo MotionGen 封裝"""

    def __init__(
        self,
        robot_config_path: str,
        device: str = "cuda",
        interpolation_steps: int = 100,
        voxel_size: float = 0.02,
    ):
        import yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        from curobo.geom.types import WorldConfig

        self._device = device
        self._interpolation_steps = interpolation_steps
        self._voxel_size = voxel_size

        # 讀取 robot config
        with open(robot_config_path) as f:
            robot_cfg = yaml.safe_load(f)

        robot_config_name = robot_cfg.get("robot_config_name", "xarm7.yml")

        logger.info("Loading cuRobo MotionGen with %s on %s ...", robot_config_name, device)

        # 建立初始世界（需要至少一個障礙物才能 warmup）
        # 放一個遠處的小方塊，不會和任何 joint config 碰撞
        from curobo.geom.types import Cuboid
        dummy_obstacle = Cuboid(
            name="dummy",
            pose=[5.0, 5.0, 5.0, 1, 0, 0, 0],
            dims=[0.01, 0.01, 0.01],
        )
        initial_world = WorldConfig(cuboid=[dummy_obstacle])

        # 建立 MotionGenConfig（預分配碰撞緩存以容納點雲體素）
        # 注意：暫時禁用自碰撞檢查。xArm7 collision spheres 是近似的，
        # 在某些合法 config（如 zero config）會誤報自碰撞。
        # 世界碰撞（與場景點雲的碰撞避免）仍然啟用。
        # TODO: 使用精確的 xArm7 collision mesh 後再啟用自碰撞
        collision_cache = {"obb": 600, "mesh": 20}
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_name,
            world_model=initial_world,
            interpolation_dt=0.02,
            interpolation_steps=interpolation_steps,
            collision_cache=collision_cache,
            self_collision_check=False,
        )
        self._motion_gen = MotionGen(motion_gen_config)
        self._motion_gen.warmup()

        # 保存 KinematicsSolver 引用（FK 用）
        self._kin_solver = self._motion_gen.kinematics

        logger.info("cuRobo MotionGen ready.")

    def update_world_voxels(self, points: np.ndarray) -> int:
        """用點雲更新碰撞世界

        Parameters
        ----------
        points : (N, 3) float32, arm 座標系

        Returns
        -------
        int : 體素數量
        """
        import torch
        from curobo.geom.types import WorldConfig, Cuboid

        n_pts = len(points)
        if n_pts == 0:
            logger.warning("Empty point cloud, skipping world update")
            return 0

        # 用 cuRobo 的體素化方式建構碰撞世界
        # 降採樣 + 建構體素格
        voxel_size = self._voxel_size

        # 體素化：量化到格子
        quantized = np.floor(points / voxel_size).astype(np.int32)
        unique_voxels = np.unique(quantized, axis=0)
        n_voxels = len(unique_voxels)

        # 將每個體素轉為 cuboid
        cuboids = []
        centers = (unique_voxels.astype(np.float32) + 0.5) * voxel_size
        for i, center in enumerate(centers):
            cuboids.append(Cuboid(
                name=f"voxel_{i}",
                pose=[float(center[0]), float(center[1]), float(center[2]), 1, 0, 0, 0],
                dims=[voxel_size, voxel_size, voxel_size],
            ))

        # 限制 cuboid 數量（cuRobo 效能考量）
        max_cuboids = 500
        if len(cuboids) > max_cuboids:
            indices = np.random.choice(len(cuboids), max_cuboids, replace=False)
            cuboids = [cuboids[i] for i in indices]
            n_voxels = max_cuboids

        world_config = WorldConfig(cuboid=cuboids)
        self._motion_gen.update_world(world_config)

        logger.info("World updated: %d voxels (from %d points)", n_voxels, n_pts)
        return n_voxels

    def plan_single(
        self,
        start_joints: list[float] | np.ndarray,
        goal_pose: np.ndarray,
    ) -> PlanResult:
        """規劃從起始關節到目標位姿的無碰撞軌跡

        Parameters
        ----------
        start_joints : 7-DoF 關節角度（弧度）
        goal_pose : 4×4 SE(3) 目標位姿

        Returns
        -------
        PlanResult
        """
        import torch
        from curobo.types.robot import JointState
        from curobo.types.math import Pose

        device = self._device

        # 起始狀態
        js = torch.tensor(
            [start_joints], dtype=torch.float32, device=device,
        )
        start_state = JointState.from_position(js)

        # 目標位姿：拆分 position + quaternion
        R = goal_pose[:3, :3]
        t = goal_pose[:3, 3]
        quat = self._rotation_matrix_to_quaternion(R)  # [w, x, y, z]

        goal_position = torch.tensor(
            [[t[0], t[1], t[2]]], dtype=torch.float32, device=device,
        )
        goal_quaternion = torch.tensor(
            [[quat[0], quat[1], quat[2], quat[3]]], dtype=torch.float32, device=device,
        )
        goal = Pose(position=goal_position, quaternion=goal_quaternion)

        # 規劃
        try:
            result = self._motion_gen.plan_single(start_state, goal)

            if result.success[0]:
                # 取插值後的軌跡
                traj = result.get_interpolated_plan()
                joint_positions = traj.position.cpu().numpy()  # (T, num_joints)
                # 估計持續時間
                dt = 0.02  # interpolation_dt
                duration = len(joint_positions) * dt

                return PlanResult(
                    success=True,
                    joint_positions=joint_positions,
                    estimated_duration=duration,
                )
            else:
                status_str = str(result.status) if result.status is not None else "UNKNOWN"
                error_msg = f"cuRobo: {status_str}"
                logger.warning("Planning failed: %s", error_msg)
                return PlanResult(
                    success=False,
                    joint_positions=np.zeros((0, 7)),
                    estimated_duration=0.0,
                    error_message=error_msg,
                )
        except Exception as exc:
            logger.error("cuRobo plan_single exception: %s", exc)
            return PlanResult(
                success=False,
                joint_positions=np.zeros((0, 7)),
                estimated_duration=0.0,
                error_message=str(exc),
            )

    def forward_kinematics(self, joints: list[float]) -> np.ndarray:
        """Forward Kinematics → 4×4 EE pose

        Parameters
        ----------
        joints : 7-DoF 關節角度（弧度）

        Returns
        -------
        np.ndarray : 4×4 SE(3)
        """
        import torch

        js = torch.tensor(
            [joints], dtype=torch.float32, device=self._device,
        )
        state = self._kin_solver.get_state(js)

        # 取 EE pose
        ee_pos = state.ee_position[0].cpu().numpy()     # (3,)
        ee_quat = state.ee_quaternion[0].cpu().numpy()   # (4,) [w, x, y, z]

        # 組裝 4×4
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self._quaternion_to_rotation_matrix(ee_quat)
        pose[:3, 3] = ee_pos
        return pose

    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> list[float]:
        """3×3 → [w, x, y, z]（cuRobo 預設 quaternion 格式）"""
        # Shepperd's method
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return [w, x, y, z]

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """[w, x, y, z] → 3×3 rotation matrix"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
