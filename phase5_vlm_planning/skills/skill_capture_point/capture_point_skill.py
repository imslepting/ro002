"""CapturePoint 技能：使用 GraspGen 估計 6-DoF 抓取位姿

數據流：
    SAM3 best_mask + metric depth → mask 裁剪點雲 (M,3)
    → T_cam2arm 座標變換 → workspace 過濾
    → GraspGen 推理（內部自動中心化 + 還原）
    → 碰撞過濾（可選）
    → grasp_pose_arm (4×4) + grasp_score

GraspGen API（實際）：
    - load_grasp_cfg(gripper_yml_path) → OmegaConf config
    - GraspGenSampler(cfg) → sampler（載入 diffusion + discriminator 到 CUDA）
    - GraspGenSampler.run_inference(object_pc, sampler, ...) → (grasps (K,4,4), conf (K,))
    - sample() 內部會做 mean-centering 並在輸出時加回 centroid
    - 碰撞過濾需另外呼叫 filter_colliding_grasps()
    - 夾爪寬度是固定屬性，從 gripper config YAML 讀取
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from shared.types import CapturePointResult
from . import pointcloud_cropper
from . import grasp_visualizer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("skill_capture_point", {})


class CapturePointSkill:
    """GraspGen 6-DoF 抓取位姿估計技能封裝"""

    def __init__(self, device: str = "cuda"):
        """載入 GraspGen 模型到指定裝置

        Parameters
        ----------
        device : 'cuda' or 'cpu'
        """
        self._cfg = _load_config()
        self._device = device

        logger.info("Loading GraspGen model on %s ...", device)

        # 延遲導入 GraspGen
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info

        gripper_config = self._cfg.get(
            "gripper_config",
            "external/GraspGen/models/checkpoints/graspgen_robotiq_2f_140.yml",
        )
        gripper_config_path = str(_PROJECT_ROOT / gripper_config)

        # 載入 config + 建立 sampler
        grasp_cfg = load_grasp_cfg(gripper_config_path)
        self._sampler = GraspGenSampler(grasp_cfg)
        self._grasp_cfg = grasp_cfg

        # 夾爪固定屬性
        gripper_name = grasp_cfg.data.gripper_name
        self._gripper_info = get_gripper_info(gripper_name)
        self._gripper_width = float(grasp_cfg.data.get("width", 0.136))  # fallback

        # 從 gripper YAML 讀取 width（如有）
        gripper_yaml_path = (
            Path(__file__).resolve().parents[3]
            / "external" / "GraspGen" / "config" / "grippers"
            / f"{gripper_name}.yaml"
        )
        if gripper_yaml_path.exists():
            with open(gripper_yaml_path) as f:
                gconf = yaml.safe_load(f)
            if "width" in gconf:
                self._gripper_width = float(gconf["width"])

        self._num_candidates = self._cfg.get("num_candidates", 10)
        self._num_grasps = self._cfg.get("num_grasps", 200)
        self._filter_collisions = self._cfg.get("filter_collisions", True)
        self._workspace_limits = self._cfg.get("workspace_limits", {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.0, 0.6],
        })
        # 桌面高度過濾：自動從場景點雲估計，或用固定值
        # "auto" = 自動估計（場景點雲的下百分位），數值 = 固定高度
        self._table_height = self._cfg.get("table_height", "auto")

        # approach 方向約束：只保留接近指定方向的 grasp
        # approach_direction: [x, y, z] 目標方向（arm 座標系），null=不約束
        # approach_threshold: 內積閾值 (0~1)，越大越嚴格，0.5≈60°
        self._approach_direction = self._cfg.get("approach_direction", None)
        self._approach_threshold = self._cfg.get("approach_threshold", 0.5)

        logger.info(
            "GraspGen loaded. gripper=%s, width=%.4f m",
            gripper_name, self._gripper_width,
        )

    def capture(
        self,
        rgb_image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        T_cam2arm: np.ndarray,
        scene_points_arm: np.ndarray | None = None,
    ) -> CapturePointResult:
        """估計 6-DoF 抓取位姿

        Parameters
        ----------
        rgb_image : BGR uint8 (H, W, 3)
        depth : float32 (H, W) metric depth（公尺）
        mask : bool (H, W) 來自 SAM3 best_mask
        K : 3×3 相機內參矩陣
        T_cam2arm : 4×4 相機→機械臂座標變換
        scene_points_arm : (N, 3) 場景點雲（arm 座標系），用於碰撞過濾。
            若為 None 且 filter_collisions=True，則從 depth 全圖生成。

        Returns
        -------
        CapturePointResult
        """
        from grasp_gen.grasp_server import GraspGenSampler

        # 1. mask → 裁剪點雲（相機座標系）= 物件點雲
        points_cam, colors = pointcloud_cropper.mask_to_pointcloud(
            depth, K, mask, rgb_image,
        )
        logger.info("Cropped point cloud: %d points", len(points_cam))

        if len(points_cam) < 10:
            logger.warning("Too few points (%d), returning empty result", len(points_cam))
            return self._empty_result(rgb_image, len(points_cam))

        # 2. 座標變換 cam → arm
        points_arm = pointcloud_cropper.transform_points(points_cam, T_cam2arm)

        # 3. workspace 過濾
        points_arm, colors = pointcloud_cropper.filter_workspace(
            points_arm, colors, self._workspace_limits,
        )
        logger.info("After workspace filter: %d points", len(points_arm))

        if len(points_arm) < 10:
            logger.warning("Too few points after filtering (%d)", len(points_arm))
            return self._empty_result(rgb_image, len(points_arm))

        # 4. GraspGen 推理
        #    run_inference 內部的 sample() 會自動做 mean-centering
        #    並在輸出 grasps[:, :3, 3] 加回 centroid
        grasps, grasp_conf = GraspGenSampler.run_inference(
            object_pc=points_arm.astype(np.float32),
            grasp_sampler=self._sampler,
            grasp_threshold=-1.0,  # 不過濾，取 topk
            num_grasps=self._num_grasps,
            topk_num_grasps=self._num_candidates,
        )

        # grasps: torch.Tensor (K, 4, 4), grasp_conf: torch.Tensor (K,)
        if len(grasps) == 0:
            logger.warning("GraspGen returned 0 candidates")
            return self._empty_result(rgb_image, len(points_arm))

        grasps_np = grasps.cpu().numpy()       # (K, 4, 4)
        scores_np = grasp_conf.cpu().numpy()    # (K,)

        # 5. 過濾鏈 — 每步保存中間結果，fallback 只退回上一步
        # 5a. 碰撞過濾（可選）
        if self._filter_collisions:
            prev_grasps, prev_scores = grasps_np.copy(), scores_np.copy()
            grasps_np, scores_np = self._apply_collision_filter(
                grasps_np, scores_np, points_arm,
                depth, K, mask, T_cam2arm, scene_points_arm,
            )
            if len(grasps_np) == 0:
                logger.warning("All grasps filtered by collision, reverting")
                grasps_np, scores_np = prev_grasps, prev_scores

        # 5b. 桌面高度過濾
        prev_grasps, prev_scores = grasps_np.copy(), scores_np.copy()
        grasps_np, scores_np = self._apply_table_filter(
            grasps_np, scores_np, points_arm,
        )
        if len(grasps_np) == 0:
            logger.warning("All grasps filtered by table height, reverting")
            grasps_np, scores_np = prev_grasps, prev_scores

        # 5c. approach 方向過濾（內建漸進式放寬，保證至少返回 1 個結果）
        grasps_np, scores_np = self._apply_approach_filter(
            grasps_np, scores_np,
        )

        n_candidates = len(scores_np)

        # 6. 按 score 排序（高→低），取最佳
        order = np.argsort(-scores_np)
        grasps_np = grasps_np[order]
        scores_np = scores_np[order]

        best_pose = grasps_np[0].copy()  # (4, 4) arm 座標系
        best_score = float(scores_np[0])

        # 7. 反投影到像素座標 → 轉回相機座標系
        T_arm2cam = np.linalg.inv(T_cam2arm)

        # 用接觸點（而非 TCP）投影，讓可視化標記落在物體上
        contact_arm = grasp_visualizer.compute_contact_point(best_pose, points_arm)
        contact_cam = pointcloud_cropper.transform_points(
            contact_arm.reshape(1, 3), T_arm2cam,
        )[0]
        grasp_pixel = grasp_visualizer.project_point_to_pixel(contact_cam, K)

        # 8. 計算投影的夾爪寬度（像素）和旋轉角
        z_cam = contact_cam[2]
        grasp_width_px = (
            self._gripper_width * K[0, 0] / z_cam if z_cam > 0 else 50.0
        )

        # 使用 grasp 旋轉矩陣在相機座標系的 x 軸方向投影
        R_grasp_arm = best_pose[:3, :3]
        R_arm2cam = T_arm2cam[:3, :3]
        R_grasp_cam = R_arm2cam @ R_grasp_arm
        gripper_dir_cam = R_grasp_cam[:, 0]  # x 軸 = 夾爪張開方向
        rotation_2d = float(np.arctan2(gripper_dir_cam[1], gripper_dir_cam[0]))

        # 9. 可視化
        # TCP 像素（用於畫 approach 箭頭）
        tcp_cam = pointcloud_cropper.transform_points(
            best_pose[:3, 3].reshape(1, 3), T_arm2cam,
        )[0]
        tcp_pixel = grasp_visualizer.project_point_to_pixel(tcp_cam, K)

        annotated = grasp_visualizer.annotate_grasp(
            rgb_image,
            grasp_pixel=grasp_pixel,
            grasp_score=best_score,
            grasp_width_px=grasp_width_px,
            rotation_2d=rotation_2d,
            label="grasp",
            tcp_pixel=tcp_pixel,
        )

        return CapturePointResult(
            pose_arm=best_pose,
            grasp_width=self._gripper_width,
            grasp_score=best_score,
            grasp_pixel=grasp_pixel,
            num_candidates=n_candidates,
            annotated_image=annotated,
            cropped_cloud_size=len(points_arm),
        )

    def _apply_collision_filter(
        self,
        grasps_np: np.ndarray,
        scores_np: np.ndarray,
        object_points_arm: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        T_cam2arm: np.ndarray,
        scene_points_arm: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """碰撞過濾：移除與場景點雲碰撞的 grasps"""
        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

        # 如果沒有提供場景點雲，從 depth 全圖生成（排除物件 mask）
        if scene_points_arm is None:
            scene_mask = ~mask & (depth > 0)
            scene_pts_cam, _ = pointcloud_cropper.mask_to_pointcloud(
                depth, K, scene_mask,
            )
            if len(scene_pts_cam) > 0:
                scene_points_arm = pointcloud_cropper.transform_points(
                    scene_pts_cam, T_cam2arm,
                )
            else:
                logger.warning("No scene points for collision filter, skipping")
                return grasps_np, scores_np

        # 降採樣場景點雲（加速碰撞檢查）
        max_scene_pts = 8192
        if len(scene_points_arm) > max_scene_pts:
            indices = np.random.choice(
                len(scene_points_arm), max_scene_pts, replace=False,
            )
            scene_points_arm = scene_points_arm[indices]

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_points_arm,
            grasp_poses=grasps_np,
            gripper_collision_mesh=self._gripper_info.collision_mesh,
            collision_threshold=0.02,
        )

        logger.info(
            "Collision filter: %d/%d collision-free",
            collision_free_mask.sum(), len(grasps_np),
        )

        return grasps_np[collision_free_mask], scores_np[collision_free_mask]

    def _apply_table_filter(
        self,
        grasps_np: np.ndarray,
        scores_np: np.ndarray,
        object_points_arm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """過濾 TCP 位於桌面/支撐面以下的 grasp

        點雲只有表面，桌面以下沒有點 → 碰撞檢查會通過，但夾爪實際上會撞桌子。
        用場景點雲的下百分位估計桌面高度，拒絕 TCP 低於此高度的 grasp。
        """
        if len(grasps_np) == 0:
            return grasps_np, scores_np

        # 估計或使用固定桌面高度
        if self._table_height == "auto":
            # T_cam2arm=Identity 時 arm 座標 = 相機座標：
            #   x=右, y=下, z=前（深度）
            # 桌面是場景中 y 值最大的平面（最低處）
            # 但一般設定下 arm z=上，桌面 z 最小
            # 用物體點雲各軸的 5th percentile 作為「底部」
            table_z = float(np.percentile(object_points_arm[:, 2], 5))
            table_y = float(np.percentile(object_points_arm[:, 1], 95))
            table_x_lo = float(np.percentile(object_points_arm[:, 0], 5))
            table_x_hi = float(np.percentile(object_points_arm[:, 0], 95))
        else:
            # 固定值：假設 arm 座標 z 軸為高度
            table_z = float(self._table_height)
            table_y = None
            table_x_lo = None
            table_x_hi = None

        # 取每個 grasp 的 TCP 位置
        tcp_positions = grasps_np[:, :3, 3]  # (K, 3)

        # 也檢查夾爪指尖位置（TCP 沿 approach 方向延伸 gripper_depth）
        approach_dirs = grasps_np[:, :3, 2]  # (K, 3) z 軸
        gripper_depth = 0.195
        fingertip_positions = tcp_positions + approach_dirs * gripper_depth

        if self._table_height == "auto":
            # 物體底部以下的都拒絕（TCP 或指尖任一低於物體底部）
            # 用各軸獨立判斷：主要看 z（深度方向）超出物體範圍
            # 簡化：拒絕 TCP 或指尖的任一座標超出物體 bounding box + margin
            margin = 0.02  # 2cm 容差
            keep = np.ones(len(grasps_np), dtype=bool)
            for pos in [tcp_positions, fingertip_positions]:
                # z 軸（通常是深度或高度）不低於物體底部
                keep &= pos[:, 2] >= (table_z - margin)
                # y 軸不超過物體底部（相機座標系 y=下）
                if table_y is not None:
                    keep &= pos[:, 1] <= (table_y + margin)
        else:
            # 固定桌面高度：TCP 和指尖的 z 都必須高於桌面
            keep = (tcp_positions[:, 2] >= table_z) & (fingertip_positions[:, 2] >= table_z)

        n_before = len(grasps_np)
        filtered_grasps = grasps_np[keep]
        filtered_scores = scores_np[keep]

        logger.info(
            "Table filter: %d/%d above surface (table_z=%.3f)",
            keep.sum(), n_before, table_z,
        )

        return filtered_grasps, filtered_scores

    def _apply_approach_filter(
        self, grasps_np: np.ndarray, scores_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """過濾 approach 方向不符合約束的 grasp

        漸進式放寬：先用嚴格閾值，若全被過濾則逐步放寬。
        如果所有閾值都無法通過，選擇最接近目標方向的 grasp。
        保證至少返回 1 個結果。
        """
        if self._approach_direction is None:
            return grasps_np, scores_np
        if len(grasps_np) == 0:
            return grasps_np, scores_np

        target = np.array(self._approach_direction, dtype=np.float64)
        target = target / (np.linalg.norm(target) + 1e-12)

        approach_dirs = grasps_np[:, :3, 2]  # (K, 3)
        norms = np.linalg.norm(approach_dirs, axis=1, keepdims=True)
        approach_dirs = approach_dirs / (norms + 1e-12)

        dots = approach_dirs @ target  # (K,)

        # 漸進式放寬：嚴格 → 中等 → 寬鬆
        thresholds = [self._approach_threshold, 0.7, 0.5]
        for threshold in thresholds:
            keep = dots >= threshold
            if keep.any():
                logger.info(
                    "Approach filter: %d/%d pass (threshold=%.2f, target=%s)",
                    keep.sum(), len(grasps_np), threshold, target.tolist(),
                )
                return grasps_np[keep], scores_np[keep]

        # 所有閾值都無法通過 — 選最接近目標方向的 grasp
        best_idx = int(np.argmax(dots))
        logger.warning(
            "Approach filter: none pass, using most aligned "
            "(dot=%.3f, target=%s)",
            dots[best_idx], target.tolist(),
        )
        return grasps_np[best_idx:best_idx + 1], scores_np[best_idx:best_idx + 1]

    def warmup(self) -> None:
        """空跑一次暖機"""
        from grasp_gen.grasp_server import GraspGenSampler

        logger.info("CapturePoint warmup ...")
        dummy_pts = np.random.randn(1000, 3).astype(np.float32) * 0.1
        GraspGenSampler.run_inference(
            object_pc=dummy_pts,
            grasp_sampler=self._sampler,
            num_grasps=10,
            topk_num_grasps=5,
            min_grasps=1,
            max_tries=1,
        )
        logger.info("CapturePoint warmup done.")

    @staticmethod
    def _empty_result(rgb_image: np.ndarray, cloud_size: int) -> CapturePointResult:
        """返回空結果（點雲太少或無候選時）"""
        return CapturePointResult(
            pose_arm=np.eye(4),
            grasp_width=0.0,
            grasp_score=0.0,
            grasp_pixel=(0, 0),
            num_candidates=0,
            annotated_image=rgb_image.copy(),
            cropped_cloud_size=cloud_size,
        )
