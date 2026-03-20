"""Open3D 軌跡可視化 helpers

提供 EE 路徑線段、路點球體、碰撞體素、夾爪線框等 3D 可視化元素。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def compute_ee_path(
    waypoints,
    fk_fn: Callable[[list[float]], np.ndarray],
) -> np.ndarray:
    """每個 waypoint → FK → EE position → (T, 3)

    Parameters
    ----------
    waypoints : list[TrajectoryWaypoint]
    fk_fn : joints → 4×4 pose

    Returns
    -------
    np.ndarray : (T, 3) EE 位置序列
    """
    positions = []
    for wp in waypoints:
        pose = fk_fn(wp.joints)
        positions.append(pose[:3, 3].copy())
    return np.array(positions, dtype=np.float64)


def make_ee_path_lineset(ee_positions: np.ndarray):
    """漸變色線段 cyan→yellow（start→end）

    Parameters
    ----------
    ee_positions : (T, 3)

    Returns
    -------
    o3d.geometry.LineSet
    """
    import open3d as o3d

    n = len(ee_positions)
    if n < 2:
        return o3d.geometry.LineSet()

    lines = [[i, i + 1] for i in range(n - 1)]

    # 漸變色：cyan (0, 1, 1) → yellow (1, 1, 0)
    colors = []
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        r = t
        g = 1.0
        b = 1.0 - t
        colors.append([r, g, b])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(ee_positions)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_waypoint_spheres(
    ee_positions: np.ndarray,
    every_n: int = 10,
    radius: float = 0.004,
):
    """每 N 個路點放一個小球

    Parameters
    ----------
    ee_positions : (T, 3)
    every_n : 採樣間隔
    radius : 球體半徑

    Returns
    -------
    list[o3d.geometry.TriangleMesh]
    """
    import open3d as o3d

    spheres = []
    n = len(ee_positions)
    for i in range(0, n, every_n):
        t = i / max(n - 1, 1)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(ee_positions[i])
        # 漸變色同路徑
        r = t
        g = 1.0
        b = 1.0 - t
        sphere.paint_uniform_color([r, g, b])
        spheres.append(sphere)
    return spheres


def make_collision_voxel_cloud(
    points: np.ndarray,
    voxel_size: float,
):
    """紅色半透明體素網格

    Parameters
    ----------
    points : (N, 3) 碰撞點雲
    voxel_size : 體素大小

    Returns
    -------
    o3d.geometry.VoxelGrid
    """
    import open3d as o3d

    if len(points) == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0.8, 0.2, 0.2])  # 紅色

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size,
    )
    return voxel_grid


def make_gripper_at_config(
    pose: np.ndarray,
    gripper_width: float,
    gripper_depth: float,
    color: list[float],
):
    """夾爪線框（複用 capture_point 的 7 點模型）

    Parameters
    ----------
    pose : 4×4 SE(3) EE pose
    gripper_width : 夾爪張開寬度（公尺）
    gripper_depth : 夾爪深度（公尺）
    color : [R, G, B] 線框顏色

    Returns
    -------
    o3d.geometry.LineSet
    """
    import open3d as o3d

    hw = gripper_width / 2
    pts_local = np.array([
        [-hw * 0.3, 0, 0],     # 0: base left
        [hw * 0.3, 0, 0],      # 1: base right
        [-hw, 0, gripper_depth * 0.5],  # 2: left finger root
        [hw, 0, gripper_depth * 0.5],   # 3: right finger root
        [-hw, 0, gripper_depth],  # 4: left finger tip
        [hw, 0, gripper_depth],   # 5: right finger tip
        [0, 0, 0],              # 6: base center (TCP)
    ], dtype=np.float64)

    edges = [
        [0, 1],
        [0, 2], [1, 3],
        [2, 4], [3, 5],
        [6, 0], [6, 1],
    ]

    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = (R @ pts_local.T).T + t

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls
