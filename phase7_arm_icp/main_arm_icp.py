"""Phase 7 - Arm mesh ICP calibration (camera to arm extrinsics).

Usage example:
    # Live capture + ICP (no pre-saved phase8 cloud required)
    conda run -n ro002 python phase7_arm_icp/main_arm_icp.py

    # Offline mode using an existing cloud file
    conda run -n ro002 python phase7_arm_icp/main_arm_icp.py \
        --target-cloud phase8_realsense_pointcloud/outputs/cloud_xxx.ply
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase7_arm_icp.src.icp_calibrator import (  # noqa: E402
    calibrate_cam_to_arm,
    load_model_from_mesh_dir,
    load_target_cloud,
    preview_alignment,
    interactive_manual_alignment,
    save_result_json,
)
from phase8_realsense_pointcloud.src.pointcloud_builder import build_pointcloud  # noqa: E402
from phase8_realsense_pointcloud.src.realsense_provider import RealSenseProvider  # noqa: E402


def load_config(config_path: str) -> dict:
    path = config_path
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_xyz(value: list[float] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("Expected exactly 3 values")
    return float(value[0]), float(value[1]), float(value[2])


def _update_settings_t_cam_to_arm(settings_path: str, t_cam_to_arm: list[list[float]]) -> None:
    with open(settings_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("arm", {})
    cfg["arm"]["T_cam2arm"] = t_cam_to_arm

    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _capture_target_cloud_from_realsense(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    p8 = cfg.get("phase8_realsense", {})
    provider = RealSenseProvider(
        width=int(p8.get("width", 640)),
        height=int(p8.get("height", 480)),
        fps=int(p8.get("fps", 30)),
        warmup_frames=int(p8.get("warmup_frames", 20)),
    )
    provider.start()
    try:
        color, depth = provider.get_frames(timeout_ms=1500)
        intr = provider.intrinsics
        points, colors = build_pointcloud(
            depth,
            color,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
            min_depth=float(p8.get("min_depth", 0.15)),
            max_depth=float(p8.get("max_depth", 5.0)),
            stride=int(p8.get("cloud_stride", 2)),
        )
    finally:
        provider.stop()

    if points.shape[0] == 0:
        raise RuntimeError("Live capture returned zero valid points. Adjust phase8_realsense depth range.")
    return points, colors


def _save_temp_cloud_npz(points: np.ndarray, colors: np.ndarray) -> str:
    out_dir = os.path.join(_ROOT, "phase7_arm_icp", "outputs", "cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "live_capture.npz")
    np.savez(out_path, points=points.astype(np.float32), colors=colors.astype(np.float32))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase7 arm mesh ICP calibration")
    p.add_argument("--config", default="config/settings.yaml", help="settings.yaml path")
    p.add_argument(
        "--mesh-dir",
        default=None,
        help="Directory that contains robot arm mesh files",
    )
    p.add_argument("--target-cloud", default=None, help="Target cloud path (.ply/.pcd/.npz). If omitted, capture live from RealSense.")

    # 6DoF manual initialization (equivalent to six sliders/rods)
    p.add_argument("--init-txyz", nargs=3, type=float, default=None, help="Initial translation x y z (m)")
    p.add_argument(
        "--init-rxyz-deg",
        nargs=3,
        type=float,
        default=None,
        help="Initial XYZ Euler rotation in degrees",
    )

    p.add_argument("--voxel-size", type=float, default=None, help="Downsample voxel size in meters")
    p.add_argument("--points-per-mesh", type=int, default=None, help="Poisson samples per mesh")
    p.add_argument("--max-corr", type=float, default=None, help="ICP max correspondence distance")
    p.add_argument("--max-iters", type=int, default=None, help="ICP max iterations")
    p.add_argument("--point-to-point", action="store_true", help="Use point-to-point ICP instead of point-to-plane")

    p.add_argument("--crop-min", nargs=3, type=float, default=None, help="Optional target crop min x y z")
    p.add_argument("--crop-max", nargs=3, type=float, default=None, help="Optional target crop max x y z")

    p.add_argument(
        "--result-json",
        default=None,
        help="Output JSON path",
    )
    p.add_argument("--settings-path", default="config/settings.yaml", help="settings.yaml path for writing T_cam2arm")
    p.add_argument("--update-settings", action="store_true", help="Write solved T_cam2arm into settings")
    p.add_argument("--preview", action="store_true", help="Open Open3D preview after solving")
    p.add_argument("--interactive-preview", action="store_true", help="Open interactive manual alignment GUI before ICP")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    p7_cfg = cfg.get("phase7_arm_icp", {})

    mesh_dir = args.mesh_dir or p7_cfg.get("mesh_dir", "assets/ra605_710")
    if not os.path.isabs(mesh_dir):
        mesh_dir = os.path.join(_ROOT, mesh_dir)

    voxel_size = float(args.voxel_size if args.voxel_size is not None else p7_cfg.get("voxel_size", 0.01))
    points_per_mesh = int(args.points_per_mesh if args.points_per_mesh is not None else p7_cfg.get("points_per_mesh", 8000))
    max_corr = float(args.max_corr if args.max_corr is not None else p7_cfg.get("max_correspondence_distance", 0.04))
    max_iters = int(args.max_iters if args.max_iters is not None else p7_cfg.get("max_iterations", 80))
    result_json_cfg = p7_cfg.get("result_json", "phase7_arm_icp/outputs/arm_icp_result.json")
    result_json_arg = args.result_json if args.result_json is not None else result_json_cfg

    init_txyz = _parse_xyz(args.init_txyz) or _parse_xyz(p7_cfg.get("init_txyz")) or (0.5, 0.0, 0.8)
    init_rxyz = _parse_xyz(args.init_rxyz_deg) or _parse_xyz(p7_cfg.get("init_rxyz_deg")) or (-90.0, 0.0, 90.0)
    crop_min = _parse_xyz(args.crop_min) or _parse_xyz(p7_cfg.get("crop_min"))
    crop_max = _parse_xyz(args.crop_max) or _parse_xyz(p7_cfg.get("crop_max"))
    auto_update_settings = bool(p7_cfg.get("auto_update_settings", True))
    auto_preview = bool(p7_cfg.get("preview", False))

    target_cloud_path = args.target_cloud
    if target_cloud_path is None:
        print("[Phase7 Arm ICP] capture live cloud from RealSense...")
        points, colors = _capture_target_cloud_from_realsense(cfg)
        target_cloud_path = _save_temp_cloud_npz(points, colors)
        print(f"- live points: {points.shape[0]}")
        print(f"- temp cloud: {target_cloud_path}")
    elif not os.path.isabs(target_cloud_path):
        target_cloud_path = os.path.join(_ROOT, target_cloud_path)

    # If interactive preview requested, let user manually align a virtual frame to the live/loaded cloud
    if args.interactive_preview or bool(p7_cfg.get("interactive_preview", False)):
        print("[Phase7 Arm ICP] launching interactive manual alignment GUI...")
        target = load_target_cloud(target_cloud_path, voxel_size=voxel_size)
        init_r = init_rxyz
        init_t = init_txyz
        init_r_mat = None
        # compose initial transform
        init_r_mat = None
        # build initial 4x4
        from phase7_arm_icp.src.icp_calibrator import euler_xyz_deg_to_matrix

        init_mat = euler_xyz_deg_to_matrix(*init_rxyz)
        init_mat = np.eye(4, dtype=np.float64)
        init_mat[:3, :3] = euler_xyz_deg_to_matrix(*init_rxyz)
        init_mat[:3, 3] = np.asarray(init_txyz, dtype=np.float64)

        new_mat = interactive_manual_alignment(target, init_mat)
        # extract t and r
        new_t = tuple(new_mat[:3, 3].tolist())
        # convert rotation to euler deg using small helper in icp_calibrator if available
        try:
            from phase7_arm_icp.src.icp_calibrator import _matrix_to_euler_xyz_deg as _m2e

            new_r = _m2e(new_mat)
        except Exception:
            # fallback: approximate via scipy-like approach
            def _mat_to_euler_xyz_deg_local(mat):
                r = mat[:3, :3]
                sy = -r[0, 2]
                ay = np.arcsin(np.clip(sy, -1.0, 1.0))
                cy = np.cos(ay)
                if abs(cy) > 1e-6:
                    ax = np.arctan2(r[1, 2], r[2, 2])
                    az = np.arctan2(r[0, 1], r[0, 0])
                else:
                    ax = np.arctan2(-r[2, 1], r[1, 1])
                    az = 0.0
                return tuple(np.rad2deg([ax, ay, az]))

            new_r = _mat_to_euler_xyz_deg_local(new_mat)

        init_txyz = new_t
        init_rxyz = new_r

    result = calibrate_cam_to_arm(
        mesh_dir=mesh_dir,
        target_cloud_path=target_cloud_path,
        init_translation_xyz=init_txyz,
        init_rotation_xyz_deg=init_rxyz,
        voxel_size=voxel_size,
        points_per_mesh=points_per_mesh,
        crop_min_xyz=crop_min,
        crop_max_xyz=crop_max,
        max_correspondence_distance=max_corr,
        max_iterations=max_iters,
        point_to_plane=not args.point_to_point,
    )

    print("[Phase7 Arm ICP] done")
    print(f"- fitness: {result.icp.fitness:.6f}")
    print(f"- inlier_rmse: {result.icp.inlier_rmse:.6f}")
    print("- T_cam2arm:")
    for row in result.t_cam_to_arm:
        print("  ", [float(v) for v in row])

    result_json = result_json_arg
    if not os.path.isabs(result_json):
        result_json = os.path.join(_ROOT, result_json)
    save_result_json(result_json, result)
    print(f"- saved result: {result_json}")

    if args.update_settings or auto_update_settings:
        settings_path = args.settings_path
        if not os.path.isabs(settings_path):
            settings_path = os.path.join(_ROOT, settings_path)
        _update_settings_t_cam_to_arm(settings_path, result.t_cam_to_arm.tolist())
        print(f"- updated settings: {settings_path}")

    if args.preview or auto_preview:
        model = load_model_from_mesh_dir(
            mesh_dir,
            points_per_mesh=points_per_mesh,
            voxel_size=voxel_size,
        )
        target = load_target_cloud(target_cloud_path, voxel_size=voxel_size)
        preview_alignment(model, target, result.t_arm_to_cam)


if __name__ == "__main__":
    main()
