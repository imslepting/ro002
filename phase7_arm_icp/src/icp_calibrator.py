"""Mesh-to-pointcloud ICP utilities for arm extrinsic calibration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class IcpResult:
    fitness: float
    inlier_rmse: float
    transform_model_to_cam: np.ndarray


@dataclass
class CalibrationResult:
    t_cam_to_arm: np.ndarray
    t_arm_to_cam: np.ndarray
    icp: IcpResult


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency checked at runtime
        raise RuntimeError("open3d is required for phase7_arm_icp") from exc
    return o3d


def euler_xyz_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return rotation matrix for intrinsic XYZ Euler angles in degrees."""
    ax, ay, az = np.deg2rad([rx, ry, rz])

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return rx_m @ ry_m @ rz_m


def compose_transform(r: np.ndarray, txyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(txyz, dtype=np.float64)
    return t


def _collect_mesh_files(mesh_dir: str) -> list[str]:
    mesh_files: list[str] = []
    exts = {".stl", ".obj", ".ply", ".dae"}
    for root, _, files in os.walk(mesh_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                mesh_files.append(os.path.join(root, name))
    mesh_files.sort()
    return mesh_files


def load_model_from_mesh_dir(
    mesh_dir: str,
    *,
    points_per_mesh: int = 10000,
    voxel_size: float = 0.01,
):
    o3d = _require_open3d()

    mesh_files = _collect_mesh_files(mesh_dir)
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found in: {mesh_dir}")

    merged = o3d.geometry.PointCloud()
    for mf in mesh_files:
        mesh = o3d.io.read_triangle_mesh(mf)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            continue
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=points_per_mesh)
        merged += pcd

    if merged.is_empty():
        raise RuntimeError("Mesh files loaded but sampled model point cloud is empty")

    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)

    return merged


def load_target_cloud(path: str, voxel_size: float = 0.01):
    o3d = _require_open3d()

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        arr = np.load(path)
        if "points" not in arr:
            raise ValueError(".npz target cloud must include 'points' array")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(arr["points"], dtype=np.float64))
        if "colors" in arr:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(arr["colors"], dtype=np.float64))
    else:
        pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise RuntimeError(f"Target cloud is empty: {path}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def crop_cloud_axis_aligned(pcd, min_xyz: np.ndarray | None, max_xyz: np.ndarray | None):
    o3d = _require_open3d()
    if min_xyz is None or max_xyz is None:
        return pcd
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_xyz, max_bound=max_xyz)
    return pcd.crop(bbox)


def run_icp(
    model_cloud,
    target_cloud,
    init_t_model_to_cam: np.ndarray,
    *,
    max_correspondence_distance: float = 0.04,
    max_iterations: int = 80,
    point_to_plane: bool = True,
) -> IcpResult:
    o3d = _require_open3d()

    src = model_cloud
    tgt = target_cloud

    if point_to_plane:
        radius = max(0.02, max_correspondence_distance * 2.0)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance,
        init_t_model_to_cam,
        estimation,
        criteria,
    )

    return IcpResult(
        fitness=float(reg.fitness),
        inlier_rmse=float(reg.inlier_rmse),
        transform_model_to_cam=np.asarray(reg.transformation, dtype=np.float64),
    )


def calibrate_cam_to_arm(
    mesh_dir: str,
    target_cloud_path: str,
    *,
    init_translation_xyz: tuple[float, float, float],
    init_rotation_xyz_deg: tuple[float, float, float],
    voxel_size: float = 0.01,
    points_per_mesh: int = 8000,
    crop_min_xyz: tuple[float, float, float] | None = None,
    crop_max_xyz: tuple[float, float, float] | None = None,
    max_correspondence_distance: float = 0.04,
    max_iterations: int = 80,
    point_to_plane: bool = True,
) -> CalibrationResult:
    model = load_model_from_mesh_dir(
        mesh_dir,
        points_per_mesh=points_per_mesh,
        voxel_size=voxel_size,
    )
    target = load_target_cloud(target_cloud_path, voxel_size=voxel_size)

    target = crop_cloud_axis_aligned(
        target,
        np.array(crop_min_xyz, dtype=np.float64) if crop_min_xyz is not None else None,
        np.array(crop_max_xyz, dtype=np.float64) if crop_max_xyz is not None else None,
    )
    if target.is_empty():
        raise RuntimeError("Target cloud became empty after crop")

    init_r = euler_xyz_deg_to_matrix(*init_rotation_xyz_deg)
    init_t = compose_transform(init_r, np.array(init_translation_xyz, dtype=np.float64))

    icp_res = run_icp(
        model,
        target,
        init_t,
        max_correspondence_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        point_to_plane=point_to_plane,
    )

    t_cam_to_arm = np.linalg.inv(icp_res.transform_model_to_cam)
    return CalibrationResult(
        t_cam_to_arm=t_cam_to_arm,
        t_arm_to_cam=icp_res.transform_model_to_cam,
        icp=icp_res,
    )


def save_result_json(path: str, result: CalibrationResult) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "T_model_arm_to_cam": result.t_arm_to_cam.tolist(),
        "T_cam_to_arm": result.t_cam_to_arm.tolist(),
        "icp": {
            "fitness": result.icp.fitness,
            "inlier_rmse": result.icp.inlier_rmse,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def preview_alignment(model_cloud, target_cloud, t_model_to_cam: np.ndarray) -> None:
    o3d = _require_open3d()

    src = o3d.geometry.PointCloud(model_cloud)
    src.transform(t_model_to_cam.copy())
    src.paint_uniform_color([1.0, 0.3, 0.2])
    tgt = o3d.geometry.PointCloud(target_cloud)
    tgt.paint_uniform_color([0.2, 0.8, 1.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries(
        [tgt, src, axis],
        window_name="Phase7 Arm ICP Preview (target=cyan, model=red)",
    )


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


def interactive_manual_alignment(target_cloud, init_t_model_to_cam: np.ndarray) -> np.ndarray:
    """Open an interactive Open3D window with sliders to manually align a coordinate
    frame (model) to the provided target point cloud. Returns adjusted 4x4 transform.
    """
    try:
        import open3d as o3d  # type: ignore[import-not-found]
        from open3d import visualization as vis  # type: ignore[import-not-found]
        from open3d.visualization import gui  # type: ignore[import-not-found]
        from open3d.visualization import rendering  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - runtime GUI dependency
        # Fallback: non-interactive preview
        preview_alignment(None, target_cloud, init_t_model_to_cam)
        return init_t_model_to_cam

    # Prepare geometries
    tgt = o3d.geometry.PointCloud(target_cloud)
    tgt.paint_uniform_color([0.2, 0.8, 1.0])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    frame.compute_vertex_normals()

    # Extract initial params
    init_t = init_t_model_to_cam[:3, 3].astype(np.float64)
    init_rx, init_ry, init_rz = _matrix_to_euler_xyz_deg(init_t_model_to_cam)

    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Phase7 Manual Alignment", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([0.0, 0.0, 0.0, 1.0])
    w.add_child(scene)

    # Add target cloud and frame to the scene
    scene.scene.add_geometry("target", tgt, rendering.MaterialRecord())
    scene.scene.add_geometry("frame", frame, rendering.MaterialRecord())

    def set_frame_transform(tx, ty, tz, rx_deg, ry_deg, rz_deg):
        r = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
        t = compose_transform(r, np.array([tx, ty, tz], dtype=np.float64))
        scene.scene.set_geometry_transform("frame", t)

    # Initial placement
    set_frame_transform(init_t[0], init_t[1], init_t[2], init_rx, init_ry, init_rz)

    # UI panel on right
    em = w.theme.font_size
    panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
    w.add_child(panel)

    # sliders
    sx = gui.Slider(gui.Slider.INT)
    try:
        sx.set_limits(-1000, 1000)
    except Exception:
        pass
    # set initial value compatible with different Open3D versions
    try:
        sx.set_value(int(init_t[0] * 1000))
    except Exception:
        sx.int_value = int(init_t[0] * 1000)

    sy = gui.Slider(gui.Slider.INT)
    try:
        sy.set_limits(-1000, 1000)
    except Exception:
        pass
    try:
        sy.set_value(int(init_t[1] * 1000))
    except Exception:
        sy.int_value = int(init_t[1] * 1000)

    sz = gui.Slider(gui.Slider.INT)
    try:
        sz.set_limits(-1000, 1000)
    except Exception:
        pass
    try:
        sz.set_value(int(init_t[2] * 1000))
    except Exception:
        sz.int_value = int(init_t[2] * 1000)

    srx = gui.Slider(gui.Slider.INT)
    try:
        srx.set_limits(-180, 180)
    except Exception:
        pass
    try:
        srx.set_value(int(init_rx))
    except Exception:
        srx.int_value = int(init_rx)

    sry = gui.Slider(gui.Slider.INT)
    try:
        sry.set_limits(-180, 180)
    except Exception:
        pass
    try:
        sry.set_value(int(init_ry))
    except Exception:
        sry.int_value = int(init_ry)

    srz = gui.Slider(gui.Slider.INT)
    try:
        srz.set_limits(-180, 180)
    except Exception:
        pass
    try:
        srz.set_value(int(init_rz))
    except Exception:
        srz.int_value = int(init_rz)

    def _get_slider_value(sl):
        try:
            return sl.get_value()
        except Exception:
            # some Open3D versions expose int_value
            return getattr(sl, "int_value", 0)


    def _on_slider(_: int) -> None:
        tx = _get_slider_value(sx) / 1000.0
        ty = _get_slider_value(sy) / 1000.0
        tz = _get_slider_value(sz) / 1000.0
        rx = float(_get_slider_value(srx))
        ry = float(_get_slider_value(sry))
        rz = float(_get_slider_value(srz))
        set_frame_transform(tx, ty, tz, rx, ry, rz)

    sx.set_on_value_changed(_on_slider)
    sy.set_on_value_changed(_on_slider)
    sz.set_on_value_changed(_on_slider)
    srx.set_on_value_changed(_on_slider)
    sry.set_on_value_changed(_on_slider)
    srz.set_on_value_changed(_on_slider)

    panel.add_child(gui.Label("Translation (m) x,y,z"))
    panel.add_child(sx)
    panel.add_child(sy)
    panel.add_child(sz)
    panel.add_child(gui.Label("Rotation (deg) rx,ry,rz"))
    panel.add_child(srx)
    panel.add_child(sry)
    panel.add_child(srz)

    done = gui.Button("Done - Accept Transform")
    result_transform = {"mat": init_t_model_to_cam.copy()}

    def _on_done():
        tx = _get_slider_value(sx) / 1000.0
        ty = _get_slider_value(sy) / 1000.0
        tz = _get_slider_value(sz) / 1000.0
        rx = float(_get_slider_value(srx))
        ry = float(_get_slider_value(sry))
        rz = float(_get_slider_value(srz))
        r = euler_xyz_deg_to_matrix(rx, ry, rz)
        result_transform["mat"] = compose_transform(r, np.array([tx, ty, tz], dtype=np.float64))
        app.quit()

    done.set_on_clicked(lambda _: _on_done())
    panel.add_child(done)

    # Layout callback
    def on_layout(layout_context):
        r = w.content_rect
        scene.frame = r
        panel.frame = gui.Rect(r.get_right() - 300, r.y, 300, r.height)

    w.set_on_layout(on_layout)

    app.run()

    return result_transform["mat"]
