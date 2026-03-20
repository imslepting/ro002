"""Bundle Adjustment — 多相機外參全域優化

使用 scipy.optimize.least_squares 聯合優化所有相機的 T_w2c 位姿，
最小化所有觀測的重投影誤差。每塊標定板以 6DoF 位姿參數化（保持剛體約束）。

適用於 ≥3 台相機場景；2 台相機時 BA 退化為 stereoCalibrate，不建議使用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from phase1_intrinsics.src.board_generator import create_board
from phase1_intrinsics.src.charuco_detector import detect_charuco
from phase1_intrinsics.src.intrinsics_io import load_calib_result, load_intrinsics
from phase2_extrinsics.src.extrinsics_io import load_extrinsics, Rt_to_T44


# ---------------------------------------------------------------------------
# 數據結構
# ---------------------------------------------------------------------------

@dataclass
class BAResult:
    """Bundle Adjustment 結果"""
    T_w2c: dict[str, np.ndarray]   # 優化後的 4x4 位姿
    rms_before: float               # 優化前 RMS (px)
    rms_after: float                # 優化後 RMS (px)
    num_observations: int
    num_cameras: int
    passed: bool                    # rms_after < max_ba_residual


# ---------------------------------------------------------------------------
# 位姿拓撲建立
# ---------------------------------------------------------------------------

def build_initial_poses(
    extrinsics_path: str,
    reference_cam: str = "cam0",
) -> dict[str, np.ndarray]:
    """從 extrinsics.json 建立星形拓撲 T_w2c

    reference_cam 設為世界原點 (T44 = I)，其他相機透過配對鏈推算。
    extrinsics.json 中的 R, T 代表 cam_right 到 cam_left 的變換。

    Returns:
        {"cam0": eye(4), "cam1": T44_01, ...}
    """
    data = load_extrinsics(extrinsics_path)
    if not data:
        raise ValueError(f"extrinsics.json 為空或不存在: {extrinsics_path}")

    # 收集所有相機名稱
    all_cams = set()
    for pair_info in data.values():
        all_cams.add(pair_info["cam_left"])
        all_cams.add(pair_info["cam_right"])

    if reference_cam not in all_cams:
        raise ValueError(
            f"reference_cam '{reference_cam}' 不在已標定相機中: {all_cams}"
        )

    # 建立配對圖 (鄰接表)：cam_a -> [(cam_b, T_b2a)]
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {c: [] for c in all_cams}
    for pair_info in data.values():
        cam_l = pair_info["cam_left"]
        cam_r = pair_info["cam_right"]
        R = np.array(pair_info["R"])
        T = np.array(pair_info["T"])
        T_r2l = Rt_to_T44(R, T)  # cam_right → cam_left

        adjacency[cam_l].append((cam_r, T_r2l))
        # 反向：cam_left → cam_right = inv(T_r2l)
        adjacency[cam_r].append((cam_l, np.linalg.inv(T_r2l)))

    # BFS 從 reference_cam 出發，計算 T_w2c（world = reference_cam 坐標系）
    T_w2c = {reference_cam: np.eye(4)}
    queue = [reference_cam]
    visited = {reference_cam}

    while queue:
        current = queue.pop(0)
        for neighbor, T_neighbor2current in adjacency[current]:
            if neighbor in visited:
                continue
            # T_w2c[neighbor] = T_neighbor2current^{-1} @ ... 不對
            # world = reference_cam 座標系
            # T_w2c[current] 把世界座標轉到 current 相機座標
            # T_neighbor2current 把 neighbor 座標轉到 current 座標
            # T_w2c[neighbor] = inv(T_neighbor2current) @ T_w2c[current]?
            # 不對。T_neighbor2current 是 neighbor→current，所以：
            # T_w2c[neighbor] = T_current2neighbor @ T_w2c[current]
            #                 = inv(T_neighbor2current) @ T_w2c[current]
            # 等等，重新思考。
            # T_w2c[cam] 把 world 點轉到 cam 座標：p_cam = T_w2c[cam] @ p_world
            # T_neighbor2current 把 neighbor 座標的點轉到 current 座標：
            #   p_current = T_neighbor2current @ p_neighbor
            # 所以 p_neighbor = inv(T_neighbor2current) @ p_current
            #                 = inv(T_neighbor2current) @ T_w2c[current] @ p_world
            # → T_w2c[neighbor] = inv(T_neighbor2current) @ T_w2c[current]
            T_w2c[neighbor] = np.linalg.inv(T_neighbor2current) @ T_w2c[current]
            visited.add(neighbor)
            queue.append(neighbor)

    unreachable = all_cams - visited
    if unreachable:
        raise ValueError(
            f"以下相機無法從 {reference_cam} 到達: {unreachable}。"
            "請確認 extrinsics.json 的配對鏈完整。"
        )

    return T_w2c


# ---------------------------------------------------------------------------
# 觀測收集
# ---------------------------------------------------------------------------

def collect_observations(
    pair_base_dir: str,
    extrinsics_path: str,
    intrinsics_path: str,
    board,
    dictionary,
    T_w2c: dict[str, np.ndarray],
    min_corners: int = 6,
) -> tuple[list[tuple[str, int, np.ndarray]], list[np.ndarray]]:
    """遍歷所有已標定配對的 raw_pairs 圖片，收集觀測

    對每張圖片偵測 ChArUco 角點，用 solvePnP 求板位姿，轉到世界座標。

    Returns:
        observations: list of (cam_key, board_idx, pts_2d)
            pts_2d shape (N, 2)
        board_poses_world: list of 4x4 ndarray (板在世界座標的位姿)
    """
    data = load_extrinsics(extrinsics_path)
    intrinsics_data = load_intrinsics(intrinsics_path)

    observations: list[tuple[str, int, np.ndarray]] = []
    board_poses_world: list[np.ndarray] = []
    board_idx = 0

    for pair_name, pair_info in data.items():
        cam_left = pair_info["cam_left"]
        cam_right = pair_info["cam_right"]

        pair_dir = os.path.join(pair_base_dir, pair_name)
        left_dir = os.path.join(pair_dir, "left")
        right_dir = os.path.join(pair_dir, "right")

        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            print(f"[BA] 跳過配對 {pair_name}: 目錄不存在")
            continue

        # 載入內參
        calib_l = load_calib_result(cam_left, intrinsics_path)
        calib_r = load_calib_result(cam_right, intrinsics_path)
        if calib_l is None or calib_r is None:
            print(f"[BA] 跳過配對 {pair_name}: 內參未找到")
            continue

        left_files = sorted(f for f in os.listdir(left_dir) if f.endswith(".png"))
        right_files = sorted(f for f in os.listdir(right_dir) if f.endswith(".png"))
        n_pairs = min(len(left_files), len(right_files))

        for i in range(n_pairs):
            left_path = os.path.join(left_dir, left_files[i])
            right_path = os.path.join(right_dir, right_files[i])

            left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            if left_gray is None or right_gray is None:
                continue

            # 偵測左圖角點
            det_l = detect_charuco(
                left_gray, board, dictionary,
                refine_subpix=True, min_corners=min_corners,
            )
            if not det_l.success:
                continue

            obj_pts_l, img_pts_l = board.matchImagePoints(
                det_l.charuco_corners, det_l.charuco_ids,
            )
            if obj_pts_l is None or len(obj_pts_l) < min_corners:
                continue

            # solvePnP 求板在 cam_left 座標下的位姿
            ok, rvec_l, tvec_l = cv2.solvePnP(
                obj_pts_l, img_pts_l,
                calib_l.K, calib_l.D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue

            R_board_cam_l, _ = cv2.Rodrigues(rvec_l)
            T_board_cam_l = Rt_to_T44(R_board_cam_l, tvec_l)

            # 轉到世界座標：T_board_world = inv(T_w2c[cam_left]) @ T_board_cam_l
            T_board_world = np.linalg.inv(T_w2c[cam_left]) @ T_board_cam_l
            board_poses_world.append(T_board_world)

            # 左圖觀測
            observations.append((
                cam_left,
                board_idx,
                img_pts_l.reshape(-1, 2).astype(np.float64),
            ))

            # 偵測右圖角點
            det_r = detect_charuco(
                right_gray, board, dictionary,
                refine_subpix=True, min_corners=min_corners,
            )
            if det_r.success:
                obj_pts_r, img_pts_r = board.matchImagePoints(
                    det_r.charuco_corners, det_r.charuco_ids,
                )
                if obj_pts_r is not None and len(obj_pts_r) >= min_corners:
                    observations.append((
                        cam_right,
                        board_idx,
                        img_pts_r.reshape(-1, 2).astype(np.float64),
                    ))

            board_idx += 1

    print(f"[BA] 收集到 {len(observations)} 條觀測, {len(board_poses_world)} 塊板位姿")
    return observations, board_poses_world


# ---------------------------------------------------------------------------
# 參數打包 / 解包
# ---------------------------------------------------------------------------

def pack_params(
    cam_keys: list[str],
    T_w2c: dict[str, np.ndarray],
    board_poses: list[np.ndarray],
) -> np.ndarray:
    """將相機位姿和板位姿打包為一維參數向量

    cam_keys: 非 reference 的相機列表（reference 固定不優化）
    每台相機: rotvec(3) + tvec(3) = 6
    每塊板: rotvec(3) + tvec(3) = 6
    """
    params = []

    # 相機位姿
    for cam in cam_keys:
        T = T_w2c[cam]
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3]
        params.extend(rvec.flatten())
        params.extend(tvec.flatten())

    # 板位姿
    for T_board in board_poses:
        rvec, _ = cv2.Rodrigues(T_board[:3, :3])
        tvec = T_board[:3, 3]
        params.extend(rvec.flatten())
        params.extend(tvec.flatten())

    return np.array(params, dtype=np.float64)


def unpack_params(
    params: np.ndarray,
    cam_keys: list[str],
    reference_cam: str,
    n_boards: int,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """從一維參數向量解包相機和板位姿"""
    n_cams = len(cam_keys)
    idx = 0

    T_w2c = {reference_cam: np.eye(4)}

    for cam in cam_keys:
        rvec = params[idx:idx + 3]
        tvec = params[idx + 3:idx + 6]
        idx += 6
        R, _ = cv2.Rodrigues(rvec)
        T_w2c[cam] = Rt_to_T44(R, tvec)

    board_poses = []
    for _ in range(n_boards):
        rvec = params[idx:idx + 3]
        tvec = params[idx + 3:idx + 6]
        idx += 6
        R, _ = cv2.Rodrigues(rvec)
        board_poses.append(Rt_to_T44(R, tvec))

    return T_w2c, board_poses


# ---------------------------------------------------------------------------
# 重投影殘差
# ---------------------------------------------------------------------------

def _compute_residuals(
    params: np.ndarray,
    cam_keys: list[str],
    reference_cam: str,
    n_boards: int,
    board,
    observations: list[tuple[str, int, np.ndarray]],
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """計算所有觀測的重投影殘差向量"""
    T_w2c, board_poses = unpack_params(params, cam_keys, reference_cam, n_boards)

    # 取得板的 3D 角點（板坐標系下）
    # board.getChessboardCorners() 返回所有角點的 3D 坐標
    obj_pts_board = board.getChessboardCorners()  # (M, 3)

    residuals = []

    for cam_key, board_idx, pts_2d_observed in observations:
        if board_idx >= len(board_poses):
            continue

        T_cam = T_w2c[cam_key]          # 世界→相機
        T_board = board_poses[board_idx]  # 板→世界

        # 板 3D 點轉到相機座標
        T_cam_board = T_cam @ T_board  # 板→相機
        R_cb = T_cam_board[:3, :3]
        t_cb = T_cam_board[:3, 3]

        rvec_cb, _ = cv2.Rodrigues(R_cb)
        K, D = intrinsics[cam_key]

        # 觀測可能只包含部分角點，需要取對應的 obj_pts
        n_observed = len(pts_2d_observed)
        n_board_pts = len(obj_pts_board)

        if n_observed == n_board_pts:
            obj_pts_used = obj_pts_board
        else:
            # 觀測數量與板角點數量不同，取前 n_observed 個
            # 注意：這裡假設觀測的角點是從 matchImagePoints 得到的，
            # 已經按 ID 排序並與 obj_pts_board 的子集對應。
            # 由於 matchImagePoints 返回的點是按 board 定義排序的，
            # 我們需要使用全部板角點但只取匹配的子集。
            # 實際上 collect_observations 中 matchImagePoints 返回的
            # obj_pts 是子集，所以這裡我們需要存儲對應的 obj_pts。
            # 為了正確處理，我們在 observations 中存儲 obj_pts。
            obj_pts_used = obj_pts_board[:n_observed]

        projected, _ = cv2.projectPoints(
            obj_pts_used.reshape(-1, 1, 3).astype(np.float64),
            rvec_cb, t_cb,
            K, D,
        )
        projected = projected.reshape(-1, 2)

        diff = projected - pts_2d_observed
        residuals.append(diff.flatten())

    if not residuals:
        return np.array([0.0])
    return np.concatenate(residuals)


# ---------------------------------------------------------------------------
# 改進的觀測收集（附帶 obj_pts）
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """單條觀測"""
    cam_key: str
    board_idx: int
    pts_2d: np.ndarray    # (N, 2) 觀測到的像素座標
    obj_pts: np.ndarray   # (N, 3) 板座標系下的 3D 點


def collect_observations_with_objpts(
    pair_base_dir: str,
    extrinsics_path: str,
    intrinsics_path: str,
    board,
    dictionary,
    T_w2c: dict[str, np.ndarray],
    min_corners: int = 6,
) -> tuple[list[Observation], list[np.ndarray]]:
    """收集觀測（含每條觀測的 obj_pts），用於正確的殘差計算

    Returns:
        observations: list of Observation
        board_poses_world: list of 4x4 ndarray
    """
    data = load_extrinsics(extrinsics_path)

    observations: list[Observation] = []
    board_poses_world: list[np.ndarray] = []
    board_idx = 0

    for pair_name, pair_info in data.items():
        cam_left = pair_info["cam_left"]
        cam_right = pair_info["cam_right"]

        pair_dir = os.path.join(pair_base_dir, pair_name)
        left_dir = os.path.join(pair_dir, "left")
        right_dir = os.path.join(pair_dir, "right")

        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            print(f"[BA] 跳過配對 {pair_name}: 目錄不存在")
            continue

        calib_l = load_calib_result(cam_left, intrinsics_path)
        calib_r = load_calib_result(cam_right, intrinsics_path)
        if calib_l is None or calib_r is None:
            print(f"[BA] 跳過配對 {pair_name}: 內參未找到")
            continue

        left_files = sorted(f for f in os.listdir(left_dir) if f.endswith(".png"))
        right_files = sorted(f for f in os.listdir(right_dir) if f.endswith(".png"))
        n_pairs = min(len(left_files), len(right_files))

        for i in range(n_pairs):
            left_path = os.path.join(left_dir, left_files[i])
            right_path = os.path.join(right_dir, right_files[i])

            left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            if left_gray is None or right_gray is None:
                continue

            # 偵測左圖
            det_l = detect_charuco(
                left_gray, board, dictionary,
                refine_subpix=True, min_corners=min_corners,
            )
            if not det_l.success:
                continue

            obj_pts_l, img_pts_l = board.matchImagePoints(
                det_l.charuco_corners, det_l.charuco_ids,
            )
            if obj_pts_l is None or len(obj_pts_l) < min_corners:
                continue

            # solvePnP 求板在 cam_left 座標下的位姿
            ok, rvec_l, tvec_l = cv2.solvePnP(
                obj_pts_l, img_pts_l,
                calib_l.K, calib_l.D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue

            R_board_cam_l, _ = cv2.Rodrigues(rvec_l)
            T_board_cam_l = Rt_to_T44(R_board_cam_l, tvec_l)
            T_board_world = np.linalg.inv(T_w2c[cam_left]) @ T_board_cam_l
            board_poses_world.append(T_board_world)

            # 左圖觀測
            observations.append(Observation(
                cam_key=cam_left,
                board_idx=board_idx,
                pts_2d=img_pts_l.reshape(-1, 2).astype(np.float64),
                obj_pts=obj_pts_l.reshape(-1, 3).astype(np.float64),
            ))

            # 偵測右圖
            det_r = detect_charuco(
                right_gray, board, dictionary,
                refine_subpix=True, min_corners=min_corners,
            )
            if det_r.success:
                obj_pts_r, img_pts_r = board.matchImagePoints(
                    det_r.charuco_corners, det_r.charuco_ids,
                )
                if obj_pts_r is not None and len(obj_pts_r) >= min_corners:
                    observations.append(Observation(
                        cam_key=cam_right,
                        board_idx=board_idx,
                        pts_2d=img_pts_r.reshape(-1, 2).astype(np.float64),
                        obj_pts=obj_pts_r.reshape(-1, 3).astype(np.float64),
                    ))

            board_idx += 1

    print(f"[BA] 收集到 {len(observations)} 條觀測, {len(board_poses_world)} 塊板位姿")
    return observations, board_poses_world


# ---------------------------------------------------------------------------
# 殘差函數（使用 Observation）
# ---------------------------------------------------------------------------

def _residuals_fn(
    params: np.ndarray,
    cam_keys: list[str],
    reference_cam: str,
    n_boards: int,
    observations: list[Observation],
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """計算所有觀測的重投影殘差"""
    T_w2c, board_poses = unpack_params(params, cam_keys, reference_cam, n_boards)

    residuals = []
    for obs in observations:
        if obs.board_idx >= len(board_poses):
            continue

        T_cam = T_w2c[obs.cam_key]
        T_board = board_poses[obs.board_idx]
        T_cam_board = T_cam @ T_board

        rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
        tvec = T_cam_board[:3, 3]
        K, D = intrinsics[obs.cam_key]

        projected, _ = cv2.projectPoints(
            obs.obj_pts.reshape(-1, 1, 3),
            rvec, tvec, K, D,
        )
        projected = projected.reshape(-1, 2)

        diff = projected - obs.pts_2d
        residuals.append(diff.flatten())

    if not residuals:
        return np.array([0.0])
    return np.concatenate(residuals)


def _compute_rms(residuals: np.ndarray) -> float:
    """從殘差向量計算 RMS (px)"""
    if len(residuals) <= 1:
        return 0.0
    # residuals 是 [dx1, dy1, dx2, dy2, ...] 交錯排列
    n_pts = len(residuals) // 2
    if n_pts == 0:
        return 0.0
    return float(np.sqrt(np.sum(residuals ** 2) / n_pts))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_bundle_adjustment(
    intrinsics_path: str,
    extrinsics_path: str,
    pair_base_dir: str,
    board,
    dictionary,
    reference_cam: str = "cam0",
    max_ba_residual: float = 0.5,
) -> BAResult:
    """執行 Bundle Adjustment 全域優化

    Args:
        intrinsics_path: intrinsics.json 路徑
        extrinsics_path: extrinsics.json 路徑
        pair_base_dir: raw_pairs 目錄（含 cam0_cam1/left, right 等子目錄）
        board: CharucoBoard 對象
        dictionary: ArUco 字典
        reference_cam: 世界原點相機
        max_ba_residual: 最大允許殘差 (px)

    Returns:
        BAResult
    """
    # 1. 建立初始位姿
    print("[BA] 建立初始位姿拓撲...")
    T_w2c = build_initial_poses(extrinsics_path, reference_cam)
    print(f"[BA] 共 {len(T_w2c)} 台相機: {sorted(T_w2c.keys())}")

    # 2. 收集觀測
    print("[BA] 收集觀測...")
    observations, board_poses = collect_observations_with_objpts(
        pair_base_dir, extrinsics_path, intrinsics_path,
        board, dictionary, T_w2c,
    )

    if not observations or not board_poses:
        raise ValueError("[BA] 無有效觀測，無法執行 Bundle Adjustment")

    # 載入內參（固定不優化）
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam in T_w2c:
        calib = load_calib_result(cam, intrinsics_path)
        if calib is None:
            raise ValueError(f"[BA] 相機 {cam} 的內參未找到")
        intrinsics[cam] = (calib.K.astype(np.float64), calib.D.astype(np.float64))

    # 3. 打包參數
    cam_keys = sorted(c for c in T_w2c if c != reference_cam)
    x0 = pack_params(cam_keys, T_w2c, board_poses)

    print(f"[BA] 參數維度: {len(x0)} "
          f"({len(cam_keys)} 台相機 × 6 + {len(board_poses)} 塊板 × 6)")

    # 計算優化前殘差
    r0 = _residuals_fn(x0, cam_keys, reference_cam, len(board_poses),
                        observations, intrinsics)
    rms_before = _compute_rms(r0)
    print(f"[BA] 優化前 RMS: {rms_before:.4f} px")

    # 4. 運行優化
    print("[BA] 運行 least_squares 優化...")
    result = least_squares(
        _residuals_fn,
        x0,
        args=(cam_keys, reference_cam, len(board_poses),
              observations, intrinsics),
        method="lm",
        verbose=0,
    )

    # 5. 解包結果
    T_w2c_opt, _ = unpack_params(
        result.x, cam_keys, reference_cam, len(board_poses),
    )
    rms_after = _compute_rms(result.fun)
    passed = rms_after < max_ba_residual

    print(f"[BA] 優化後 RMS: {rms_after:.4f} px")
    print(f"[BA] {'PASS' if passed else 'FAIL'} "
          f"(閾值: {max_ba_residual} px)")

    return BAResult(
        T_w2c=T_w2c_opt,
        rms_before=rms_before,
        rms_after=rms_after,
        num_observations=len(observations),
        num_cameras=len(T_w2c),
        passed=passed,
    )
