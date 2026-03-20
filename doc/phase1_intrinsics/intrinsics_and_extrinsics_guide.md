# 多相機標定 → 統一 Metric 點雲 技術執行指導

## 系統概覽

```
[硬體]
cam1, cam2  → 水平並排雙目（標定 stereo 外參）
cam3, cam4  → 任意角度，與 cam1 有公共視野即可

[軟體流程]
Phase 1: 所有相機單目內參標定（ChArUco + cornerSubPix）
Phase 2: 雙目外參標定（cam1 ↔ cam2）
Phase 3: 多相機外參標定（cam1 ↔ cam3, cam1 ↔ cam4，星形拓撲）
Phase 4: scipy Bundle Adjustment 全域優化
Phase 5: Fast-FoundationStereo → metric 深度圖（cam1/cam2）
Phase 6: DA3Nested 多視角推理 → 統一 metric 點雲
```

---

## 依賴安裝

```bash
pip install opencv-contrib-python numpy scipy open3d
pip install torch torchvision
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
# Fast-FoundationStereo 待官方 code release，先用 FoundationStereo 替代
pip install git+https://github.com/NVlabs/FoundationStereo.git
```

---

## Phase 1：單目內參標定

### 1.1 製作 ChArUco 標定板

```python
import cv2
import numpy as np

ARUCO_DICT   = cv2.aruco.DICT_4X4_50
BOARD_COLS   = 9       # 方格列數
BOARD_ROWS   = 6       # 方格行數
SQUARE_SIZE  = 0.04    # 單位公尺，每格邊長（打印時量實際大小後修改）
MARKER_SIZE  = 0.03    # ArUco 碼邊長（= SQUARE_SIZE * 0.75 為宜）

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS),
    SQUARE_SIZE,
    MARKER_SIZE,
    dictionary
)

# 輸出圖片（600 dpi 打印 A4）
board_img = board.generateImage((2400, 1600), marginSize=20)
cv2.imwrite("charuco_board.png", board_img)
```

> 打印後用尺量 SQUARE_SIZE 的實際尺寸，填回上方常數。

---

### 1.2 採集標定圖像

每台相機各自拍攝，**不需要同步**。

```python
import cv2

def collect_calibration_images(cam_index: int, output_dir: str, n_images: int = 60):
    """
    互動式採集：按 SPACE 拍照，按 Q 結束
    建議：板子放不同角度（傾斜 ±30°）、不同距離（0.3m ~ 1.5m）、不同位置（畫面四角都要涵蓋）
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    count = 0
    while count < n_images:
        ret, frame = cap.read()
        cv2.imshow(f"cam{cam_index} - {count}/{n_images} (SPACE=save, Q=quit)", frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.imwrite(f"{output_dir}/frame_{count:04d}.png", frame)
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 四台相機分別執行
collect_calibration_images(0, "calib_data/cam0")
collect_calibration_images(1, "calib_data/cam1")
collect_calibration_images(2, "calib_data/cam2")
collect_calibration_images(3, "calib_data/cam3")
```

---

### 1.3 單目標定函數

```python
import cv2
import numpy as np
import glob
import json

ARUCO_DICT  = cv2.aruco.DICT_4X4_50
BOARD_COLS  = 9
BOARD_ROWS  = 6
SQUARE_SIZE = 0.04
MARKER_SIZE = 0.03

def calibrate_single_camera(image_dir: str, cam_id: int) -> dict:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, dictionary
    )
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params  = cv2.aruco.CharucoParameters()

    all_charuco_corners = []
    all_charuco_ids     = []
    image_size          = None

    paths = sorted(glob.glob(f"{image_dir}/*.png"))
    for path in paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # 偵測 ArUco 角點
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
        if ids is None or len(ids) < 6:
            continue

        # 插值 ChArUco 角點
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board,
            charucoParameters=charuco_params
        )
        if not ret or charuco_corners is None or len(charuco_corners) < 6:
            continue

        # cornerSubPix 精化
        charuco_corners = cv2.cornerSubPix(
            gray, charuco_corners,
            winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        )
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    print(f"[cam{cam_id}] 有效幀數: {len(all_charuco_corners)}/{len(paths)}")
    assert len(all_charuco_corners) >= 20, "有效幀數不足，請補充採集"

    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image_size, None, None
    )

    result = {
        "cam_id":      cam_id,
        "K":           K.tolist(),
        "D":           D.tolist(),
        "image_size":  list(image_size),
        "rms_error":   ret
    }
    print(f"[cam{cam_id}] RMS 重投影誤差: {ret:.4f} px  (目標 < 1.0 px)")
    return result


# 執行四台相機標定
results = {}
for i in range(4):
    results[f"cam{i}"] = calibrate_single_camera(f"calib_data/cam{i}", i)

with open("intrinsics.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Phase 2：雙目外參標定（cam0 ↔ cam1）

### 2.1 同步採集圖像對

> cam0/cam1 必須**同時**拍攝同一個 ChArUco 板靜止狀態。

```python
def collect_stereo_pairs(cam_idx_l: int, cam_idx_r: int, output_dir: str, n_pairs: int = 50):
    import os
    os.makedirs(f"{output_dir}/left",  exist_ok=True)
    os.makedirs(f"{output_dir}/right", exist_ok=True)
    cap_l = cv2.VideoCapture(cam_idx_l)
    cap_r = cv2.VideoCapture(cam_idx_r)
    count = 0
    while count < n_pairs:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        display = np.hstack([
            cv2.resize(frame_l, (640, 480)),
            cv2.resize(frame_r, (640, 480))
        ])
        cv2.imshow(f"stereo ({count}/{n_pairs}) SPACE=save", display)
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.imwrite(f"{output_dir}/left/frame_{count:04d}.png",  frame_l)
            cv2.imwrite(f"{output_dir}/right/frame_{count:04d}.png", frame_r)
            count += 1
        elif key == ord('q'):
            break
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

collect_stereo_pairs(0, 1, "stereo_data/cam01")
# cam0 ↔ cam2, cam0 ↔ cam3 同理
collect_stereo_pairs(0, 2, "stereo_data/cam02")
collect_stereo_pairs(0, 3, "stereo_data/cam03")
```

---

### 2.2 外參標定函數

```python
def calibrate_extrinsics(
    pair_dir: str,
    K_l: np.ndarray, D_l: np.ndarray,
    K_r: np.ndarray, D_r: np.ndarray,
    image_size: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    輸入：已標定好的兩台相機內參，以及同步圖像對目錄
    輸出：R, T（右相機相對於左相機的旋轉與平移）
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, dictionary
    )

    obj_pts_all, img_pts_l_all, img_pts_r_all = [], [], []

    left_paths  = sorted(glob.glob(f"{pair_dir}/left/*.png"))
    right_paths = sorted(glob.glob(f"{pair_dir}/right/*.png"))

    for lp, rp in zip(left_paths, right_paths):
        gray_l = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2GRAY)

        def detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            if ids is None or len(ids) < 6:
                return None, None
            ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if not ret or ch_corners is None or len(ch_corners) < 6:
                return None, None
            ch_corners = cv2.cornerSubPix(
                gray, ch_corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
            # 取得對應的 3D 物點
            obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
            return obj_points, img_points

        obj_l, img_l = detect(gray_l)
        obj_r, img_r = detect(gray_r)
        if obj_l is None or obj_r is None:
            continue

        # 取兩相機共同看到的角點
        # 簡化：直接用左圖的物點（兩者應一致，棋盤格未移動）
        obj_pts_all.append(obj_l)
        img_pts_l_all.append(img_l)
        img_pts_r_all.append(img_r)

    print(f"[外參標定] 有效對數: {len(obj_pts_all)}")
    assert len(obj_pts_all) >= 15, "有效對數不足"

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_all, img_pts_l_all, img_pts_r_all,
        K_l, D_l, K_r, D_r,
        image_size, flags=flags
    )
    print(f"[外參標定] RMS: {ret:.4f} px")
    return R, T


# 載入內參
def load_intrinsics(path: str):
    with open(path) as f:
        data = json.load(f)
    cams = {}
    for key, val in data.items():
        cams[key] = {
            "K": np.array(val["K"]),
            "D": np.array(val["D"]),
            "image_size": tuple(val["image_size"])
        }
    return cams

cams = load_intrinsics("intrinsics.json")

R01, T01 = calibrate_extrinsics(
    "stereo_data/cam01",
    cams["cam0"]["K"], cams["cam0"]["D"],
    cams["cam1"]["K"], cams["cam1"]["D"],
    cams["cam0"]["image_size"]
)
R02, T02 = calibrate_extrinsics(
    "stereo_data/cam02",
    cams["cam0"]["K"], cams["cam0"]["D"],
    cams["cam2"]["K"], cams["cam2"]["D"],
    cams["cam0"]["image_size"]
)
R03, T03 = calibrate_extrinsics(
    "stereo_data/cam03",
    cams["cam0"]["K"], cams["cam0"]["D"],
    cams["cam3"]["K"], cams["cam3"]["D"],
    cams["cam0"]["image_size"]
)
```

---

## Phase 3：建立 4×4 位姿矩陣（cam0 為世界原點）

```python
def Rt_to_T44(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """R(3,3), T(3,1) → T44 世界座標系到相機座標系的變換"""
    T44 = np.eye(4)
    T44[:3, :3] = R
    T44[:3,  3] = T.flatten()
    return T44

# cam0 = 世界原點
T_w2c = {
    "cam0": np.eye(4),
    "cam1": Rt_to_T44(R01, T01),
    "cam2": Rt_to_T44(R02, T02),
    "cam3": Rt_to_T44(R03, T03),
}

# 儲存
extrinsics_raw = {k: v.tolist() for k, v in T_w2c.items()}
with open("extrinsics_raw.json", "w") as f:
    json.dump(extrinsics_raw, f, indent=2)
```

---

## Phase 4：scipy Bundle Adjustment 全域優化

### 4.1 概念

優化目標：最小化所有相機、所有角點的重投影誤差總和。

```
min Σ_i Σ_j || π(K_i, D_i, T_i, P_j) - p_ij ||²

π = 投影函數（3D點 → 2D像素）
P_j = 棋盤格角點的 3D 座標
p_ij = 相機 i 觀測到的角點 j 的 2D 座標
```

---

### 4.2 Bundle Adjustment 實作

```python
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def project_points(K, D, R_vec, T, points_3d):
    """將 3D 點投影到 2D，包含畸變"""
    pts_2d, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        R_vec, T, K, D
    )
    return pts_2d.reshape(-1, 2)

def pack_params(T_w2c: dict, points_3d_list: list) -> np.ndarray:
    """
    打包所有優化變數為一維向量：
    - 每台相機位姿：rotvec(3) + tvec(3) = 6 個參數（cam0 固定不優化）
    - 每個棋盤格在世界座標下的位姿：rotvec(3) + tvec(3) = 6 個參數
    """
    params = []
    cam_keys = ["cam1", "cam2", "cam3"]
    for key in cam_keys:
        T44 = np.array(T_w2c[key])
        R_vec = Rotation.from_matrix(T44[:3, :3]).as_rotvec()
        t_vec = T44[:3, 3]
        params.extend(R_vec.tolist())
        params.extend(t_vec.tolist())
    for pts in points_3d_list:
        # 每個棋盤格觀測（世界系下的位姿）
        # 此處簡化：直接優化 3D 點座標
        params.extend(pts.flatten().tolist())
    return np.array(params)

def unpack_params(params: np.ndarray, T_w2c_init: dict, points_3d_list: list):
    cam_keys = ["cam1", "cam2", "cam3"]
    T_w2c_opt = {"cam0": np.eye(4)}
    idx = 0
    for key in cam_keys:
        R_vec = params[idx:idx+3]; idx += 3
        t_vec = params[idx:idx+3]; idx += 3
        R_mat = Rotation.from_rotvec(R_vec).as_matrix()
        T44   = np.eye(4)
        T44[:3, :3] = R_mat
        T44[:3,  3] = t_vec
        T_w2c_opt[key] = T44
    pts_opt = []
    for pts in points_3d_list:
        n = pts.size
        pts_opt.append(params[idx:idx+n].reshape(pts.shape))
        idx += n
    return T_w2c_opt, pts_opt


def build_ba_residuals(
    observations: list,  # list of (cam_key, pts_3d_world, pts_2d_obs)
    cams: dict,          # 內參字典
    T_w2c_init: dict,
    points_3d_list: list
):
    """
    observations: 每條觀測 = (cam_key, 3D點集, 2D觀測點集)
    """
    x0 = pack_params(T_w2c_init, points_3d_list)

    def residuals(params):
        T_w2c_opt, pts_opt = unpack_params(params, T_w2c_init, points_3d_list)
        res = []
        for (cam_key, pts3d_idx, pts_2d_obs) in observations:
            T44  = T_w2c_opt[cam_key]
            R_vec = Rotation.from_matrix(T44[:3, :3]).as_rotvec()
            t_vec = T44[:3, 3]
            K = np.array(cams[cam_key]["K"])
            D = np.array(cams[cam_key]["D"])
            pts_3d = pts_opt[pts3d_idx]
            pts_2d_proj = project_points(K, D, R_vec, t_vec, pts_3d)
            res.append((pts_2d_proj - pts_2d_obs).flatten())
        return np.concatenate(res)

    result = least_squares(
        residuals, x0,
        method='lm',           # Levenberg-Marquardt，適合中小規模
        max_nfev=2000,
        ftol=1e-8, xtol=1e-8
    )
    print(f"[BA] 初始殘差: {np.sqrt(np.mean(residuals(x0)**2)):.4f} px")
    print(f"[BA] 優化殘差: {np.sqrt(np.mean(result.fun**2)):.4f} px")
    T_w2c_opt, _ = unpack_params(result.x, T_w2c_init, points_3d_list)
    return T_w2c_opt

# ---- 執行 BA（需要先從標定採集中整理好 observations） ----
# T_w2c_optimized = build_ba_residuals(observations, cams, T_w2c, points_3d_list)

# 儲存優化後外參
# extrinsics_opt = {k: v.tolist() for k, v in T_w2c_optimized.items()}
# with open("extrinsics_optimized.json", "w") as f:
#     json.dump(extrinsics_opt, f, indent=2)
```

> **注意**：`observations` 的整理需要結合 Phase 2 的角點偵測結果，把每次採集的棋盤格 3D 位姿和各相機觀測的 2D 角點配對。此部分依具體採集腳本整合。

---

## Phase 5：Fast-FoundationStereo → Metric 深度圖

> 當前使用 FoundationStereo（官方 code）代替，待 Fast-FoundationStereo 發布後替換。

```python
import cv2
import numpy as np
import json

def stereo_rectify_and_depth(
    img_l: np.ndarray,
    img_r: np.ndarray,
    K_l: np.ndarray, D_l: np.ndarray,
    K_r: np.ndarray, D_r: np.ndarray,
    R: np.ndarray, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    輸入：原始圖像對 + 標定參數
    輸出：rectified_l, rectified_r, focal（像素）, baseline（公尺）
    """
    h, w = img_l.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)

    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    focal    = P1[0, 0]           # 像素焦距
    baseline = abs(T[0, 0]) if T.shape == (3,) else abs(T.flatten()[0])  # 公尺

    return rect_l, rect_r, Q, focal, baseline


def disparity_to_metric_depth(disparity: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    """
    disparity: FoundationStereo 輸出的視差圖（像素）
    回傳 metric 深度圖（公尺），disparity=0 的地方設為 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disparity > 0, focal * baseline / disparity, 0.0)
    return depth.astype(np.float32)


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    深度圖 → 相機座標系下的點雲
    回傳 (N, 3) ndarray
    """
    h, w = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts[pts[:, 2] > 0]
```

---

## Phase 6：DA3Nested 多視角推理 → 統一 Metric 點雲

```python
import torch
import numpy as np
import json
from depth_anything_3.api import DepthAnything3

def load_calibration(intrinsics_path: str, extrinsics_path: str):
    with open(intrinsics_path)  as f: intr = json.load(f)
    with open(extrinsics_path)  as f: extr = json.load(f)
    K_list = []
    T_list = []
    for key in ["cam0", "cam1", "cam2", "cam3"]:
        K = np.array(intr[key]["K"], dtype=np.float32)
        T = np.array(extr[key],      dtype=np.float32)  # (4, 4) world-to-cam
        K_list.append(K)
        T_list.append(T)
    return np.stack(K_list), np.stack(T_list)  # (4,3,3), (4,4,4)


def run_da3_metric(
    image_paths: list[str],
    intrinsics_path: str,
    extrinsics_path: str,
    output_dir: str = "./da3_output"
) -> object:
    """
    image_paths: [cam0_img, cam1_img, cam2_img, cam3_img]
    回傳 prediction 物件（含 .depth, .conf, .extrinsics, .intrinsics）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device)

    K_arr, T_arr = load_calibration(intrinsics_path, extrinsics_path)

    # DA3 內參格式：歸一化焦距，需要轉換
    # DA3 extrinsics: world-to-cam (4,4)，與我們的格式一致
    prediction = model.inference(
        image      = image_paths,
        extrinsics = T_arr,         # (N, 4, 4)
        intrinsics = K_arr,         # (N, 3, 3)
        export_dir = output_dir,
        export_format = "ply"       # 直接輸出點雲
    )

    print(f"depth shape: {prediction.depth.shape}")    # (4, H, W)
    print(f"conf  shape: {prediction.conf.shape}")     # (4, H, W)
    return prediction


def scale_align_with_stereo(
    da3_depth: np.ndarray,        # DA3 輸出的深度圖（如果不是 metric）
    stereo_depth: np.ndarray,     # Fast-FoundationStereo metric 深度圖
    conf_mask: np.ndarray = None  # DA3 信心圖，低信心點不參與對齊
) -> tuple[float, float]:
    """
    用 RANSAC 魯棒線性回歸求解：stereo_depth = s * da3_depth + t
    僅在使用 DA3 Main Series（非 Nested）時需要此步驟
    """
    from scipy.optimize import least_squares

    mask = (stereo_depth > 0.1) & (stereo_depth < 5.0) & (da3_depth > 0)
    if conf_mask is not None:
        mask &= conf_mask > np.percentile(conf_mask, 40)

    z_s = stereo_depth[mask].flatten()
    z_d = da3_depth[mask].flatten()

    # RANSAC
    best_s, best_t, best_inliers = 1.0, 0.0, 0
    n_iter = 100
    thresh = 0.05  # 公尺

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(len(z_s), size=min(50, len(z_s)), replace=False)
        A   = np.stack([z_d[idx], np.ones_like(z_d[idx])], axis=1)
        try:
            st, _, _, _ = np.linalg.lstsq(A, z_s[idx], rcond=None)
        except:
            continue
        s_c, t_c = st
        residuals = np.abs(z_s - (s_c * z_d + t_c))
        inliers   = np.sum(residuals < thresh)
        if inliers > best_inliers:
            best_s, best_t, best_inliers = s_c, t_c, inliers

    # 用所有 inlier 做最終最小二乘
    residuals = np.abs(z_s - (best_s * z_d + best_t))
    inlier_mask = residuals < thresh
    A_full = np.stack([z_d[inlier_mask], np.ones(inlier_mask.sum())], axis=1)
    st_final, _, _, _ = np.linalg.lstsq(A_full, z_s[inlier_mask], rcond=None)

    print(f"[尺度對齊] s={st_final[0]:.4f}, t={st_final[1]:.4f}, inliers={inlier_mask.sum()}")
    return float(st_final[0]), float(st_final[1])
```

---

## 完整執行腳本

```python
# main_pipeline.py
import cv2
import numpy as np

CAM_INDICES = [0, 1, 2, 3]

def capture_sync_frame(cap_list: list) -> list[np.ndarray]:
    """盡量同步讀取四台相機的一幀"""
    frames = []
    for cap in cap_list:
        ret, frame = cap.read()
        assert ret
        frames.append(frame)
    return frames

if __name__ == "__main__":
    import json

    # --- 載入標定結果 ---
    with open("intrinsics.json")          as f: intr_data = json.load(f)
    with open("extrinsics_optimized.json") as f: extr_data = json.load(f)

    K = {k: np.array(v["K"]) for k, v in intr_data.items()}
    D = {k: np.array(v["D"]) for k, v in intr_data.items()}
    T_w2c = {k: np.array(v) for k, v in extr_data.items()}

    # --- 開啟相機 ---
    caps = [cv2.VideoCapture(i) for i in CAM_INDICES]
    frames = capture_sync_frame(caps)
    for i, cap in enumerate(caps):
        cv2.imwrite(f"/tmp/cam{i}.jpg", frames[i])

    # --- Phase 5: 雙目 metric 深度 ---
    R01 = T_w2c["cam1"][:3, :3]
    T01 = T_w2c["cam1"][:3,  3]
    rect_l, rect_r, Q, focal, baseline = stereo_rectify_and_depth(
        frames[0], frames[1],
        K["cam0"], D["cam0"],
        K["cam1"], D["cam1"],
        R01, T01.reshape(3,1)
    )
    # TODO: 接 FoundationStereo/Fast-FoundationStereo inference
    # disparity = foundation_stereo_infer(rect_l, rect_r)
    # stereo_depth = disparity_to_metric_depth(disparity, focal, baseline)

    # --- Phase 6: DA3 metric 點雲 ---
    prediction = run_da3_metric(
        image_paths     = [f"/tmp/cam{i}.jpg" for i in range(4)],
        intrinsics_path = "intrinsics.json",
        extrinsics_path = "extrinsics_optimized.json",
        output_dir      = "./output"
    )
    # DA3Nested 輸出已是 metric，直接使用
    # 如使用 DA3 Main Series，執行 scale_align_with_stereo() 修正

    print("輸出點雲: ./output/scene.ply")
```

---

## 標定質量驗收標準

| 指標 | 合格閾值 | 檢查方式 |
|---|---|---|
| 單目 RMS 重投影誤差 | < 1.0 px | `calibrateCamera` 返回值 |
| 外參標定 RMS | < 1.0 px | `stereoCalibrate` 返回值 |
| BA 優化後殘差 | < 0.5 px | `least_squares` 殘差 |
| 雙目視差圖極線對齊 | 極線偏差 < 2 px | 目視平行線條 |
| 統一點雲棋盤格角點誤差 | < 5 mm | 量角點間距對比實際值 |

---

