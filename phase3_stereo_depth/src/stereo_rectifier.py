"""立體校正器 — 載入標定參數，計算校正映射，提供 focal/baseline"""

from __future__ import annotations

import cv2
import numpy as np

from phase1_intrinsics.src.intrinsics_io import load_calib_result
from phase2_extrinsics.src.extrinsics_io import load_pair_result


class StereoRectifier:
    """載入 Phase 1/2 標定結果並提供立體校正功能"""

    def __init__(
        self,
        intrinsics_path: str,
        extrinsics_path: str,
        pair_name: str = "cam0_cam1",
    ):
        cam_left, cam_right = pair_name.split("_", 1)

        # 載入內參
        calib_l = load_calib_result(cam_left, intrinsics_path)
        calib_r = load_calib_result(cam_right, intrinsics_path)
        if calib_l is None or calib_r is None:
            missing = []
            if calib_l is None:
                missing.append(cam_left)
            if calib_r is None:
                missing.append(cam_right)
            raise ValueError(f"Missing intrinsics for: {', '.join(missing)}")

        # 載入外參
        pair_data = load_pair_result(pair_name, extrinsics_path)
        if pair_data is None:
            raise ValueError(f"Missing extrinsics for pair: {pair_name}")

        K_l, D_l = calib_l.K, calib_l.D
        K_r, D_r = calib_r.K, calib_r.D
        R = np.array(pair_data["R"])
        T = np.array(pair_data["T"])
        self._image_size = calib_l.image_size  # (width, height)

        # stereoRectify — 取得 P1, P2, Q 以提取 focal/baseline
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, self._image_size, R, T,
            alpha=0,
        )

        # 計算校正映射表
        self._map1_l, self._map2_l = cv2.initUndistortRectifyMap(
            K_l, D_l, R1, P1, self._image_size, cv2.CV_16SC2,
        )
        self._map1_r, self._map2_r = cv2.initUndistortRectifyMap(
            K_r, D_r, R2, P2, self._image_size, cv2.CV_16SC2,
        )

        self._Q = Q
        self._P1 = P1
        self._focal_length = float(P1[0, 0])
        self._baseline = float(abs(T.flatten()[0]))

    def rectify(
        self, img_left: np.ndarray, img_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """套用預計算的校正映射表"""
        rect_l = cv2.remap(img_left, self._map1_l, self._map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self._map1_r, self._map2_r, cv2.INTER_LINEAR)
        return rect_l, rect_r

    @property
    def focal_length(self) -> float:
        """校正後焦距（像素），來自 P1[0,0]"""
        return self._focal_length

    @property
    def baseline(self) -> float:
        """基線長度（公尺），來自 |T[0]|"""
        return self._baseline

    @property
    def Q(self) -> np.ndarray:
        """4x4 視差到深度投影矩陣"""
        return self._Q

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height)"""
        return self._image_size
