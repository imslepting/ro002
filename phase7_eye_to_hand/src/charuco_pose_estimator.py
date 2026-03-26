from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from phase1_intrinsics.src.board_generator import create_board
from phase1_intrinsics.src.charuco_detector import detect_charuco


@dataclass
class TargetPoseResult:
    success: bool
    R_target2cam: np.ndarray | None
    t_target2cam: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    num_corners: int
    debug_frame: np.ndarray | None


class CharucoPoseEstimator:
    def __init__(
        self,
        charuco_cfg: dict,
        K: np.ndarray,
        D: np.ndarray,
        min_corners: int = 6,
    ):
        self._board, self._dictionary = create_board(charuco_cfg)
        self._K = np.asarray(K, dtype=np.float64)
        self._D = np.asarray(D, dtype=np.float64)
        self._min_corners = int(min_corners)

    def estimate(self, bgr_frame: np.ndarray) -> TargetPoseResult:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        det = detect_charuco(
            gray,
            board=self._board,
            dictionary=self._dictionary,
            refine_subpix=True,
            min_corners=self._min_corners,
        )

        debug = bgr_frame.copy()
        if det.charuco_corners is not None and det.charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(debug, det.charuco_corners, det.charuco_ids)

        if not det.success or det.charuco_corners is None or det.charuco_ids is None:
            return TargetPoseResult(
                success=False,
                R_target2cam=None,
                t_target2cam=None,
                rvec=None,
                tvec=None,
                num_corners=det.num_corners,
                debug_frame=debug,
            )

        obj_all = self._board.getChessboardCorners().astype(np.float64)
        ids = det.charuco_ids.reshape(-1)
        obj_pts = obj_all[ids]
        img_pts = det.charuco_corners.reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=self._K,
            distCoeffs=self._D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return TargetPoseResult(
                success=False,
                R_target2cam=None,
                t_target2cam=None,
                rvec=None,
                tvec=None,
                num_corners=det.num_corners,
                debug_frame=debug,
            )

        R, _ = cv2.Rodrigues(rvec)
        cv2.drawFrameAxes(debug, self._K, self._D, rvec, tvec, 0.06)

        return TargetPoseResult(
            success=True,
            R_target2cam=R.astype(np.float64),
            t_target2cam=tvec.reshape(3, 1).astype(np.float64),
            rvec=rvec.reshape(3, 1).astype(np.float64),
            tvec=tvec.reshape(3, 1).astype(np.float64),
            num_corners=det.num_corners,
            debug_frame=debug,
        )
