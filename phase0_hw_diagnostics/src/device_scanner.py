"""掃描並枚舉所有可用相機設備"""

from __future__ import annotations

import glob
import cv2

from shared.types import DetectedDevice


def scan_video_devices(max_index: int = 16) -> list[DetectedDevice]:
    """嘗試打開 index 0 ~ max_index-1，返回所有可用的設備列表"""
    # 建立 /dev/videoN → index 的映射
    dev_paths = {}
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(path.replace("/dev/video", ""))
            dev_paths[idx] = path
        except ValueError:
            pass

    devices: list[DetectedDevice] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        is_open = cap.isOpened()
        backend = cap.getBackendName() if is_open else ""
        cap.release()

        if is_open:
            devices.append(DetectedDevice(
                index=idx,
                is_open=True,
                backend_name=backend,
                device_path=dev_paths.get(idx, f"/dev/video{idx}"),
            ))

    return devices


def match_expected_cameras(
    detected: list[DetectedDevice],
    camera_config: dict,
) -> dict[str, dict]:
    """將檢測到的設備與 settings.yaml 中的期望相機進行比對

    返回: {cam_name: {"expected_index": int, "role": str, "detected": bool, "device": DetectedDevice | None}}
    """
    detected_by_index = {d.index: d for d in detected}

    result = {}
    for cam_name, cam_info in camera_config.items():
        expected_idx = cam_info["index"]
        device = detected_by_index.get(expected_idx)
        result[cam_name] = {
            "expected_index": expected_idx,
            "role": cam_info.get("role", "unknown"),
            "detected": device is not None,
            "device": device,
        }

    return result
