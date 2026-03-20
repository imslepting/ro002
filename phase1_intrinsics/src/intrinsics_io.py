"""intrinsics.json 合併式讀寫 — 每次只寫入一台相機，不覆蓋其他"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from shared.types import CalibResult


def load_intrinsics(path: str) -> dict:
    """讀取 intrinsics.json，不存在或為空則返回空 dict"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def save_intrinsics(calib_result: CalibResult, path: str) -> None:
    """合併寫入單台相機的標定結果，不覆蓋其他相機"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = load_intrinsics(path)

    data[calib_result.cam_name] = {
        "K": calib_result.K.tolist(),
        "D": calib_result.D.tolist(),
        "image_size": list(calib_result.image_size),
        "rms": round(calib_result.rms, 6),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[IO] 已保存 {calib_result.cam_name} 標定結果至 {path}")


def get_calibrated_cameras(path: str) -> set[str]:
    """返回已標定的相機名稱集合"""
    data = load_intrinsics(path)
    return set(data.keys())


def load_calib_result(cam_name: str, path: str) -> CalibResult | None:
    """從 intrinsics.json 載入單台相機的 CalibResult"""
    data = load_intrinsics(path)
    if cam_name not in data:
        return None
    entry = data[cam_name]
    return CalibResult(
        cam_name=cam_name,
        K=np.array(entry["K"]),
        D=np.array(entry["D"]),
        image_size=tuple(entry["image_size"]),
        rms=entry["rms"],
    )
