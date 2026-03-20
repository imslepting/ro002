"""extrinsics.json 合併式讀寫 — 每次只寫入一個配對，不覆蓋其他"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np


def Rt_to_T44(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """3x3 旋轉 + 3x1 平移 → 4x4 齊次變換矩陣"""
    T44 = np.eye(4)
    T44[:3, :3] = R
    T44[:3, 3] = T.flatten()
    return T44


def load_extrinsics(path: str) -> dict:
    """讀取 extrinsics.json，不存在或為空則返回空 dict"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def save_extrinsics(result: dict, path: str) -> None:
    """合併寫入單對的標定結果，不覆蓋其他配對

    Args:
        result: dict with keys: pair_name, cam_left, cam_right, R, T, rms, num_pairs_used
        path: extrinsics.json 路徑
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = load_extrinsics(path)

    pair_name = result["pair_name"]
    R = np.array(result["R"])
    T = np.array(result["T"])
    T44 = Rt_to_T44(R, T)

    data[pair_name] = {
        "cam_left": result["cam_left"],
        "cam_right": result["cam_right"],
        "R": R.tolist(),
        "T": T.tolist(),
        "T_right_to_left": T44.tolist(),
        "rms": round(result["rms"], 6),
        "num_pairs_used": result["num_pairs_used"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[IO] 已保存 {pair_name} 外參結果至 {path}")


def get_calibrated_pairs(path: str) -> set[str]:
    """返回已標定的配對名稱集合"""
    data = load_extrinsics(path)
    return set(data.keys())


def load_pair_result(pair_name: str, path: str) -> dict | None:
    """從 extrinsics.json 載入單對的結果"""
    data = load_extrinsics(path)
    if pair_name not in data:
        return None
    return data[pair_name]
