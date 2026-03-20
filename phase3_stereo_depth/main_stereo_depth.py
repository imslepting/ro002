"""Phase 3 — Stereo Depth GUI 主入口

用法:
    python phase3_stereo_depth/main_stereo_depth.py
"""

from __future__ import annotations

import os
import sys

import yaml

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase3_stereo_depth.src.tk_gui import run_stereo_depth_gui


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")
    extrinsics_path = os.path.join(_ROOT, "phase2_extrinsics", "outputs", "extrinsics.json")

    print("[Phase 3] 啟動 Stereo Depth GUI...")
    run_stereo_depth_gui(
        cfg=config,
        intrinsics_path=intrinsics_path,
        extrinsics_path=extrinsics_path,
        pair_name="cam0_cam1",
    )
    print("[Phase 3] 結束。")


if __name__ == "__main__":
    main()
