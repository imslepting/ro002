"""Phase 8 - RealSense live point cloud viewer.

Usage:
    conda run -n ro002 python phase8_realsense_pointcloud/main_realsense_pointcloud.py
"""

from __future__ import annotations

import os
import sys

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase8_realsense_pointcloud.src.tk_gui import run_realsense_gui


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    print("[Phase 8] Start RealSense point cloud viewer...")
    run_realsense_gui(cfg, _ROOT)
    print("[Phase 8] Exit")


if __name__ == "__main__":
    main()
