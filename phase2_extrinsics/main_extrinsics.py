"""Phase 2 — 雙目外參標定 GUI 主入口

用法:
    python phase2_extrinsics/main_extrinsics.py
"""

from __future__ import annotations

import os
import sys

import yaml

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.board_generator import create_board
from phase2_extrinsics.src.tk_gui import run_extrinsics_gui_tk
from phase2_extrinsics.src.extrinsics_io import save_extrinsics, get_calibrated_pairs


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_extrinsics_gui(config: dict) -> None:
    """主外參標定 GUI 流程"""
    cameras_cfg = config["cameras"]
    charuco_cfg = config["calibration"]["charuco"]
    acceptance_cfg = config["calibration"]["acceptance"]
    max_rms = acceptance_cfg.get("max_rms_extrinsics", 1.0)

    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")
    extrinsics_path = os.path.join(_ROOT, "phase2_extrinsics", "outputs", "extrinsics.json")
    pair_base_dir = os.path.join(_ROOT, "phase2_extrinsics", "outputs", "raw_pairs")
    reports_dir = os.path.join(_ROOT, "phase2_extrinsics", "outputs", "reports")

    board, dictionary = create_board(charuco_cfg)

    def _save(result_dict):
        save_extrinsics(result_dict, extrinsics_path)
        print(f"[Phase 2] {result_dict['pair_name']}: 外參標定已保存!")

    def _get_calibrated():
        return get_calibrated_pairs(extrinsics_path)

    run_extrinsics_gui_tk(
        cameras_cfg=cameras_cfg,
        charuco_cfg=charuco_cfg,
        max_rms=max_rms,
        intrinsics_path=intrinsics_path,
        extrinsics_path=extrinsics_path,
        pair_base_dir=pair_base_dir,
        reports_dir=reports_dir,
        board=board,
        dictionary=dictionary,
        save_extrinsics_fn=_save,
        get_calibrated_pairs_fn=_get_calibrated,
    )


def main() -> None:
    config = load_config()
    print("[Phase 2] 啟動外參標定 GUI...")
    run_extrinsics_gui(config)
    print("[Phase 2] 標定結束。")


if __name__ == "__main__":
    main()
