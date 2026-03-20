"""Phase 1 — 相機內參標定 GUI 主入口

用法:
    python phase1_intrinsics/main_intrinsics.py              # 互動 GUI
    python phase1_intrinsics/main_intrinsics.py --generate-board  # 只生成標定板
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.board_generator import create_board, generate_board_image, save_board_pdf
from phase1_intrinsics.src.tk_gui import run_calibration_gui_tk
from phase1_intrinsics.src.intrinsics_io import save_intrinsics, get_calibrated_cameras


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_board(config: dict) -> None:
    """生成 ChArUco 標定板 PDF"""
    charuco_cfg = config["calibration"]["charuco"]
    board, _ = create_board(charuco_cfg)
    board_image = generate_board_image(board)
    pdf_path = os.path.join(_ROOT, "assets", "charuco_board.pdf")
    save_board_pdf(
        board_image, pdf_path,
        cols=charuco_cfg["cols"],
        rows=charuco_cfg["rows"],
        square_size_m=charuco_cfg["square_size"],
    )


def run_calibration_gui(config: dict) -> None:
    """主標定 GUI 流程（tkinter 版）"""
    cameras_cfg = config["cameras"]
    charuco_cfg = config["calibration"]["charuco"]
    acceptance_cfg = config["calibration"]["acceptance"]
    max_rms = acceptance_cfg.get("max_rms_intrinsics", 1.0)

    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")
    reports_dir = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "reports")

    board, dictionary = create_board(charuco_cfg)

    def _generate_board():
        generate_board(config)

    def _get_calibrated():
        return get_calibrated_cameras(intrinsics_path)

    def _save(calib_result):
        save_intrinsics(calib_result, intrinsics_path)
        print(f"[Phase 1] {calib_result.cam_name}: 標定已保存!")

    run_calibration_gui_tk(
        cameras_cfg=cameras_cfg,
        charuco_cfg=charuco_cfg,
        max_rms=max_rms,
        intrinsics_path=intrinsics_path,
        reports_dir=reports_dir,
        board=board,
        dictionary=dictionary,
        generate_board_fn=_generate_board,
        get_calibrated_fn=_get_calibrated,
        save_intrinsics_fn=_save,
        root_dir=_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Camera Intrinsics Calibration")
    parser.add_argument("--generate-board", action="store_true",
                        help="Only generate the ChArUco calibration board PDF")
    args = parser.parse_args()

    config = load_config()

    if args.generate_board:
        generate_board(config)
        return

    print("[Phase 1] 啟動內參標定 GUI...")
    run_calibration_gui(config)
    print("[Phase 1] 標定結束。")


if __name__ == "__main__":
    main()
