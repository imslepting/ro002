"""Phase 0 — 硬體診斷 GUI 主入口

用法:
    python -m phase0_hw_diagnostics.main_diagnostics
    # 或
    python phase0_hw_diagnostics/main_diagnostics.py

快捷鍵:
    Q — 退出
    S — 保存診斷報告
    R — 重新掃描設備
"""

from __future__ import annotations

import os
import sys

import yaml

# 確保項目根目錄在 sys.path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.types import CameraTestResult
from phase0_hw_diagnostics.src.device_scanner import scan_video_devices, match_expected_cameras
from phase0_hw_diagnostics.src.camera_tester import test_camera
from phase0_hw_diagnostics.src.tk_gui import run_tk_display
from phase0_hw_diagnostics.src.hw_report import save_report


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """載入全局配置"""
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_scan_and_test(config: dict) -> dict[str, CameraTestResult | None]:
    """掃描設備並逐台測試，返回 {cam_name: CameraTestResult | None}"""
    cameras_cfg = config["cameras"]
    p0_cfg = config.get("phase0", {})

    max_index = p0_cfg.get("scan_max_index", 16)
    test_frames = p0_cfg.get("test_frames", 30)
    sharpness_thr = p0_cfg.get("sharpness_threshold", 50.0)
    brightness_range = tuple(p0_cfg.get("brightness_range", [30, 230]))

    print("[Phase 0] 掃描視頻設備...")
    detected = scan_video_devices(max_index=max_index)
    print(f"[Phase 0] 檢測到 {len(detected)} 個視頻設備: "
          f"{[d.index for d in detected]}")

    matched = match_expected_cameras(detected, cameras_cfg)

    results: dict[str, CameraTestResult | None] = {}
    for cam_name, match in sorted(matched.items()):
        if not match["detected"]:
            print(f"[Phase 0] {cam_name} (index {match['expected_index']}): 未檢測到")
            results[cam_name] = None
            continue

        print(f"[Phase 0] 測試 {cam_name} (index {match['expected_index']})...")
        result = test_camera(
            cam_index=match["expected_index"],
            n_frames=test_frames,
            sharpness_threshold=sharpness_thr,
            brightness_range=brightness_range,
        )
        print(f"[Phase 0]   {result.status} — "
              f"{result.resolution[0]}x{result.resolution[1]} "
              f"{result.fps_measured:.1f}fps")
        if result.warnings:
            for w in result.warnings:
                print(f"[Phase 0]   ! {w}")
        results[cam_name] = result

    return results


def main() -> None:
    config = load_config()
    cameras_cfg = config["cameras"]
    p0_cfg = config.get("phase0", {})
    tile_size = tuple(p0_cfg.get("tile_size", [640, 480]))

    # 初始掃描 + 測試
    test_results = run_scan_and_test(config)

    # 保存回調
    def on_save(frames: dict):
        save_report(test_results, frames)

    # 重新掃描回調
    def on_rescan():
        nonlocal test_results
        test_results = run_scan_and_test(config)
        return test_results

    # 啟動 GUI
    print("[Phase 0] 啟動診斷 GUI...")
    run_tk_display(
        camera_configs=cameras_cfg,
        test_results=test_results,
        tile_size=tile_size,
        on_save=on_save,
        on_rescan=on_rescan,
    )
    print("[Phase 0] 診斷結束。")


if __name__ == "__main__":
    main()
