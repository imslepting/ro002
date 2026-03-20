"""生成硬體診斷報告（JSON + 截圖）"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np

from shared.types import CameraTestResult


def save_report(
    test_results: dict[str, CameraTestResult | None],
    frames: dict[str, np.ndarray] | None = None,
    output_dir: str = "phase0_hw_diagnostics/outputs/reports",
) -> str:
    """保存診斷報告

    Args:
        test_results: {cam_name: CameraTestResult | None}
        frames: {cam_name: BGR frame} 當前幀截圖
        output_dir: 輸出目錄

    Returns:
        報告 JSON 文件路徑
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 構建報告數據
    n_expected = len(test_results)
    n_ok = sum(1 for r in test_results.values() if r and r.status == "OK")
    n_err = sum(1 for r in test_results.values() if r is None or r.status == "ERROR")

    if n_err > 0:
        overall = "ERROR"
    elif n_ok < n_expected:
        overall = "WARNING"
    else:
        overall = "OK"

    report = {
        "timestamp": timestamp,
        "system": {
            "expected_cameras": n_expected,
            "detected_cameras": n_expected - n_err,
            "overall_status": overall,
        },
        "cameras": {},
    }

    for cam_name, result in sorted(test_results.items()):
        if result is None:
            report["cameras"][cam_name] = {
                "connected": False,
                "status": "ERROR",
                "warnings": ["設備未找到"],
            }
        else:
            report["cameras"][cam_name] = {
                "connected": True,
                "resolution": list(result.resolution),
                "fps_reported": round(result.fps_reported, 1),
                "fps_measured": round(result.fps_measured, 1),
                "frame_success_rate": round(result.frame_success_rate, 3),
                "mean_brightness": round(result.mean_brightness, 1),
                "sharpness_score": round(result.sharpness_score, 1),
                "is_color": result.is_color,
                "status": result.status,
                "warnings": result.warnings,
            }

    # 保存 JSON
    report_path = os.path.join(output_dir, f"{timestamp}_hw_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存截圖
    if frames:
        snap_dir = os.path.join(output_dir, f"{timestamp}_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        for cam_name, frame in frames.items():
            cv2.imwrite(os.path.join(snap_dir, f"{cam_name}.png"), frame)

    print(f"[Phase 0] 報告已保存: {report_path}")
    return report_path
