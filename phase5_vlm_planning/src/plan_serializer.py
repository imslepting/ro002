"""Session 存檔 + plan.json 序列化"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from shared.types import SAM3Result, CapturePointResult


class PlanSerializer:
    """管理 session 存檔和 plan.json 生成"""

    def __init__(self, output_base: str = "phase5_vlm_planning/outputs"):
        self._output_base = output_base

    def save_session(
        self,
        task_text: str,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        sam3_result: SAM3Result | None = None,
        capture_result: CapturePointResult | None = None,
        dialogue: list[dict] | None = None,
    ) -> str:
        """保存 session 到輸出目錄，返回 session_dir"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self._output_base, "sessions", timestamp)
        os.makedirs(session_dir, exist_ok=True)

        # 任務描述
        with open(os.path.join(session_dir, "task_input.txt"), "w") as f:
            f.write(task_text)

        # RGB 快照
        if rgb is not None:
            cv2.imwrite(os.path.join(session_dir, "rgb_snapshot.jpg"), rgb)

        # 深度圖
        if depth is not None:
            np.save(os.path.join(session_dir, "depth.npy"), depth)

        # SAM3 標注圖
        if sam3_result is not None and sam3_result.annotated_image is not None:
            cv2.imwrite(
                os.path.join(session_dir, "sam3_annotated.jpg"),
                sam3_result.annotated_image,
            )
            # 保存 mask
            if sam3_result.best_mask is not None:
                np.save(
                    os.path.join(session_dir, "best_mask.npy"),
                    sam3_result.best_mask,
                )

        # Grasp 標注圖
        if capture_result is not None and capture_result.annotated_image is not None:
            cv2.imwrite(
                os.path.join(session_dir, "grasp_viz.jpg"),
                capture_result.annotated_image,
            )

        # 對話記錄
        if dialogue:
            with open(os.path.join(session_dir, "vlm_dialogue.jsonl"), "w") as f:
                for turn in dialogue:
                    # 過濾掉大的 base64 圖片數據
                    clean = self._clean_turn(turn)
                    f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        return session_dir

    def save_plan(
        self,
        task_text: str,
        capture_result: CapturePointResult,
        session_dir: str,
    ) -> str:
        """保存 plan.json，返回文件路徑"""
        pose = capture_result.pose_arm
        pos = pose[:3, 3].tolist()
        rot = pose[:3, :3].tolist()

        plan = {
            "task": task_text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "grasp": {
                "position": pos,
                "rotation": rot,
                "score": float(capture_result.grasp_score),
                "width_m": float(capture_result.grasp_width),
                "pixel": list(capture_result.grasp_pixel),
                "num_candidates": capture_result.num_candidates,
            },
            "session_dir": session_dir,
        }

        plan_path = os.path.join(session_dir, "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return plan_path

    @staticmethod
    def _clean_turn(turn: dict) -> dict:
        """移除 base64 圖片數據，只保留元數據"""
        clean = {}
        for k, v in turn.items():
            if k == "content" and isinstance(v, list):
                clean_content = []
                for block in v:
                    if isinstance(block, dict):
                        if block.get("type") == "image":
                            clean_content.append({
                                "type": "image",
                                "note": "(base64 data removed)",
                            })
                        elif block.get("type") == "tool_result":
                            clean_sub = []
                            for sub in block.get("content", []):
                                if isinstance(sub, dict) and sub.get("type") == "image":
                                    clean_sub.append({
                                        "type": "image",
                                        "note": "(base64 data removed)",
                                    })
                                else:
                                    clean_sub.append(sub)
                            clean_content.append({
                                **block,
                                "content": clean_sub,
                            })
                        else:
                            clean_content.append(block)
                    else:
                        clean_content.append(block)
                clean[k] = clean_content
            else:
                clean[k] = v
        return clean
