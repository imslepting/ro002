"""Agent Tool 定義 + 執行器

定義 VLM Agent 可用的工具 schema，以及執行各工具的 ToolExecutor。
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from shared.types import SAM3Result, CapturePointResult
from phase5_vlm_planning.src.vlm_client import VLMClient

log = logging.getLogger(__name__)

# ── Tool Schema 定義（Anthropic API tool-use 格式） ──

TOOLS = [
    {
        "name": "capture_scene",
        "description": (
            "抓取當前相機畫面。返回場景 RGB 圖像和深度圖。"
            "在開始任何操作前應先調用此工具觀察場景。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "segment_object",
        "description": (
            "使用 SAM3 模型分割目標物件。輸入物件的視覺描述，返回分割標注圖像"
            "（目標物件會被高亮顯示）。觀察返回的圖像判斷分割是否正確。"
            "【重要】object_description 必須使用英文，SAM3 模型只接受英文輸入。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "目標物件的英文視覺描述（必須英文），例如 \"red cup\"、\"blue bottle on the left\"",
                },
            },
            "required": ["object_description"],
        },
    },
    {
        "name": "compute_grasp",
        "description": (
            "計算機器人夾爪的抓取位姿。必須先成功執行 segment_object。"
            "返回抓取標注圖像（綠色十字=接觸點，紅色箭頭=夾爪靠近方向，"
            "綠色線=夾爪張開方向）和抓取參數。觀察返回的圖像判斷抓取位置是否合理。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "save_plan",
        "description": (
            "保存執行計劃到 plan.json。必須在確認抓取位置合理後調用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "任務描述（用於日誌記錄）",
                },
            },
            "required": ["task_description"],
        },
    },
]


class ToolExecutor:
    """執行 VLM 調用的工具"""

    def __init__(
        self,
        sam3_skill,
        capture_skill,
        K_rect: np.ndarray,
        T_cam2arm: np.ndarray,
        plan_serializer=None,
        on_log: Callable[[str], None] | None = None,
    ):
        self._sam3 = sam3_skill
        self._capture = capture_skill
        self._K = K_rect
        self._T = T_cam2arm
        self._plan_serializer = plan_serializer
        self._on_log = on_log

        # 中間狀態
        self._snapshot_rgb: np.ndarray | None = None
        self._snapshot_depth: np.ndarray | None = None
        self._sam3_result: SAM3Result | None = None
        self._capture_result: CapturePointResult | None = None
        self._session_dir: str | None = None

        # Skip verify flag
        self._skip_verify = False
        # LLM 模式：tool result 不包含圖片，只用文字描述
        self._no_images = False

    def set_snapshot(self, rgb: np.ndarray, depth: np.ndarray):
        """設置當前場景快照"""
        self._snapshot_rgb = rgb.copy()
        self._snapshot_depth = depth.copy()

    def set_skip_verify(self, flag: bool):
        self._skip_verify = flag

    def set_no_images(self, flag: bool):
        """LLM 模式：tool result 不返回圖片，只用文字"""
        self._no_images = flag

    def execute(self, tool_name: str, tool_input: dict) -> list[dict]:
        """執行工具，返回 tool_result content blocks

        Returns: list of content blocks, e.g.:
          [{"type": "image", ...}, {"type": "text", "text": "..."}]
        """
        self._log(f"Executing tool: {tool_name}")
        try:
            if tool_name == "capture_scene":
                return self._exec_capture_scene()
            elif tool_name == "segment_object":
                return self._exec_segment(tool_input)
            elif tool_name == "compute_grasp":
                return self._exec_grasp()
            elif tool_name == "save_plan":
                return self._exec_save_plan(tool_input)
            else:
                return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]
        except Exception as e:
            log.exception(f"Tool execution error: {tool_name}")
            return [{"type": "text", "text": f"Tool error: {e}"}]

    def _exec_capture_scene(self) -> list[dict]:
        if self._snapshot_rgb is None or self._snapshot_depth is None:
            return [{"type": "text", "text": "Error: 沒有可用的場景快照，請先啟動相機"}]

        # 不返回圖片 — VLM 已在初始輸入中收到場景圖，無需重複查看
        # 只返回深度統計文字，供 VLM 決策
        valid = self._snapshot_depth > 0
        h, w = self._snapshot_depth.shape[:2]
        if valid.any():
            d_min = self._snapshot_depth[valid].min()
            d_max = self._snapshot_depth[valid].max()
            d_mean = self._snapshot_depth[valid].mean()
            text = (
                f"場景已捕獲，分辨率 {w}x{h}，"
                f"深度範圍 {d_min:.2f}-{d_max:.2f}m，"
                f"平均深度 {d_mean:.2f}m，"
                f"有效深度像素: {valid.sum()}/{h*w}。"
                f"（場景圖片與你已看到的初始圖片相同，請根據之前的圖片判斷。）"
            )
        else:
            text = f"場景已捕獲，分辨率 {w}x{h}，但深度圖無有效數據"

        if self._skip_verify:
            text += "\n用戶已手動確認此結果正確，請直接進行下一步。"
            self._skip_verify = False

        self._log(text)
        return [{"type": "text", "text": text}]

    def _exec_segment(self, inp: dict) -> list[dict]:
        if self._snapshot_rgb is None:
            return [{"type": "text", "text": "Error: 沒有場景圖像，請先 capture_scene"}]

        desc = inp.get("object_description", "")
        if not desc:
            return [{"type": "text", "text": "Error: 需要提供 object_description"}]

        self._log(f"SAM3 分割: {desc}")
        result = self._sam3.segment(self._snapshot_rgb, desc)
        self._sam3_result = result

        blocks = []
        # VLM 模式返回標注圖，LLM 模式只返回文字
        if not self._no_images:
            blocks.append(VLMClient.make_image_block(result.annotated_image))

        if len(result.masks) == 0:
            text = f"分割完成：未找到匹配 \"{desc}\" 的物件"
        else:
            text = (
                f"分割完成：{len(result.masks)} 個候選 mask，"
                f"最高分 {result.best_score:.3f}，"
                f"mask 像素數 {result.best_mask.sum()}"
            )
            if self._no_images:
                text += "（LLM 模式：跳過圖片驗證，請根據數值判斷並直接進行下一步）"

        if self._skip_verify:
            text += "\n用戶已手動確認此結果正確，請直接進行下一步。"
            self._skip_verify = False

        blocks.append({"type": "text", "text": text})
        self._log(text)
        return blocks

    def _exec_grasp(self) -> list[dict]:
        if self._sam3_result is None or len(self._sam3_result.masks) == 0:
            return [{"type": "text", "text": "Error: 需要先成功執行 segment_object"}]
        if self._snapshot_rgb is None or self._snapshot_depth is None:
            return [{"type": "text", "text": "Error: 沒有場景數據"}]

        self._log("計算抓取位姿...")
        result = self._capture.capture(
            self._snapshot_rgb,
            self._snapshot_depth,
            self._sam3_result.best_mask,
            self._K,
            self._T,
        )
        self._capture_result = result

        # 不發送圖片給 VLM（避免 ClaudeCodeVLM 額外 Read 延遲）
        # 抓取標注圖由 GUI 直接從 capture_result.annotated_image 顯示
        blocks = []

        if result.num_candidates == 0:
            text = "抓取計算完成：未找到有效抓取姿態"
        else:
            pos = result.pose_arm[:3, 3]
            text = (
                f"抓取計算完成：score={result.grasp_score:.3f}，"
                f"夾爪寬度={result.grasp_width*1000:.1f}mm，"
                f"位置(arm)=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]，"
                f"候選數={result.num_candidates}，"
                f"點雲大小={result.cropped_cloud_size}。"
                f"（抓取標注圖已顯示在 GUI 中，score > 0.8 通常可靠，"
                f"請根據數值判斷是否合理並決定下一步。）"
            )

        if self._skip_verify:
            text += "\n用戶已手動確認此結果正確，請直接進行下一步。"
            self._skip_verify = False

        blocks.append({"type": "text", "text": text})
        self._log(text)
        return blocks

    def _exec_save_plan(self, inp: dict) -> list[dict]:
        if self._capture_result is None or self._capture_result.num_candidates == 0:
            return [{"type": "text",
                      "text": "Error: 沒有有效的抓取結果，無法保存計劃"}]

        task_desc = inp.get("task_description", "unknown task")

        if self._plan_serializer is not None:
            from phase5_vlm_planning.src.plan_serializer import PlanSerializer
            ser: PlanSerializer = self._plan_serializer

            # 保存 session
            self._session_dir = ser.save_session(
                task_text=task_desc,
                rgb=self._snapshot_rgb,
                depth=self._snapshot_depth,
                sam3_result=self._sam3_result,
                capture_result=self._capture_result,
            )

            # 保存 plan.json
            plan_path = ser.save_plan(
                task_text=task_desc,
                capture_result=self._capture_result,
                session_dir=self._session_dir,
            )

            text = f"計劃已保存到 {plan_path}"
        else:
            text = "計劃已生成（PlanSerializer 未配置，未存檔）"

        self._log(text)
        return [{"type": "text", "text": text}]

    def _log(self, msg: str):
        log.info(msg)
        if self._on_log:
            self._on_log(msg)

    @property
    def sam3_result(self) -> SAM3Result | None:
        return self._sam3_result

    @property
    def capture_result(self) -> CapturePointResult | None:
        return self._capture_result

    @property
    def session_dir(self) -> str | None:
        return self._session_dir
