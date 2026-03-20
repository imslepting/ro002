"""Tool-Use Agentic Loop

VLM 自主決策循環：接收用戶指令，通過 tool-use 調用 SAM3、CapturePoint 等工具。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from shared.types import SAM3Result, CapturePointResult
from phase5_vlm_planning.src.vlm_client import VLMClient
from phase5_vlm_planning.src.agent_tools import TOOLS, ToolExecutor

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 ARM-VLM Agent，一個機器人手臂操作規劃助手。

你的任務是根據用戶指令，使用提供的工具完成物件抓取規劃。

## 工作流程

1. **觀察場景**：場景圖像已在用戶消息中提供，直接觀察即可。若需深度統計可用 capture_scene。
2. **分割目標**：使用 segment_object 工具分割用戶指定的物件
   - 仔細觀察返回的標注圖像，確認分割是否正確
   - 如果分割錯誤，修改描述重試（最多 3 次）
3. **計算抓取**：使用 compute_grasp 工具計算夾爪抓取位姿
   - 抓取標注圖會顯示在 GUI 中，根據返回的 score 和位置數值判斷是否合理
   - score > 0.8 通常表示抓取可靠，直接進行下一步
   - 如果 score 過低或位置明顯不合理，可以重新分割後再計算（最多 2 次）
4. **保存計劃**：確認無誤後使用 save_plan 保存執行計劃

## 注意事項
- 每次只能分割一個目標物件
- **segment_object 的 object_description 必須使用英文**（例如用戶說「紅色杯子」→ 傳入 "red cup"）
- compute_grasp 必須在成功 segment_object 之後調用
- 觀察每一步的返回圖像，自行判斷結果是否正確
- 用中文回覆用戶
"""


@dataclass
class AgentTurn:
    """一個對話回合的記錄"""
    role: str                    # "user" | "assistant" | "tool_result"
    content: list[dict]          # content blocks
    timestamp: str = ""
    token_usage: dict | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")


@dataclass
class AgentResult:
    """Agent 執行結果"""
    success: bool
    task_text: str
    final_message: str           # VLM 最後的回覆文字
    sam3_result: SAM3Result | None = None
    capture_result: CapturePointResult | None = None
    plan_path: str | None = None
    session_dir: str | None = None
    turns: list[AgentTurn] = field(default_factory=list)
    total_tokens: int = 0
    error_message: str = ""


class AgentLoop:
    """Tool-use agentic loop"""

    MAX_TURNS = 20  # 安全上限

    def __init__(
        self,
        vlm: VLMClient,
        tool_executor: ToolExecutor,
        system_prompt: str | None = None,
        on_turn: Callable[[AgentTurn], None] | None = None,
    ):
        self._vlm = vlm
        self._executor = tool_executor
        self._system = system_prompt or SYSTEM_PROMPT
        self._on_turn = on_turn
        self._messages: list[dict] = []
        self._turns: list[AgentTurn] = []
        self._cancelled = False
        self._total_tokens = 0

    def run(self, task_text: str,
            scene_image: np.ndarray | None = None) -> AgentResult:
        """執行 agentic loop（阻塞，在背景線程呼叫）"""
        self._cancelled = False
        self._messages = []
        self._turns = []
        self._total_tokens = 0

        # 構造初始 user message
        user_content = []
        if scene_image is not None:
            user_content.append(VLMClient.make_image_block(scene_image))
        user_content.append({"type": "text", "text": task_text})

        user_msg = {"role": "user", "content": user_content}
        self._messages.append(user_msg)
        self._record_turn(AgentTurn(role="user", content=user_content))

        # Agentic loop
        final_text = ""
        turn_count = 0

        while turn_count < self.MAX_TURNS and not self._cancelled:
            turn_count += 1
            log.info(f"Agent turn {turn_count}")

            try:
                response = self._vlm.create(
                    messages=self._messages,
                    system=self._system,
                    tools=TOOLS,
                    temperature=0.3,
                )
            except Exception as e:
                log.exception("VLM API error")
                return AgentResult(
                    success=False,
                    task_text=task_text,
                    final_message="",
                    error_message=f"VLM API error: {e}",
                    turns=self._turns,
                    total_tokens=self._total_tokens,
                )

            # 累計 token
            usage = response.get("usage", {})
            self._total_tokens += usage.get("input_tokens", 0)
            self._total_tokens += usage.get("output_tokens", 0)

            # 記錄 assistant turn
            content = response.get("content", [])
            stop_reason = response.get("stop_reason", "end_turn")

            assistant_turn = AgentTurn(
                role="assistant",
                content=content,
                token_usage=usage,
            )
            self._record_turn(assistant_turn)

            # 追加 assistant message 到歷史
            self._messages.append({
                "role": "assistant",
                "content": content,
            })

            # 提取最終文字
            for block in content:
                if block.get("type") == "text":
                    final_text = block["text"]

            # end_turn → 結束
            if stop_reason == "end_turn":
                break

            # tool_use → 執行工具
            if stop_reason == "tool_use":
                tool_results_content = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue

                    if self._cancelled:
                        break

                    tool_name = block["name"]
                    tool_input = block.get("input", {})
                    tool_id = block["id"]

                    log.info(f"Executing tool: {tool_name}({tool_input})")
                    result_blocks = self._executor.execute(tool_name, tool_input)

                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_blocks,
                    })

                if self._cancelled:
                    break

                # 追加 tool results 到歷史
                tool_result_msg = {
                    "role": "user",
                    "content": tool_results_content,
                }
                self._messages.append(tool_result_msg)

                # 記錄 tool result turn
                tool_turn = AgentTurn(
                    role="tool_result",
                    content=tool_results_content,
                )
                self._record_turn(tool_turn)

        # 構建結果
        if self._cancelled:
            return AgentResult(
                success=False,
                task_text=task_text,
                final_message=final_text,
                sam3_result=self._executor.sam3_result,
                capture_result=self._executor.capture_result,
                error_message="Cancelled by user",
                turns=self._turns,
                total_tokens=self._total_tokens,
            )

        if turn_count >= self.MAX_TURNS:
            return AgentResult(
                success=False,
                task_text=task_text,
                final_message=final_text,
                sam3_result=self._executor.sam3_result,
                capture_result=self._executor.capture_result,
                error_message=f"Max turns ({self.MAX_TURNS}) exceeded",
                turns=self._turns,
                total_tokens=self._total_tokens,
            )

        return AgentResult(
            success=True,
            task_text=task_text,
            final_message=final_text,
            sam3_result=self._executor.sam3_result,
            capture_result=self._executor.capture_result,
            plan_path=None,  # 從 executor 獲取
            session_dir=self._executor.session_dir,
            turns=self._turns,
            total_tokens=self._total_tokens,
        )

    def cancel(self):
        """設定取消 flag"""
        self._cancelled = True

    def _record_turn(self, turn: AgentTurn):
        self._turns.append(turn)
        if self._on_turn:
            self._on_turn(turn)

    @property
    def messages(self) -> list[dict]:
        return self._messages
