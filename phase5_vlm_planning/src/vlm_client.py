"""VLM Client 抽象接口 + 雙 Provider 實現

支持:
- OpenAIVLM: 通過 openai SDK 調用 GPT-4o（原生 tool-use）
- ClaudeCodeVLM: 通過 claude CLI subprocess 調用 Claude（結構化 prompt 模擬 tool-use）
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VLMClient(ABC):
    """抽象 VLM 接口 — 支持 tool-use"""

    @abstractmethod
    def create(self, messages: list[dict], system: str,
               tools: list[dict] | None = None,
               temperature: float = 0.3) -> dict:
        """發送對話 + tools，返回標準化回應

        Returns: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
            ],
            "stop_reason": "tool_use" | "end_turn",
            "usage": {"input_tokens": N, "output_tokens": N}
        }
        """

    @staticmethod
    def make_image_block(image_bgr: np.ndarray, quality: int = 85) -> dict:
        """BGR ndarray → base64 JPEG image block（通用格式）"""
        ok, buf = cv2.imencode(".jpg", image_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Failed to encode image to JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        }

    @staticmethod
    def make_tool_result(tool_use_id: str, content: list[dict]) -> dict:
        """構造 tool_result message block"""
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }],
        }


class OpenAIVLM(VLMClient):
    """OpenAI GPT-4o 實現 — 原生 tool-use 支持"""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 4096,
                 api_key: str | None = None):
        import openai
        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def create(self, messages: list[dict], system: str,
               tools: list[dict] | None = None,
               temperature: float = 0.3) -> dict:
        # 轉換 messages 格式 (Anthropic → OpenAI)
        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append(self._convert_message(msg))

        # 轉換 tools 格式
        oai_tools = None
        if tools:
            oai_tools = []
            for t in tools:
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                })

        # 調用 API，帶重試
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self._model,
                    "messages": oai_messages,
                    "max_tokens": self._max_tokens,
                    "temperature": temperature,
                }
                if oai_tools:
                    kwargs["tools"] = oai_tools
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt < 2 and ("rate" in str(e).lower() or "timeout" in str(e).lower()):
                    wait = 2 ** attempt
                    log.warning(f"OpenAI API error (attempt {attempt+1}): {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        # 轉換回應為標準格式
        choice = resp.choices[0]
        content_blocks = []
        stop_reason = "end_turn"

        if choice.message.content:
            content_blocks.append({
                "type": "text",
                "text": choice.message.content,
            })

        if choice.message.tool_calls:
            stop_reason = "tool_use"
            for tc in choice.message.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }

        return {
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": usage,
        }

    def _convert_message(self, msg: dict) -> dict:
        """標準格式 message → OpenAI 格式"""
        role = msg["role"]
        content = msg.get("content", [])

        if role == "assistant":
            # 拆分 text 和 tool_use
            text_parts = []
            tool_calls = []
            for block in content:
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })
            result = {"role": "assistant"}
            if text_parts:
                result["content"] = "\n".join(text_parts)
            else:
                result["content"] = None
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        elif role == "user":
            # 檢查是否包含 tool_result
            if content and isinstance(content, list) and content[0].get("type") == "tool_result":
                # OpenAI 用 role=tool
                results = []
                for block in content:
                    text_parts = []
                    for sub in block.get("content", []):
                        if sub["type"] == "text":
                            text_parts.append(sub["text"])
                        elif sub["type"] == "image":
                            # OpenAI tool results 不直接支持圖片
                            # 把圖片描述放進文字
                            text_parts.append("[圖片已返回，請根據之前的上下文繼續]")
                    results.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": "\n".join(text_parts) if text_parts else "Done",
                    })
                # OpenAI 需要一個個返回 tool results
                # 但我們只能返回一個 message，所以如果多個就需要特殊處理
                # 通常一次只有一個 tool call
                if len(results) == 1:
                    return results[0]
                # 多個 tool results — 返回第一個
                return results[0]

            # 一般 user message
            oai_content = []
            for block in content:
                if block["type"] == "text":
                    oai_content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    b64 = block["source"]["data"]
                    media_type = block["source"]["media_type"]
                    oai_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                            "detail": "high",
                        },
                    })
            return {"role": "user", "content": oai_content}

        return {"role": role, "content": str(content)}


class ClaudeCodeVLM(VLMClient):
    """Claude Code CLI subprocess 實現 — 通過訂閱制使用 Claude

    圖片傳遞方式：
    - 將圖片保存為臨時 jpg 文件
    - 在 prompt 中引用路徑，指示 Claude 用 Read 工具讀取
    - 使用 --tools "Read" 只允許 Read 工具
    - 使用 --add-dir 授權臨時目錄
    - --max-turns 3 允許 Claude 讀取多張圖片後回覆
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096):
        self._model = model
        self._max_tokens = max_tokens
        self._tmp_dir = tempfile.mkdtemp(prefix="vlm_agent_")

    def create(self, messages: list[dict], system: str,
               tools: list[dict] | None = None,
               temperature: float = 0.3) -> dict:
        # 收集本次需要的圖片文件
        image_paths = []

        # 構造 prompt
        prompt_parts = []
        prompt_parts.append(f"[SYSTEM]\n{system}")

        if tools:
            prompt_parts.append(self._format_tools(tools))

        # 只保存最後一條 user message 的圖片（最新的 tool result 或場景圖）
        # 歷史圖片用文字描述代替，大幅減少 Read 調用次數
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        for i, msg in enumerate(messages):
            save_images = (i >= last_user_idx)  # 只有最新 user msg 保存圖片
            prompt_parts.append(
                self._format_message(msg, image_paths, save_images)
            )

        # 末尾指示
        has_images = len(image_paths) > 0
        if has_images:
            prompt_parts.append(
                "\n[IMPORTANT] 上面提到的圖片文件請先用 Read 工具查看，"
                "然後根據圖片內容和文字信息回覆。"
                "如需調用工具，用上面定義的 JSON 格式回覆。"
            )
        else:
            prompt_parts.append(
                "\n[IMPORTANT] 請根據上面的信息回覆。"
                "如需調用工具，用上面定義的 JSON 格式回覆。"
            )

        full_prompt = "\n\n".join(prompt_parts)

        # 構造 CLI 命令
        cmd = [
            "claude", "-p", "--output-format", "json",
            "--model", self._model,
        ]

        if has_images:
            # 允許 Read 工具讀取圖片
            cmd.extend(["--tools", "Read"])
            cmd.extend(["--add-dir", self._tmp_dir])
            # 每張圖片需要 1 turn 讀取，再加 2 turn 餘裕回覆
            max_turns = len(image_paths) * 2 + 3
            cmd.extend(["--max-turns", str(min(max_turns, 10))])
        else:
            # 純文字模式，不需要任何工具
            cmd.extend(["--tools", ""])
            cmd.extend(["--max-turns", "1"])

        # 調用 claude CLI
        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return self._error_response("Claude CLI timeout (180s)")
        except FileNotFoundError:
            return self._error_response("claude CLI not found in PATH")

        if result.returncode != 0:
            stderr = result.stderr.strip()
            log.error(f"claude CLI error (rc={result.returncode}): {stderr[:300]}")
            if not result.stdout.strip():
                return self._error_response(f"CLI error: {stderr[:200]}")

        # 解析回應
        raw = result.stdout.strip()
        text = ""
        usage = {"input_tokens": 0, "output_tokens": 0}

        if raw:
            try:
                response = json.loads(raw)
                text = self._extract_text(response)
                if isinstance(response, dict):
                    resp_usage = response.get("usage", {})
                    usage["input_tokens"] = (
                        resp_usage.get("input_tokens", 0)
                        + resp_usage.get("cache_read_input_tokens", 0)
                        + resp_usage.get("cache_creation_input_tokens", 0)
                    )
                    usage["output_tokens"] = resp_usage.get("output_tokens", 0)
            except json.JSONDecodeError:
                text = raw

        if not text:
            return self._error_response("Empty response from Claude CLI")

        # 解析 tool calls
        content_blocks, stop_reason = self._parse_tool_calls(text, tools)

        return {
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": usage,
        }

    def _save_image(self, image_block: dict) -> str:
        """將 image block 保存為臨時文件，返回路徑"""
        b64 = image_block["source"]["data"]
        # 用 uuid 避免衝突
        fname = f"img_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(self._tmp_dir, fname)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return path

    def _format_tools(self, tools: list[dict]) -> str:
        lines = [
            "[AVAILABLE TOOLS]",
            "你可以使用以下工具。需要調用工具時，**只用以下 JSON 格式回覆**：",
            '```json',
            '{"action": "tool_call", "tool": "tool_name", "input": {"param": "value"}}',
            '```',
            "不需要調用工具時，直接用文字回覆（不要包含 JSON）。",
            "每次回覆只能調用一個工具。",
            "",
            "可用工具：",
        ]
        for i, t in enumerate(tools, 1):
            params = t.get("input_schema", {}).get("properties", {})
            required = t.get("input_schema", {}).get("required", [])
            param_desc = ""
            if params:
                param_strs = []
                for pname, pinfo in params.items():
                    req = " (必填)" if pname in required else " (選填)"
                    param_strs.append(
                        f"    - {pname}: {pinfo.get('description', '')}{req}"
                    )
                param_desc = "\n" + "\n".join(param_strs)
            lines.append(f"{i}. {t['name']} — {t['description']}{param_desc}")
        return "\n".join(lines)

    def _format_message(self, msg: dict,
                        image_paths: list[str],
                        save_images: bool = True) -> str:
        """格式化消息，將圖片保存到文件並引用路徑

        save_images: True=保存圖片供 Read，False=用文字描述替代（歷史消息）
        """
        role = msg["role"]
        content = msg.get("content", [])
        parts = []

        if role == "user":
            parts.append("[USER]")
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(f"[TOOL RESULT (id={block['tool_use_id']})]")
                    for sub in block.get("content", []):
                        if sub.get("type") == "text":
                            parts.append(sub["text"])
                        elif sub.get("type") == "image":
                            if save_images:
                                path = self._save_image(sub)
                                image_paths.append(path)
                                parts.append(
                                    f"[工具返回了標注圖片，請用 Read 工具查看: {path}]"
                                )
                            else:
                                parts.append("[工具返回了標注圖片（已在之前查看過）]")
                elif block.get("type") == "image":
                    if save_images:
                        path = self._save_image(block)
                        image_paths.append(path)
                        parts.append(
                            f"[場景圖片，請用 Read 工具查看: {path}]"
                        )
                    else:
                        parts.append("[場景圖片（已在之前查看過）]")
        elif role == "assistant":
            parts.append("[ASSISTANT]")
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    parts.append(
                        f'```json\n{{"action": "tool_call", "tool": "{block["name"]}", '
                        f'"input": {json.dumps(block["input"], ensure_ascii=False)}}}\n```'
                    )

        return "\n".join(parts)

    def _extract_text(self, response) -> str:
        """從 claude CLI JSON 回應中提取文字"""
        if isinstance(response, dict):
            # 標準格式: {"type": "result", "result": "..."}
            if "result" in response and response["result"]:
                return str(response["result"])
            # 有時 result 為空但有 content
            if "content" in response:
                parts = []
                for block in response["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])
                if parts:
                    return "\n".join(parts)
            # error_max_turns: Claude 用完 turns 但仍有 stop_reason=tool_use
            # 嘗試從 conversations 中提取最後的文字回覆
            if response.get("subtype") == "error_max_turns":
                log.warning("Claude CLI hit max turns — extracting last text")
                # 有時最後一個 turn 的文字在 result 為空，但對話中有
                return ""
            if response.get("is_error"):
                err_msg = response.get("subtype", "unknown error")
                log.warning(f"Claude CLI returned error: {err_msg}")
                return ""
        if isinstance(response, str):
            return response
        return ""

    def _parse_tool_calls(self, text: str,
                          tools: list[dict] | None) -> tuple[list[dict], str]:
        """從文字中解析 tool call JSON"""
        content_blocks = []
        tool_names = {t["name"] for t in tools} if tools else set()

        # 嘗試匹配 ```json ... ``` 中的 tool call
        json_pattern = r'```(?:json)?\s*(\{[^`]*?"action"\s*:\s*"tool_call"[^`]*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            # 移除 JSON block 後的文字作為 text content
            remaining = re.sub(json_pattern, "", text, flags=re.DOTALL).strip()
            if remaining:
                content_blocks.append({"type": "text", "text": remaining})

            for match in matches:
                try:
                    parsed = json.loads(match)
                    tool_name = parsed.get("tool", "")
                    if tool_name in tool_names:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": tool_name,
                            "input": parsed.get("input", {}),
                        })
                except json.JSONDecodeError:
                    continue

            if any(b["type"] == "tool_use" for b in content_blocks):
                return content_blocks, "tool_use"

        # 嘗試裸 JSON（不在 code block 中）— 支持嵌套 braces
        bare_pattern = r'\{[^{}]*"action"\s*:\s*"tool_call"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        bare_matches = re.findall(bare_pattern, text)
        if bare_matches:
            remaining = text
            for match in bare_matches:
                remaining = remaining.replace(match, "", 1)
            remaining = remaining.strip()
            if remaining:
                content_blocks.append({"type": "text", "text": remaining})

            for match in bare_matches:
                try:
                    parsed = json.loads(match)
                    tool_name = parsed.get("tool", "")
                    if tool_name in tool_names:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": tool_name,
                            "input": parsed.get("input", {}),
                        })
                except json.JSONDecodeError:
                    continue

            if any(b["type"] == "tool_use" for b in content_blocks):
                return content_blocks, "tool_use"

        # 沒有 tool call
        content_blocks = [{"type": "text", "text": text}]
        return content_blocks, "end_turn"

    @staticmethod
    def _error_response(msg: str) -> dict:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Error: {msg}"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


def create_vlm(provider: str = "openai", **kwargs) -> VLMClient:
    """工廠函數 — 根據 provider 名稱創建 VLM client"""
    if provider == "openai":
        return OpenAIVLM(**kwargs)
    elif provider == "claude_code":
        return ClaudeCodeVLM(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
