"""
app/agents/chat_agent.py
核心 Agent 实现 — ReAct（Reasoning + Acting）循环。

执行流程：
    用户消息 → LLM → [tool_use] → 执行工具 → [tool_result] → LLM → ...
                    ↑___________________________________|
    直到 LLM 返回 end_turn 或 stop_reason != tool_use

流式设计：
    - 逐字符 yield SSEEvent，前端实时展示
    - 工具调用前/后发送结构化事件，前端可渲染进度指示器
    - 内部使用 AsyncGenerator，上层 API 直接 iterate 即可

注意：
    - messages 在 Agent 内部维护完整历史，避免重复传递
    - 工具结果以 Anthropic tool_result 格式追加到历史
    - 超过 max_iterations 时优雅退出并告知用户
"""
import json
from typing import AsyncGenerator, List

import anthropic

from app.core.config import settings
from app.core.exceptions import LLMError, MaxIterationsError
from app.core.logging import get_logger
from app.models.schemas import (
    ChatMessage,
    ErrorData,
    MessageStopData,
    SSEEvent,
    SSEEventType,
    TextDeltaData,
    ToolUseEndData,
    ToolUseStartData,
)
from app.tools.base import tool_registry

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个专业的智能助手，可以帮助用户查询天气、订单状态、商品库存等信息。

工作原则：
1. 优先使用工具获取实时、准确的数据，不要凭记忆猜测
2. 工具调用失败时，诚实告知用户并说明原因
3. 回答简洁、专业，必要时提供数据来源说明
4. 不要在一次回复中过度调用工具，按需调用
"""


class ChatAgent:
    """
    单 Agent 实现。
    每次对话创建一个实例（session 级别），持有完整 messages 历史。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._tools = tool_registry.to_anthropic_tools()
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _convert_messages(self, messages: List[ChatMessage]) -> List[dict]:
        """将 Pydantic ChatMessage 列表转为 Anthropic API messages 格式。"""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def run_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        主入口：流式执行 Agent ReAct 循环。
        yield SSEEvent，上层 API 将其序列化为 SSE 帧。
        """
        # 转换为内部格式，后续追加 tool_result
        internal_messages = self._convert_messages(messages)
        iteration = 0

        while iteration < settings.llm_max_iterations:
            iteration += 1
            logger.info(
                "agent_iteration",
                session_id=self.session_id,
                iteration=iteration,
            )

            # ── 调用 LLM（流式） ──────────────────────────────────────────
            collected_blocks = []   # 收集本轮所有 content block
            current_text = ""
            current_tool_uses = []  # [(tool_use_id, tool_name, input_str)]

            try:
                async with self._client.messages.stream(
                    model=settings.llm_model,
                    max_tokens=settings.llm_max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=self._tools,
                    messages=internal_messages,
                ) as stream:

                    # ── 流式处理 ─────────────────────────────────────────
                    async for event in stream:
                        # 文本 delta
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                delta_text = event.delta.text
                                current_text += delta_text
                                yield SSEEvent(
                                    type=SSEEventType.TEXT_DELTA,
                                    data=TextDeltaData(text=delta_text).model_dump(),
                                )
                            elif hasattr(event.delta, "partial_json"):
                                # 工具调用的 JSON 输入正在流式生成，静默收集
                                pass

                        elif event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                current_tool_uses.append({
                                    "id": block.id,
                                    "name": block.name,
                                    "input": "",   # 后续通过 delta 拼接
                                })

                        elif event.type == "content_block_stop":
                            pass  # 无需处理

                    # ── 获取最终完整消息 ──────────────────────────────────
                    final_message = await stream.get_final_message()
                    self._total_input_tokens += final_message.usage.input_tokens
                    self._total_output_tokens += final_message.usage.output_tokens

                    stop_reason = final_message.stop_reason
                    content_blocks = final_message.content

            except anthropic.APIStatusError as e:
                logger.error("llm_api_error", status=e.status_code, message=str(e))
                yield SSEEvent(
                    type=SSEEventType.ERROR,
                    data=ErrorData(code="LLM_ERROR", message=f"LLM API 错误：{e.message}").model_dump(),
                )
                return
            except anthropic.APIConnectionError as e:
                yield SSEEvent(
                    type=SSEEventType.ERROR,
                    data=ErrorData(code="LLM_CONNECTION_ERROR", message="LLM 连接失败，请稍后重试").model_dump(),
                )
                return

            # ── stop_reason == "end_turn"：LLM 完成回答 ──────────────────
            if stop_reason == "end_turn":
                # 将 assistant 消息追加到历史（保持对话连续性）
                internal_messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() for b in content_blocks],
                })
                yield SSEEvent(
                    type=SSEEventType.MESSAGE_STOP,
                    data=MessageStopData(
                        session_id=self.session_id,
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                    ).model_dump(),
                )
                return

            # ── stop_reason == "tool_use"：需要执行工具 ───────────────────
            if stop_reason == "tool_use":
                # 将 assistant 的工具调用追加到 messages
                internal_messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() for b in content_blocks],
                })

                # 构造 tool_result 列表
                tool_results = []
                for block in content_blocks:
                    if block.type != "tool_use":
                        continue

                    tool_id = block.id
                    tool_name = block.name
                    tool_input = block.input  # 已是 dict

                    # 通知前端工具开始
                    yield SSEEvent(
                        type=SSEEventType.TOOL_USE_START,
                        data=ToolUseStartData(
                            tool_name=tool_name,
                            tool_id=tool_id,
                            input=tool_input,
                        ).model_dump(),
                    )

                    # 执行工具
                    result = await tool_registry.execute(tool_name, tool_id, tool_input)

                    # 通知前端工具结束
                    yield SSEEvent(
                        type=SSEEventType.TOOL_USE_END,
                        data=ToolUseEndData(
                            tool_name=tool_name,
                            tool_id=tool_id,
                            output=result.content,
                            is_error=result.is_error,
                        ).model_dump(),
                    )

                    # 准备 tool_result block（Anthropic 格式）
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result.to_str(),
                        "is_error": result.is_error,
                    })

                # 将所有工具结果作为 user 消息追加（Anthropic 规范）
                internal_messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                # 继续下一轮循环
                continue

            # ── 其他 stop_reason（max_tokens 等）────────────────────────
            logger.warning("unexpected_stop_reason", stop_reason=stop_reason)
            yield SSEEvent(
                type=SSEEventType.MESSAGE_STOP,
                data=MessageStopData(
                    session_id=self.session_id,
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                ).model_dump(),
            )
            return

        # ── 超过最大迭代次数 ──────────────────────────────────────────────
        logger.error(
            "max_iterations_exceeded",
            session_id=self.session_id,
            max=settings.llm_max_iterations,
        )
        yield SSEEvent(
            type=SSEEventType.ERROR,
            data=ErrorData(
                code="MAX_ITERATIONS_EXCEEDED",
                message=f"Agent 超过最大迭代次数 ({settings.llm_max_iterations})，请简化请求后重试。",
            ).model_dump(),
        )
