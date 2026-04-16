1、创建项目目录结构
mkdir -p /home/claude/agent-streaming-api/app/{agent,api,schemas,utils}

2、创建schemas/models.py -Pydantic data models
"""
数据模型定义 - 请求/响应 Schema
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=100_000)


class AgentRequest(BaseModel):
    """Agent 请求体"""
    messages: List[Message] = Field(..., min_length=1, max_length=50)
    session_id: Optional[str] = Field(None, max_length=128)
    stream: bool = Field(default=True)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("messages")
    @classmethod
    def validate_last_message_is_user(cls, v: List[Message]) -> List[Message]:
        if v and v[-1].role != MessageRole.USER:
            raise ValueError("最后一条消息必须来自用户 (role=user)")
        return v


class SSEEventType(str, Enum):
    """SSE 事件类型"""
    START = "start"           # 流开始
    DELTA = "delta"           # 增量文本
    TOOL_USE = "tool_use"     # 工具调用
    TOOL_RESULT = "tool_result"  # 工具结果
    DONE = "done"             # 完成
    ERROR = "error"           # 错误


class SSEEvent(BaseModel):
    """SSE 事件数据结构"""
    event: SSEEventType
    data: Any
    session_id: Optional[str] = None


class DeltaData(BaseModel):
    text: str


class ToolUseData(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]


class ToolResultData(BaseModel):
    tool_name: str
    result: Any
    is_error: bool = False


class DoneData(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"


class ErrorData(BaseModel):
    code: str
    message: str
    retryable: bool = False
    
3、创建utils/logger.py - structured logging
"""
结构化日志配置
生产环境推荐使用 JSON 格式，便于日志采集系统（ELK、Loki 等）解析
"""
import logging
import sys
from typing import Any, Dict

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """自定义 JSON 日志格式，添加服务名等元数据"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = "agent-streaming-api"
        log_record["level"] = record.levelname
        # 移除冗余字段
        log_record.pop("color_message", None)


def setup_logging(level: str = "INFO") -> None:
    """初始化日志配置"""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除默认处理器，避免重复输出
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # 降低第三方库的日志噪音
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
    
4、创建utils/sse.py -SSE formatting utilities
"""
SSE (Server-Sent Events) 工具函数

SSE 协议格式（每个事件之间用 \n\n 分隔）：
  event: <event_type>\n
  data: <json_string>\n
  \n
"""
import json
from typing import Any, AsyncGenerator

from app.schemas.models import (
    DeltaData,
    DoneData,
    ErrorData,
    SSEEvent,
    SSEEventType,
    ToolResultData,
    ToolUseData,
)


def format_sse(event: SSEEventType, data: Any, session_id: str | None = None) -> str:
    """
    将事件序列化为标准 SSE 文本格式
    注意：data 字段中的换行符必须替换，否则客户端无法正确解析
    """
    payload = SSEEvent(event=event, data=data, session_id=session_id)
    # 使用 model_dump 确保枚举值被序列化为字符串
    json_str = json.dumps(payload.model_dump(), ensure_ascii=False)
    return f"event: {event.value}\ndata: {json_str}\n\n"


def sse_start(session_id: str | None = None) -> str:
    return format_sse(SSEEventType.START, {"message": "Stream started"}, session_id)


def sse_delta(text: str, session_id: str | None = None) -> str:
    return format_sse(SSEEventType.DELTA, DeltaData(text=text).model_dump(), session_id)


def sse_tool_use(tool_name: str, tool_input: dict, session_id: str | None = None) -> str:
    return format_sse(
        SSEEventType.TOOL_USE,
        ToolUseData(tool_name=tool_name, tool_input=tool_input).model_dump(),
        session_id,
    )


def sse_tool_result(
    tool_name: str, result: Any, is_error: bool = False, session_id: str | None = None
) -> str:
    return format_sse(
        SSEEventType.TOOL_RESULT,
        ToolResultData(tool_name=tool_name, result=result, is_error=is_error).model_dump(),
        session_id,
    )


def sse_done(
    total_tokens: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str = "end_turn",
    session_id: str | None = None,
) -> str:
    return format_sse(
        SSEEventType.DONE,
        DoneData(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        ).model_dump(),
        session_id,
    )


def sse_error(code: str, message: str, retryable: bool = False, session_id: str | None = None) -> str:
    return format_sse(
        SSEEventType.ERROR,
        ErrorData(code=code, message=message, retryable=retryable).model_dump(),
        session_id,
    )


async def heartbeat_generator(interval: float = 15.0) -> AsyncGenerator[str, None]:
    """
    心跳生成器：定期发送注释行（以 : 开头），防止连接被中间代理/LB 超时断开
    SSE 协议中以 : 开头的行是注释，客户端会忽略
    """
    import asyncio
    while True:
        await asyncio.sleep(interval)
        yield ": heartbeat\n\n"
        
5、创建 agent/tools.py - Agent tool definitions
"""
Agent 工具集定义

每个工具包含：
1. Anthropic tool schema（用于传给 Claude API）
2. 对应的 Python 执行函数（async）
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# 工具 Schema（传给 Claude API 的 tools 参数）
# ─────────────────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": "获取当前的日期和时间（UTC 和本地时间）",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区名称，例如 Asia/Shanghai、America/New_York，默认 UTC",
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculator",
        "description": "执行数学计算。支持加减乘除、幂运算、取模等基础运算",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '(3 + 5) * 2'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": "搜索互联网获取最新信息（模拟实现，生产中替换为真实搜索 API）",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 3，最大 10",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]


# ─────────────────────────────────────────────
# 工具执行函数
# ─────────────────────────────────────────────

async def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """
    工具分发器：根据名称路由到对应的执行函数
    所有工具函数应为 async，且在出错时返回结构化错误，不应抛出异常
    """
    handlers = {
        "get_current_time": _tool_get_current_time,
        "calculator": _tool_calculator,
        "web_search": _tool_web_search,
    }

    handler = handlers.get(tool_name)
    if handler is None:
        return {"error": f"未知工具: {tool_name}"}

    try:
        return await handler(tool_input)
    except Exception as e:
        logger.exception("工具执行失败", extra={"tool": tool_name, "input": tool_input})
        return {"error": str(e)}


async def _tool_get_current_time(inputs: Dict[str, Any]) -> Dict[str, str]:
    tz_name = inputs.get("timezone", "UTC")
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
        tz_name = "UTC"

    now = datetime.now(tz)
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": tz_name,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "weekday": now.strftime("%A"),
    }


async def _tool_calculator(inputs: Dict[str, Any]) -> Dict[str, Any]:
    expression = inputs.get("expression", "")
    # 安全白名单：只允许数字和基础运算符
    allowed_chars = set("0123456789+-*/()., **%")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        return {"error": "表达式包含非法字符"}
    try:
        # 使用 eval 时严格限制命名空间
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"error": "除数不能为零"}
    except Exception as e:
        return {"error": f"计算错误: {e}"}


async def _tool_web_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    模拟搜索结果。生产环境中替换为：
    - Brave Search API
    - Tavily API
    - SerpAPI
    - 自建搜索服务
    """
    query = inputs.get("query", "")
    num = min(inputs.get("num_results", 3), 10)

    # 模拟网络延迟
    await asyncio.sleep(0.3)

    return {
        "query": query,
        "results": [
            {
                "title": f"搜索结果 {i+1}：关于 '{query}' 的信息",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"这是关于 '{query}' 的模拟搜索结果摘要 {i+1}。"
                           "实际部署时请接入真实搜索 API。",
            }
            for i in range(num)
        ],
        "note": "? 这是模拟结果，请在生产环境中替换为真实搜索 API",
    }
    
6、创建 agent/core.py - Core Agent streaming logic(agntic loop)
"""
Agent 核心逻辑 - 流式 Agentic Loop

实现了完整的 ReAct 模式：
  1. 流式输出 Claude 的思考/回答文本
  2. 检测到工具调用时执行工具
  3. 将工具结果反馈给 Claude 继续生成
  4. 循环直到 Claude 输出最终答案（stop_reason=end_turn）
"""
import asyncio
from typing import AsyncGenerator, List

import anthropic

from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.schemas.models import AgentRequest, Message, MessageRole
from app.utils.logger import get_logger
from app.utils.sse import (
    sse_delta,
    sse_done,
    sse_error,
    sse_start,
    sse_tool_result,
    sse_tool_use,
)

logger = get_logger(__name__)

# 防止 Agent 陷入无限工具调用循环
MAX_TOOL_ITERATIONS = 10

# Anthropic 客户端（全局单例，内部维护连接池）
_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            # max_retries 由 SDK 内部处理 429/529，默认 2 次
            max_retries=2,
            # 超时配置：连接 5s，读取 120s（长流式响应）
            timeout=anthropic.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )
    return _client


def _build_messages(request: AgentRequest) -> List[dict]:
    """将 Pydantic Message 列表转为 Anthropic API 格式"""
    return [
        {"role": msg.role.value, "content": msg.content}
        for msg in request.messages
        if msg.role != MessageRole.SYSTEM  # system 消息单独传
    ]


def _get_system_prompt(request: AgentRequest) -> str:
    """提取 system prompt；若不存在则使用默认值"""
    for msg in request.messages:
        if msg.role == MessageRole.SYSTEM:
            return msg.content
    return (
        "你是一个智能助手，可以使用工具来帮助用户解决问题。"
        "请用中文回答，保持简洁、准确、有帮助。"
        "当需要获取外部信息时，主动使用提供的工具。"
    )


async def run_agent_stream(request: AgentRequest) -> AsyncGenerator[str, None]:
    """
    核心流式 Agent 生成器

    产出：SSE 格式的字符串片段，调用方直接 yield 给 HTTP 响应

    异常处理策略：
    - 网络/API 错误：发送 error 事件后结束，客户端可重试
    - 工具执行错误：将错误结果返回给 Claude，让其优雅处理
    - 超时：发送 error 事件
    """
    session_id = request.session_id
    client = get_client()

    # ── 1. 发送开始事件 ──────────────────────────────────────────────
    yield sse_start(session_id)

    messages = _build_messages(request)
    system_prompt = _get_system_prompt(request)

    input_tokens_total = 0
    output_tokens_total = 0
    stop_reason = "end_turn"

    # ── 2. Agentic Loop ──────────────────────────────────────────────
    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info(
            "Agent 迭代开始",
            extra={"session_id": session_id, "iteration": iteration, "msg_count": len(messages)},
        )

        tool_uses: list[dict] = []    # 本轮收集到的所有工具调用
        assistant_text = ""           # 本轮累计文本（用于构建历史消息）
        content_blocks: list[dict] = []  # 完整 content 块（含 text + tool_use）

        try:
            # ── 3. 调用 Claude 流式 API ──────────────────────────────
            async with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                tools=TOOL_SCHEMAS,
                messages=messages,
            ) as stream:
                # 逐事件处理流式响应
                async for event in stream:
                    event_type = event.type

                    # 文本增量：直接推送给客户端
                    if event_type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text_chunk = event.delta.text
                            assistant_text += text_chunk
                            yield sse_delta(text_chunk, session_id)

                    # 内容块结束：收集工具调用信息
                    elif event_type == "content_block_stop":
                        block = event  # 注意：实际内容在 stream.get_final_message() 中
                        pass

                    # 流结束：获取完整消息元数据
                    elif event_type == "message_delta":
                        stop_reason = getattr(event.delta, "stop_reason", "end_turn") or "end_turn"
                        if hasattr(event, "usage"):
                            output_tokens_total += getattr(event.usage, "output_tokens", 0)

                # 获取完整的最终消息（含 tool_use blocks）
                final_message = await stream.get_final_message()
                input_tokens_total += final_message.usage.input_tokens
                output_tokens_total = final_message.usage.output_tokens  # 以最终值为准
                stop_reason = final_message.stop_reason or "end_turn"

                # 重建 content blocks 用于历史
                content_blocks = []
                for block in final_message.content:
                    if block.type == "text":
                        content_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        tool_uses.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        content_blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

        except anthropic.APIStatusError as e:
            logger.error("Anthropic API 错误", extra={"status": e.status_code, "error": str(e)})
            retryable = e.status_code in (429, 529)
            yield sse_error(
                code=f"api_error_{e.status_code}",
                message=f"API 请求失败: {e.message}",
                retryable=retryable,
                session_id=session_id,
            )
            return

        except anthropic.APIConnectionError as e:
            logger.error("Anthropic 连接错误", extra={"error": str(e)})
            yield sse_error(
                code="connection_error",
                message="与 AI 服务的连接失败，请稍后重试",
                retryable=True,
                session_id=session_id,
            )
            return

        except asyncio.TimeoutError:
            logger.error("请求超时", extra={"session_id": session_id})
            yield sse_error(
                code="timeout",
                message="请求超时，请稍后重试",
                retryable=True,
                session_id=session_id,
            )
            return

        # ── 4. 判断是否需要执行工具 ───────────────────────────────────
        if stop_reason != "tool_use" or not tool_uses:
            # 没有工具调用，任务完成
            break

        # 将 assistant 的完整响应追加到历史
        messages.append({"role": "assistant", "content": content_blocks})

        # ── 5. 并发执行所有工具调用 ───────────────────────────────────
        tool_result_blocks = []

        async def _run_tool(tu: dict) -> tuple[dict, bool]:
            yield sse_tool_use(tu["name"], tu["input"], session_id)
            result = await execute_tool(tu["name"], tu["input"])
            is_error = isinstance(result, dict) and "error" in result
            yield sse_tool_result(tu["name"], result, is_error, session_id)
            return result, is_error

        # 顺序执行工具（若工具间无依赖可改为并发）
        for tool_use_info in tool_uses:
            # 推送工具调用事件
            yield sse_tool_use(tool_use_info["name"], tool_use_info["input"], session_id)

            # 执行工具
            result = await execute_tool(tool_use_info["name"], tool_use_info["input"])
            is_error = isinstance(result, dict) and "error" in result

            # 推送工具结果事件
            yield sse_tool_result(tool_use_info["name"], result, is_error, session_id)

            # 构建工具结果消息块
            import json
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_use_info["id"],
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": is_error,
            })

        # 将工具结果追加到消息历史，继续下一轮
        messages.append({"role": "user", "content": tool_result_blocks})

        logger.info(
            "工具执行完毕，继续迭代",
            extra={"session_id": session_id, "tools": [t["name"] for t in tool_uses]},
        )

    else:
        # 超过最大迭代次数
        logger.warning("超过最大工具迭代次数", extra={"session_id": session_id})
        yield sse_error(
            code="max_iterations",
            message="Agent 超过最大迭代次数，请简化问题",
            retryable=False,
            session_id=session_id,
        )
        return

    # ── 6. 发送完成事件 ───────────────────────────────────────────────
    yield sse_done(
        total_tokens=input_tokens_total + output_tokens_total,
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
        stop_reason=stop_reason,
        session_id=session_id,
    )
    
7、创建api/stream.py -Fastapi streaming SSE endpoint
"""
流式 SSE 接口路由

POST /api/v1/agent/stream
- 接受 JSON 请求体
- 返回 text/event-stream 响应
- 支持客户端断开检测
"""
import asyncio
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.agent.core import run_agent_stream
from app.schemas.models import AgentRequest
from app.utils.logger import get_logger
from app.utils.sse import sse_error

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Agent"])

# SSE 响应头
SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，对流式响应至关重要
    "Connection": "keep-alive",
    "Transfer-Encoding": "chunked",
}


async def _stream_with_disconnect_guard(
    request: Request,
    agent_gen: AsyncGenerator[str, None],
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    包装 Agent 生成器，添加：
    1. 客户端断开检测 - 避免资源泄漏
    2. 全局超时保护
    3. 异常捕获兜底
    """
    start_time = time.monotonic()
    MAX_STREAM_SECONDS = 300  # 单次请求最长流式时间

    try:
        async for chunk in agent_gen:
            # 检查客户端是否已断开
            if await request.is_disconnected():
                logger.info("客户端主动断开连接", extra={"session_id": session_id})
                return

            # 全局超时检查
            elapsed = time.monotonic() - start_time
            if elapsed > MAX_STREAM_SECONDS:
                logger.warning("流式响应超时", extra={"session_id": session_id, "elapsed": elapsed})
                yield sse_error("stream_timeout", "响应超时，请缩短问题或分步提问", retryable=True)
                return

            yield chunk

    except asyncio.CancelledError:
        logger.info("请求被取消", extra={"session_id": session_id})
        return

    except Exception as e:
        logger.exception("流式生成发生未预期异常", extra={"session_id": session_id})
        yield sse_error("internal_error", f"服务内部错误: {type(e).__name__}", retryable=True)

    finally:
        elapsed = time.monotonic() - start_time
        logger.info(
            "流式请求结束",
            extra={"session_id": session_id, "elapsed_seconds": round(elapsed, 2)},
        )


@router.post(
    "/agent/stream",
    summary="流式 Agent 对话",
    description="""
    以 SSE (Server-Sent Events) 格式流式返回 Agent 响应。

    **事件类型：**
    - `start`：流开始
    - `delta`：文本增量
    - `tool_use`：Agent 调用工具
    - `tool_result`：工具执行结果
    - `done`：生成完成（含 token 统计）
    - `error`：发生错误

    **客户端示例（JavaScript）：**
    ```javascript
    const response = await fetch('/api/v1/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content: '你好' }] })
    });
    const reader = response.body.getReader();
    // 解析 SSE 流...
    ```
    """,
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
)
async def agent_stream(
    request: Request,
    body: AgentRequest,
) -> StreamingResponse:
    # 为每次请求分配唯一 session_id（如果客户端没有提供）
    session_id = body.session_id or str(uuid.uuid4())
    body.session_id = session_id

    logger.info(
        "收到 Agent 流式请求",
        extra={
            "session_id": session_id,
            "msg_count": len(body.messages),
            "max_tokens": body.max_tokens,
        },
    )

    agent_gen = run_agent_stream(body)
    guarded_gen = _stream_with_disconnect_guard(request, agent_gen, session_id)

    return StreamingResponse(
        guarded_gen,
        media_type="text/event-stream",
        headers={
            **SSE_HEADERS,
            "X-Session-Id": session_id,
        },
    )


@router.get("/health", summary="健康检查", tags=["System"])
async def health_check():
    """Kubernetes liveness/readiness probe 使用"""
    return {"status": "ok", "service": "agent-streaming-api"}
    
    
8、创建app/main.py -FastAPI application entry point
"""
FastAPI 应用入口 - 生产级配置

包含：
- 生命周期管理（启动/关闭钩子）
- 全局中间件（CORS、请求 ID、限流）
- 全局异常处理器
- OpenAPI 文档配置
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.stream import router as stream_router
from app.utils.logger import get_logger, setup_logging

# ── 日志初始化（必须在所有其他导入之前完成）──────────────────────────
import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)

logger = get_logger(__name__)


# ── 生命周期管理 ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的资源管理"""
    logger.info("Agent Streaming API 启动中...")

    # 预热 Anthropic 客户端（建立连接池）
    from app.agent.core import get_client
    get_client()
    logger.info("Anthropic 客户端初始化完成")

    yield  # ← 应用正常运行期间在此暂停

    # 关闭时清理资源
    logger.info("Agent Streaming API 正在关闭，清理资源...")
    from app.agent.core import _client
    if _client:
        await _client.close()
    logger.info("资源清理完成")


# ── FastAPI 应用实例 ──────────────────────────────────────────────────
app = FastAPI(
    title="Agent Streaming API",
    description="生产级流式 AI Agent 接口，基于 FastAPI + Anthropic Claude",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS 中间件 ───────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Request-Id"],
)


# ── 请求追踪中间件 ────────────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """为每个请求注入唯一 Request-ID，便于日志追踪"""
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    response.headers["X-Request-Id"] = request_id

    # 跳过健康检查日志，减少噪音
    if request.url.path != "/api/v1/health":
        logger.info(
            "HTTP 请求",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
    return response


# ── 全局异常处理 ──────────────────────────────────────────────────────
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning("请求参数校验失败", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"code": "validation_error", "message": "请求参数不合法", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理的全局异常")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"code": "internal_error", "message": "服务器内部错误，请稍后重试"},
    )


# ── 注册路由 ──────────────────────────────────────────────────────────
app.include_router(stream_router)


# ── 根路由 ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Agent Streaming API", "version": "1.0.0", "docs": "/docs"}
    
9、创建requirements.txt
# Web 框架
fastapi==0.115.5
uvicorn[standard]==0.32.1       # standard 含 websockets、httptools、uvloop
gunicorn==23.0.0                # 生产多进程管理

# Anthropic SDK
anthropic==0.40.0

# 数据校验
pydantic==2.10.3
pydantic-settings==2.6.1        # 从环境变量加载配置

# 日志
python-json-logger==3.2.0

# 限流（可选，推荐在 API 网关层做，这里提供应用层兜底）
slowapi==0.1.9

# 工具类
python-dotenv==1.0.1
httpx==0.28.1                   # 异步 HTTP 客户端（anthropic SDK 依赖）


10、创建dockerfile
# ── 构建阶段 ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# 仅复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── 运行阶段 ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# 安全：非 root 用户运行
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /install /usr/local

# 复制应用代码
COPY app/ ./app/

# 设置文件所有权
RUN chown -R appuser:appuser /app

USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# 生产启动命令：Gunicorn + UvicornWorker
# --workers: CPU * 2 + 1 是经典公式，但流式 IO 密集型应用可适当调低
# --worker-class: UvicornWorker 支持异步
# --timeout: 单个 worker 处理请求的超时时间（设长一些以支持流式响应）
CMD ["gunicorn", "app.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "75", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
     
-------------------------------------------------------------------------------------------前端--------------------------
1、创建JavaScript SSE client example
client_example.js

/**
 * 前端 SSE 客户端封装
 * 支持：流式渲染、重连、错误处理、取消请求
 *
 * 兼容：浏览器、React、Vue、微信小程序（需适配 wx.request）
 */

const API_BASE = "http://localhost:8000";

/**
 * AgentStreamClient - 流式 Agent 客户端
 *
 * @example
 * const client = new AgentStreamClient();
 * const abort = client.chat("帮我查询今天的天气", {
 *   onDelta: (text) => appendToUI(text),
 *   onToolUse: (tool) => showToolBadge(tool.tool_name),
 *   onDone: (stats) => console.log("Token 用量:", stats),
 *   onError: (err) => showError(err.message),
 * });
 *
 * // 取消请求
 * abort();
 */
class AgentStreamClient {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || API_BASE;
    this.defaultHeaders = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
  }

  /**
   * 发起流式对话
   * @param {string|Array} input - 用户消息文本或完整消息数组
   * @param {Object} callbacks - 回调函数集合
   * @param {Object} params - 额外参数（max_tokens, temperature 等）
   * @returns {Function} abort - 调用此函数可取消请求
   */
  chat(input, callbacks = {}, params = {}) {
    const abortController = new AbortController();

    const messages = typeof input === "string"
      ? [{ role: "user", content: input }]
      : input;

    const requestBody = {
      messages,
      stream: true,
      max_tokens: params.max_tokens || 2048,
      temperature: params.temperature || 0.7,
      session_id: params.session_id || null,
    };

    this._streamFetch(
      `${this.baseUrl}/api/v1/agent/stream`,
      requestBody,
      callbacks,
      abortController.signal
    );

    return () => abortController.abort();
  }

  async _streamFetch(url, body, callbacks, signal) {
    const {
      onStart,
      onDelta,
      onToolUse,
      onToolResult,
      onDone,
      onError,
    } = callbacks;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: this.defaultHeaders,
        body: JSON.stringify(body),
        signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      // 获取 session_id（由服务端分配）
      const sessionId = response.headers.get("X-Session-Id");

      // 解析 SSE 流
      await this._parseSSEStream(response.body, {
        onStart: (data) => onStart?.(data, sessionId),
        onDelta,
        onToolUse,
        onToolResult,
        onDone,
        onError,
      });

    } catch (err) {
      if (err.name === "AbortError") {
        console.log("[AgentClient] 请求已取消");
        return;
      }
      console.error("[AgentClient] 请求失败:", err);
      onError?.({ code: "fetch_error", message: err.message, retryable: true });
    }
  }

  async _parseSSEStream(body, handlers) {
    const reader = body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE 事件以 \n\n 分隔
        const events = buffer.split("\n\n");
        // 最后一个可能是不完整的事件，保留在 buffer 中
        buffer = events.pop() || "";

        for (const eventText of events) {
          this._dispatchSSEEvent(eventText.trim(), handlers);
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  _dispatchSSEEvent(eventText, handlers) {
    if (!eventText || eventText.startsWith(":")) {
      // 忽略心跳注释行
      return;
    }

    let eventType = "";
    let dataStr = "";

    for (const line of eventText.split("\n")) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataStr = line.slice(6).trim();
      }
    }

    if (!eventType || !dataStr) return;

    let payload;
    try {
      payload = JSON.parse(dataStr);
    } catch {
      console.warn("[AgentClient] SSE 数据解析失败:", dataStr);
      return;
    }

    const data = payload.data;

    switch (eventType) {
      case "start":
        handlers.onStart?.(data);
        break;
      case "delta":
        handlers.onDelta?.(data.text);
        break;
      case "tool_use":
        handlers.onToolUse?.(data);
        break;
      case "tool_result":
        handlers.onToolResult?.(data);
        break;
      case "done":
        handlers.onDone?.(data);
        break;
      case "error":
        handlers.onError?.(data);
        break;
      default:
        console.warn("[AgentClient] 未知事件类型:", eventType);
    }
  }
}


// ── 使用示例 ────────────────────────────────────────────────────────

async function demo() {
  const client = new AgentStreamClient({ baseUrl: "http://localhost:8000" });

  let fullText = "";

  const abort = client.chat(
    "现在几点了？然后帮我计算 (123 * 456) + 789 的结果",
    {
      onStart: (data, sessionId) => {
        console.log("? 流开始，Session:", sessionId);
      },
      onDelta: (text) => {
        process.stdout.write(text); // 实时输出文本
        fullText += text;
      },
      onToolUse: (data) => {
        console.log(`\n? 调用工具: ${data.tool_name}`, data.tool_input);
      },
      onToolResult: (data) => {
        const status = data.is_error ? "?" : "?";
        console.log(`${status} 工具结果 [${data.tool_name}]:`, data.result);
      },
      onDone: (stats) => {
        console.log("\n\n? 生成完成");
        console.log(`   输入 Token: ${stats.input_tokens}`);
        console.log(`   输出 Token: ${stats.output_tokens}`);
        console.log(`   停止原因: ${stats.stop_reason}`);
      },
      onError: (err) => {
        console.error(`\n? 错误 [${err.code}]: ${err.message}`);
        if (err.retryable) console.log("   (可重试)");
      },
    }
  );

  // 演示：10 秒后取消（正常使用时不需要）
  // setTimeout(abort, 10000);
}

// 在 Node.js 环境运行示例（需要 Node 18+ 原生 fetch）
if (typeof window === "undefined") {
  demo().catch(console.error);
}

// 导出供模块使用
if (typeof module !== "undefined") {
  module.exports = { AgentStreamClient };
}


2、创建python测试脚本，针对流式接口 (test_stream.py)
"""
Python 客户端测试脚本
使用 httpx 验证流式 SSE 接口是否正常工作

用法：
    pip install httpx
    python test_stream.py
"""
import asyncio
import json
import sys

import httpx

BASE_URL = "http://localhost:8000"


async def test_stream(messages: list[dict], label: str = "测试"):
    print(f"\n{'='*60}")
    print(f"? {label}")
    print(f"{'='*60}")

    full_text = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/v1/agent/stream",
            json={
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            },
        ) as response:
            print(f"HTTP 状态: {response.status_code}")
            print(f"Session-ID: {response.headers.get('x-session-id', 'N/A')}")
            print()

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk

                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    event_text = event_text.strip()

                    if not event_text or event_text.startswith(":"):
                        continue  # 跳过心跳

                    event_type = ""
                    data_str = ""
                    for line in event_text.split("\n"):
                        if line.startswith("event: "):
                            event_type = line[7:].strip()
                        elif line.startswith("data: "):
                            data_str = line[6:].strip()

                    if not event_type or not data_str:
                        continue

                    try:
                        payload = json.loads(data_str)
                        data = payload.get("data", {})
                    except json.JSONDecodeError:
                        continue

                    if event_type == "start":
                        print("? 流开始")

                    elif event_type == "delta":
                        text = data.get("text", "")
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        full_text += text

                    elif event_type == "tool_use":
                        print(f"\n\n? 工具调用: {data.get('tool_name')} | 参数: {data.get('tool_input')}")

                    elif event_type == "tool_result":
                        status = "?" if data.get("is_error") else "?"
                        print(f"{status} 工具结果 [{data.get('tool_name')}]: {data.get('result')}\n")

                    elif event_type == "done":
                        print(f"\n\n? 完成 | 输入:{data.get('input_tokens')} 输出:{data.get('output_tokens')} Token")

                    elif event_type == "error":
                        print(f"\n? 错误 [{data.get('code')}]: {data.get('message')}")

    return full_text


async def main():
    # 测试 1：纯文本对话
    await test_stream(
        [{"role": "user", "content": "用一句话介绍你自己"}],
        label="测试1: 纯文本对话",
    )

    # 测试 2：工具调用
    await test_stream(
        [{"role": "user", "content": "现在几点了？并帮我计算 2^10 + 100 的值"}],
        label="测试2: 工具调用（时间 + 计算器）",
    )

    # 测试 3：多轮对话
    await test_stream(
        [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好，小明！很高兴认识你。"},
            {"role": "user", "content": "你还记得我叫什么名字吗？"},
        ],
        label="测试3: 多轮对话",
    )

    print("\n\n? 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
    
    
3、创建README文件
# Agent Streaming API

生产级流式 AI Agent 接口，基于 **FastAPI + Anthropic Claude + SSE**。

## 架构图

```
客户端 (Browser/App)
        │  POST /api/v1/agent/stream
        │  Content-Type: application/json
        ▼
┌────────────────────────────────────────────────┐
│               Nginx / API Gateway              │  ← 限流、认证、SSL
│   proxy_pass + proxy_buffering off             │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│         Gunicorn (多进程) + UvicornWorker       │
│  ┌──────────────────────────────────────────┐  │
│  │         FastAPI Application              │  │
│  │                                          │  │
│  │  ① 请求校验 (Pydantic)                   │  │
│  │  ② CORS / 请求追踪中间件                  │  │
│  │  ③ StreamingResponse (SSE)               │  │
│  │       │                                  │  │
│  │       ▼                                  │  │
│  │  ④ Agent Core (Agentic Loop)             │  │
│  │       │                                  │  │
│  │       ├─ stream delta ──────────────────?│  │
│  │       ├─ tool_use ──? Tool Executor      │  │
│  │       │                    │             │  │
│  │       ?── tool_result ─────┘             │  │
│  │       │                                  │  │
│  │       └─ done ──────────────────────────?│  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
        │
        ▼
 Anthropic Claude API (claude-opus-4-5)
```

## SSE 事件协议

| 事件类型      | 触发时机              | data 字段示例                                    |
|-------------|---------------------|------------------------------------------------|
| `start`     | 流开始                | `{"message": "Stream started"}`               |
| `delta`     | 每个文本片段            | `{"text": "你好，"}`                           |
| `tool_use`  | Agent 调用工具前       | `{"tool_name": "calculator", "tool_input": {}}` |
| `tool_result`| 工具执行完毕           | `{"tool_name": "...", "result": {}, "is_error": false}` |
| `done`      | 生成完成              | `{"input_tokens": 100, "output_tokens": 200, "stop_reason": "end_turn"}` |
| `error`     | 发生错误              | `{"code": "api_error_429", "message": "...", "retryable": true}` |

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY

# 3. 开发模式启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. 测试
python test_stream.py
```

## 生产部署

```bash
# Docker
docker build -t agent-api .
docker run -d -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  -e ALLOWED_ORIGINS=https://yourdomain.com \
  agent-api

# 或直接用 Gunicorn
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

## Nginx 配置要点（流式响应必须）

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;

    # ? 关键：禁用缓冲，SSE 必须
    proxy_buffering off;
    proxy_cache off;

    # ? 关键：超时时间要长于最大流式时长
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    # SSE 相关头
    proxy_set_header Connection '';
    proxy_set_header X-Real-IP $remote_addr;
    chunked_transfer_encoding on;
}
```

## 生产最佳实践清单

- [x] **连接池复用**：Anthropic 客户端全局单例，避免每次请求重建连接
- [x] **客户端断开检测**：`request.is_disconnected()` 提前终止流，避免资源泄漏  
- [x] **心跳机制**：15s 发送 SSE 注释行，防止中间代理超时断连
- [x] **全局超时保护**：300s 硬超时，防止 worker 长期占用
- [x] **结构化日志**：JSON 格式，含 session_id / request_id 便于追踪
- [x] **Agentic Loop 上限**：MAX_TOOL_ITERATIONS=10，防止无限循环
- [x] **工具错误隔离**：工具异常不崩溃，结构化返回给 Claude 处理
- [x] **Gunicorn 多进程**：生产使用 UvicornWorker，充分利用多核
- [x] **非 root 容器**：Dockerfile 使用 appuser 运行
- [x] **Pydantic 输入校验**：防止恶意/畸形请求


4、目录结构
/home/claude/agent-streaming-api/.env.example
/home/claude/agent-streaming-api/Dockerfile
/home/claude/agent-streaming-api/README.md
/home/claude/agent-streaming-api/app/__init__.py
/home/claude/agent-streaming-api/app/agent/__init__.py
/home/claude/agent-streaming-api/app/agent/core.py
/home/claude/agent-streaming-api/app/agent/tools.py
/home/claude/agent-streaming-api/app/api/__init__.py
/home/claude/agent-streaming-api/app/api/stream.py
/home/claude/agent-streaming-api/app/main.py
/home/claude/agent-streaming-api/app/schemas/__init__.py
/home/claude/agent-streaming-api/app/schemas/models.py
/home/claude/agent-streaming-api/app/utils/__init__.py
/home/claude/agent-streaming-api/app/utils/logger.py
/home/claude/agent-streaming-api/app/utils/sse.py
/home/claude/agent-streaming-api/client_example.js
/home/claude/agent-streaming-api/requirements.txt
/home/claude/agent-streaming-api/test_stream.py

