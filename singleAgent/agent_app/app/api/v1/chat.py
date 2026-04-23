"""
app/api/v1/chat.py
流式对话接口 — POST + SSE（Server-Sent Events）。

接口设计：
  POST /api/v1/chat
  Content-Type: application/json
  Accept: text/event-stream

  请求体：ChatRequest（见 schemas.py）

  SSE 响应流（每帧格式）：
    data: {"type": "text_delta",    "data": {"text": "..."}}
    data: {"type": "tool_use_start","data": {"tool_name": "...", ...}}
    data: {"type": "tool_use_end",  "data": {...}}
    data: {"type": "message_stop",  "data": {"session_id": "..."}}
    data: {"type": "error",         "data": {"code": "...", "message": "..."}}
"""
import uuid
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agents.chat_agent import ChatAgent
from app.core.logging import get_logger
from app.models.schemas import ChatRequest, SSEEvent, SSEEventType

router = APIRouter()
logger = get_logger(__name__)


async def _event_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    """将 Agent SSEEvent 序列化为 SSE 帧字符串。"""
    # 绑定 trace 信息到本次请求的 structlog context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        session_id=request.session_id,
        trace_id=str(uuid.uuid4()),
    )

    logger.info("chat_stream_start", message_count=len(request.messages))

    # 心跳帧（确保连接建立后立即有数据，避免某些代理超时）
    yield SSEEvent(type=SSEEventType.PING, data={"session_id": request.session_id}).to_sse()

    agent = ChatAgent(session_id=request.session_id)

    try:
        async for event in agent.run_stream(request.messages):
            yield event.to_sse()
    except Exception as e:
        logger.exception("chat_stream_error", exc_info=e)
        error_event = SSEEvent(
            type=SSEEventType.ERROR,
            data={"code": "STREAM_ERROR", "message": "流式输出发生内部错误"},
        )
        yield error_event.to_sse()
    finally:
        logger.info(
            "chat_stream_end",
            session_id=request.session_id,
        )


@router.post(
    "/chat",
    summary="流式对话接口",
    description="接受对话历史，以 SSE 流式返回 Agent 的思考、工具调用和最终回答。",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE 流式响应",
            "content": {"text/event-stream": {}},
        },
        422: {"description": "请求参数校验失败"},
        500: {"description": "服务内部错误"},
    },
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    流式对话入口。

    前端消费示例（JavaScript）：
    ```js
    const resp = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'xxx', messages: [...] }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\n\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const event = JSON.parse(line.slice(6));
          // 根据 event.type 处理
        }
      }
    }
    ```
    """
    return StreamingResponse(
        _event_generator(request),
        media_type="text/event-stream",
        headers={
            # 禁止中间代理/Nginx 缓冲 SSE
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # CORS（生产中应由中间件统一处理）
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/health", summary="健康检查", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
