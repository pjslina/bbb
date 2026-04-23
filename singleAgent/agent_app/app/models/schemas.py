"""
app/models/schemas.py
统一的请求/响应 Pydantic 模型。
SSE 消息采用分类型设计，前端可根据 type 字段路由处理逻辑。
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── 请求模型 ──────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    """POST /api/v1/chat 的请求体。"""
    session_id: str = Field(..., description="会话唯一 ID，用于追踪多轮对话上下文")
    messages: List[ChatMessage] = Field(..., min_length=1, description="完整对话历史")
    stream: bool = Field(default=True, description="是否启用 SSE 流式输出")

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "sess_abc123",
            "messages": [
                {"role": "user", "content": "帮我查一下北京今天的天气，并查询最近的订单状态"}
            ],
            "stream": True,
        }
    }}


# ── SSE 事件类型 ──────────────────────────────────────────────────────────────

class SSEEventType(str, Enum):
    # 文本 delta（流式字符）
    TEXT_DELTA = "text_delta"
    # Agent 开始思考
    THINKING_START = "thinking_start"
    # Agent 调用工具（前）
    TOOL_USE_START = "tool_use_start"
    # 工具返回结果
    TOOL_USE_END = "tool_use_end"
    # 整条消息结束
    MESSAGE_STOP = "message_stop"
    # 错误
    ERROR = "error"
    # 心跳（防止连接超时）
    PING = "ping"


class SSEEvent(BaseModel):
    """所有 SSE 消息的基础结构。前端按 type 字段区分。"""
    type: SSEEventType
    data: Optional[Any] = None

    def to_sse(self) -> str:
        """序列化为 SSE 协议格式的字符串。"""
        import json
        payload = json.dumps({"type": self.type, "data": self.data}, ensure_ascii=False)
        return f"data: {payload}\n\n"


# ── 具体事件 data 结构 ────────────────────────────────────────────────────────

class TextDeltaData(BaseModel):
    text: str


class ToolUseStartData(BaseModel):
    tool_name: str
    tool_id: str
    input: Dict[str, Any]


class ToolUseEndData(BaseModel):
    tool_name: str
    tool_id: str
    output: Any
    is_error: bool = False


class MessageStopData(BaseModel):
    session_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class ErrorData(BaseModel):
    code: str
    message: str
