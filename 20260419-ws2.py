1-readme.md

# AI Agent WebSocket Server

## 项目结构

```
agent_server/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── websocket.py          # WebSocket 路由与连接处理
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # 配置管理（环境变量）
│   │   ├── logging.py            # 结构化日志
│   │   └── exceptions.py         # 自定义异常
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent.py              # Agent 核心编排逻辑
│   │   ├── session.py            # 会话管理（Redis）
│   │   └── connection.py         # WebSocket 连接管理器
│   ├── models/
│   │   ├── __init__.py
│   │   └── messages.py           # Pydantic 消息模型
│   └── utils/
│       ├── __init__.py
│       └── rate_limiter.py       # 限流工具
├── tests/
│   ├── test_websocket.py
│   └── test_session.py
├── scripts/
│   └── ws_client_test.py         # 本地测试客户端
├── main.py                       # 应用入口
├── pyproject.toml
└── .env.example
```

## 快速启动

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 核心特性

- WebSocket 双端统一接口
- Redis 会话持久化
- 结构化 JSON 日志
- 连接级限流
- 优雅中断（abort）
- 心跳保活
- 完整错误处理与重连支持


2-pyproject.toml

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "agent-server"
version = "1.0.0"
description = "Production-grade AI Agent WebSocket Server"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "websockets>=13.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "redis[asyncio]>=5.2.0",
    "openai>=1.54.0",          # LLM 客户端，可替换为其他
    "python-dotenv>=1.0.0",
    "structlog>=24.4.0",
    "tenacity>=9.0.0",         # 重试
    "prometheus-client>=0.21.0", # 指标
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "websockets>=13.0",
]

3-config.py

"""
app/core/config.py
─────────────────
集中管理所有配置，通过环境变量注入，支持 .env 文件。
生产环境通过 K8s Secret / 配置中心注入，不应硬编码。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用基础 ──────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    # ── LLM ───────────────────────────────────────────────────────────
    llm_api_key: str = Field(..., description="LLM API Key，必填")
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_timeout: int = 60

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str = ""
    session_ttl: int = 86400          # 24h
    session_max_history: int = 50

    # ── WebSocket ─────────────────────────────────────────────────────
    ws_heartbeat_interval: int = 30
    ws_max_message_size: int = 65536   # 64KB
    ws_connection_timeout: int = 300

    # ── 限流 ──────────────────────────────────────────────────────────
    rate_limit_enabled: bool = True
    rate_limit_per_ip: int = 10
    rate_limit_msg_per_min: int = 30

    # ── CORS ──────────────────────────────────────────────────────────
    allowed_origins: list[str] = ["*"]

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature 必须在 0~2 之间")
        return v


@lru_cache
def get_settings() -> Settings:
    """单例，整个进程共享同一份配置。"""
    return Settings()
    
    
4-logging.py

"""
app/core/logging.py
───────────────────
基于 structlog 的结构化日志。
- 开发环境：彩色可读格式
- 生产环境：JSON 格式（便于 ELK / 云日志平台采集）

用法：
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("ws_connected", session_id=sid, client=client_type)
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import get_settings


def _add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """注入全局上下文：服务名、环境。"""
    event_dict["service"] = "agent-server"
    event_dict["env"] = get_settings().app_env
    return event_dict


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_dev:
        # 开发：彩色控制台输出
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # 生产：JSON，便于日志平台采集
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # 同步 stdlib logging，让 uvicorn / fastapi 日志也走 structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
    
    
5-exception.py
"""
app/core/exceptions.py
──────────────────────
统一异常体系。所有业务异常继承自 AgentBaseError，
携带 error_code 便于前端精确处理。
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    # 客户端错误 4xxx
    INVALID_MESSAGE     = 4000
    SESSION_NOT_FOUND   = 4001
    SESSION_EXPIRED     = 4002
    MESSAGE_TOO_LARGE   = 4003
    RATE_LIMITED        = 4029
    UNAUTHORIZED        = 4010

    # 服务端错误 5xxx
    AGENT_ERROR         = 5000
    LLM_TIMEOUT         = 5001
    LLM_UNAVAILABLE     = 5002
    SESSION_STORE_ERROR = 5003
    INTERNAL_ERROR      = 5999


class AgentBaseError(Exception):
    """所有业务异常的基类。"""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.INTERNAL_ERROR):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {
            "type": "error",
            "code": int(self.code),
            "message": self.message,
        }


class InvalidMessageError(AgentBaseError):
    def __init__(self, detail: str = "消息格式无效"):
        super().__init__(detail, ErrorCode.INVALID_MESSAGE)


class SessionNotFoundError(AgentBaseError):
    def __init__(self, session_id: str):
        super().__init__(f"会话不存在: {session_id}", ErrorCode.SESSION_NOT_FOUND)


class SessionExpiredError(AgentBaseError):
    def __init__(self, session_id: str):
        super().__init__(f"会话已过期: {session_id}", ErrorCode.SESSION_EXPIRED)


class MessageTooLargeError(AgentBaseError):
    def __init__(self, size: int, limit: int):
        super().__init__(
            f"消息体过大: {size} 字节，上限 {limit} 字节",
            ErrorCode.MESSAGE_TOO_LARGE,
        )


class RateLimitedError(AgentBaseError):
    def __init__(self):
        super().__init__("请求过于频繁，请稍后重试", ErrorCode.RATE_LIMITED)


class LLMTimeoutError(AgentBaseError):
    def __init__(self):
        super().__init__("LLM 响应超时", ErrorCode.LLM_TIMEOUT)


class LLMUnavailableError(AgentBaseError):
    def __init__(self, detail: str = ""):
        super().__init__(f"LLM 服务不可用: {detail}", ErrorCode.LLM_UNAVAILABLE)


class SessionStoreError(AgentBaseError):
    def __init__(self, detail: str = ""):
        super().__init__(f"会话存储异常: {detail}", ErrorCode.SESSION_STORE_ERROR)
        
        
6-message.py

"""
app/models/messages.py
──────────────────────
WebSocket 上下行消息的 Pydantic 模型。
所有消息以 JSON 传输，type 字段作为鉴别器。

上行（客户端 → 服务端）：ClientMessage
下行（服务端 → 客户端）：各种 ServerEvent
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════
# 枚举
# ══════════════════════════════════════════════════════════════════════

class ClientType(StrEnum):
    WEB         = "web"
    MINIPROGRAM = "miniprogram"
    UNKNOWN     = "unknown"


class MessageRole(StrEnum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


# ══════════════════════════════════════════════════════════════════════
# 上行消息（客户端 → 服务端）
# ══════════════════════════════════════════════════════════════════════

class ClientMeta(BaseModel):
    client: ClientType = ClientType.UNKNOWN
    app_version: str | None = None


class ChatMessage(BaseModel):
    """发起对话。"""
    type: Literal["chat"]
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = Field(None, description="None 表示新建会话")
    meta: ClientMeta = Field(default_factory=ClientMeta)

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class AbortMessage(BaseModel):
    """中断当前生成。"""
    type: Literal["abort"]
    session_id: str


class PingMessage(BaseModel):
    """客户端心跳。"""
    type: Literal["ping"]


# 判别联合类型，按 type 字段路由
ClientMessage = Annotated[
    Union[ChatMessage, AbortMessage, PingMessage],
    Field(discriminator="type"),
]


# ══════════════════════════════════════════════════════════════════════
# 下行事件（服务端 → 客户端）
# ══════════════════════════════════════════════════════════════════════

class SessionCreatedEvent(BaseModel):
    type: Literal["session_created"] = "session_created"
    session_id: str


class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    content: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    status: Literal["running", "done", "error"]
    result: Any | None = None


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    session_id: str
    usage: dict[str, int] = Field(default_factory=dict)


class AbortedEvent(BaseModel):
    type: Literal["aborted"] = "aborted"
    session_id: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: int
    message: str


class PongEvent(BaseModel):
    type: Literal["pong"] = "pong"


# 历史消息（存入 Redis 的格式）
class HistoryMessage(BaseModel):
    role: MessageRole
    content: str
    
    
7-rate_limiter.py

"""
app/utils/rate_limiter.py
─────────────────────────
基于 Redis 的滑动窗口限流器。

两层限流：
1. IP 级并发连接数限制（防止单 IP 建立过多连接）
2. 连接级消息频率限制（防止单连接发送消息过快）
"""

import time

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.exceptions import RateLimitedError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimiter:
    """
    滑动窗口限流器，使用 Redis ZSET 实现。
    key 格式：rl:{scope}:{identifier}
    """

    def __init__(self, redis: aioredis.Redis):
        self._redis = redis

    async def check_connection_limit(self, ip: str) -> None:
        """检查 IP 并发连接数，超限则抛出 RateLimitedError。"""
        if not settings.rate_limit_enabled:
            return

        key = f"rl:conn:{ip}"
        count = await self._redis.get(key)
        current = int(count) if count else 0

        if current >= settings.rate_limit_per_ip:
            logger.warning("rate_limit_conn", ip=ip, current=current)
            raise RateLimitedError()

    async def increment_connection(self, ip: str) -> None:
        """连接建立时递增计数。"""
        key = f"rl:conn:{ip}"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, settings.ws_connection_timeout)
        await pipe.execute()

    async def decrement_connection(self, ip: str) -> None:
        """连接断开时递减计数（不低于 0）。"""
        key = f"rl:conn:{ip}"
        current = await self._redis.get(key)
        if current and int(current) > 0:
            await self._redis.decr(key)

    async def check_message_rate(self, connection_id: str) -> None:
        """
        滑动窗口：60 秒内消息数不超过 rate_limit_msg_per_min。
        使用 ZSET，score = 时间戳，value = 唯一标识。
        """
        if not settings.rate_limit_enabled:
            return

        key = f"rl:msg:{connection_id}"
        now = time.time()
        window_start = now - 60.0

        pipe = self._redis.pipeline()
        # 清除窗口外的旧记录
        pipe.zremrangebyscore(key, "-inf", window_start)
        # 统计当前窗口内的请求数
        pipe.zcard(key)
        # 加入本次请求
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 60)
        results = await pipe.execute()

        current_count = results[1]
        if current_count >= settings.rate_limit_msg_per_min:
            logger.warning(
                "rate_limit_msg",
                connection_id=connection_id,
                count=current_count,
            )
            raise RateLimitedError()
            
            
8-session.py

"""
app/services/session.py
───────────────────────
会话管理服务。

职责：
- 创建 / 加载 / 删除会话
- 管理对话历史（存入 Redis List，控制最大长度）
- 会话 TTL 自动续期

Redis 数据结构：
    session:{id}:meta   → Hash  { created_at, client_type, ... }
    session:{id}:history → List [ JSON(HistoryMessage), ... ]
"""

import json
import time
import uuid

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.exceptions import SessionExpiredError, SessionNotFoundError, SessionStoreError
from app.core.logging import get_logger
from app.models.messages import HistoryMessage, MessageRole

logger = get_logger(__name__)
settings = get_settings()


class SessionService:
    def __init__(self, redis: aioredis.Redis):
        self._redis = redis

    # ── 内部 Key 工具 ──────────────────────────────────────────────────

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"session:{session_id}:history"

    # ── 公开接口 ───────────────────────────────────────────────────────

    async def create_session(self, client_type: str = "unknown") -> str:
        """创建新会话，返回 session_id。"""
        session_id = str(uuid.uuid4())
        meta = {
            "created_at": str(time.time()),
            "client_type": client_type,
            "message_count": "0",
        }
        try:
            pipe = self._redis.pipeline()
            pipe.hset(self._meta_key(session_id), mapping=meta)
            pipe.expire(self._meta_key(session_id), settings.session_ttl)
            await pipe.execute()
        except Exception as e:
            logger.error("session_create_failed", error=str(e))
            raise SessionStoreError(str(e))

        logger.info("session_created", session_id=session_id, client=client_type)
        return session_id

    async def validate_session(self, session_id: str) -> dict:
        """
        验证会话有效性，返回 meta 信息。
        不存在 → SessionNotFoundError
        已过期（元数据消失）→ SessionExpiredError
        """
        try:
            meta = await self._redis.hgetall(self._meta_key(session_id))
        except Exception as e:
            raise SessionStoreError(str(e))

        if not meta:
            # 尝试判断是真不存在还是过期（生产中可通过独立标记区分）
            raise SessionExpiredError(session_id)

        # 续期：每次访问重置 TTL
        await self._refresh_ttl(session_id)
        return {k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in meta.items()}

    async def get_history(self, session_id: str) -> list[HistoryMessage]:
        """获取会话历史消息列表（时间正序）。"""
        try:
            raw_list = await self._redis.lrange(self._history_key(session_id), 0, -1)
        except Exception as e:
            raise SessionStoreError(str(e))

        history = []
        for raw in raw_list:
            try:
                data = json.loads(raw)
                history.append(HistoryMessage(**data))
            except Exception:
                # 单条解析失败不影响整体
                logger.warning("history_parse_error", raw=raw)
        return history

    async def append_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
    ) -> None:
        """追加消息到历史，超过最大长度时裁剪头部（最早的消息）。"""
        msg = HistoryMessage(role=role, content=content)
        try:
            pipe = self._redis.pipeline()
            pipe.rpush(self._history_key(session_id), msg.model_dump_json())
            # 裁剪：保留最新的 N 条
            pipe.ltrim(
                self._history_key(session_id),
                -settings.session_max_history,
                -1,
            )
            pipe.expire(self._history_key(session_id), settings.session_ttl)
            # 递增消息计数
            pipe.hincrby(self._meta_key(session_id), "message_count", 1)
            await pipe.execute()
        except Exception as e:
            raise SessionStoreError(str(e))

    async def delete_session(self, session_id: str) -> None:
        """主动删除会话（用户登出或显式清空）。"""
        try:
            pipe = self._redis.pipeline()
            pipe.delete(self._meta_key(session_id))
            pipe.delete(self._history_key(session_id))
            await pipe.execute()
        except Exception as e:
            logger.error("session_delete_failed", session_id=session_id, error=str(e))

        logger.info("session_deleted", session_id=session_id)

    # ── 私有 ──────────────────────────────────────────────────────────

    async def _refresh_ttl(self, session_id: str) -> None:
        pipe = self._redis.pipeline()
        pipe.expire(self._meta_key(session_id), settings.session_ttl)
        pipe.expire(self._history_key(session_id), settings.session_ttl)
        await pipe.execute()
        
        
9-connection.py

"""
app/services/connection.py
──────────────────────────
WebSocket 连接管理器。

职责：
- 管理所有活跃连接（connection_id → WebSocket）
- 维护连接级 abort 标志
- 提供安全的消息发送封装（防止因连接断开导致异常扩散）
- 暴露连接统计指标

设计说明：
- connection_id 是连接级 UUID（每次 WS 握手生成）
- session_id 是业务级 UUID（跨连接持久，存 Redis）
- 两者不同：用户刷新页面 session 不变，但 connection 会重建
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionState:
    """单个 WebSocket 连接的运行时状态。"""

    connection_id: str
    websocket: WebSocket
    session_id: Optional[str] = None
    client_ip: str = "unknown"
    client_type: str = "unknown"

    # 当前是否有正在进行的生成任务
    _abort_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    # 最后活跃时间戳（用于空闲检测）
    last_active_at: float = field(default_factory=lambda: __import__("time").time())

    def request_abort(self) -> None:
        self._abort_event.set()

    def reset_abort(self) -> None:
        self._abort_event.clear()

    def should_abort(self) -> bool:
        return self._abort_event.is_set()

    def touch(self) -> None:
        import time
        self.last_active_at = time.time()


class ConnectionManager:
    """
    全进程单例，管理所有 WebSocket 连接。

    注意：单进程内有效。若部署多实例，abort 信号需通过 Redis Pub/Sub
    广播（可在此扩展，当前为单机模式）。
    """

    def __init__(self) -> None:
        # connection_id → ConnectionState
        self._connections: dict[str, ConnectionState] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        client_ip: str,
        client_type: str = "unknown",
    ) -> ConnectionState:
        """接受连接并注册，返回 ConnectionState。"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        state = ConnectionState(
            connection_id=connection_id,
            websocket=websocket,
            client_ip=client_ip,
            client_type=client_type,
        )
        async with self._lock:
            self._connections[connection_id] = state

        logger.info(
            "ws_connected",
            connection_id=connection_id,
            ip=client_ip,
            client_type=client_type,
            total_connections=len(self._connections),
        )
        return state

    async def disconnect(self, connection_id: str) -> None:
        """注销连接。"""
        async with self._lock:
            state = self._connections.pop(connection_id, None)

        if state:
            logger.info(
                "ws_disconnected",
                connection_id=connection_id,
                session_id=state.session_id,
                total_connections=len(self._connections),
            )

    async def send(self, state: ConnectionState, payload: dict) -> bool:
        """
        安全发送 JSON 消息。
        返回 True 表示发送成功，False 表示连接已断开。
        """
        try:
            await state.websocket.send_json(payload)
            state.touch()
            return True
        except Exception as e:
            logger.warning(
                "ws_send_failed",
                connection_id=state.connection_id,
                error=str(e),
            )
            return False

    def get_state(self, connection_id: str) -> Optional[ConnectionState]:
        return self._connections.get(connection_id)

    @property
    def active_count(self) -> int:
        return len(self._connections)


# 全局单例（在 main.py 中初始化后注入依赖）
connection_manager = ConnectionManager()


10-agent.py

"""
app/services/agent.py
─────────────────────
Agent 核心编排逻辑。

职责：
- 接收用户消息 + 历史，调用 LLM 流式生成
- 编排工具调用（Tool Use）
- 将 LLM 输出转化为下行事件流（AsyncGenerator）
- 不感知 WebSocket，只产出事件字典，由调用层负责发送

扩展点：
- 替换 LLM 客户端（OpenAI → 其他）只需改 _call_llm()
- 增加工具只需在 TOOLS 注册，并实现对应 _execute_tool()
"""

import asyncio
import json
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import LLMTimeoutError, LLMUnavailableError
from app.core.logging import get_logger
from app.models.messages import HistoryMessage, MessageRole

logger = get_logger(__name__)
settings = get_settings()

# ── 工具定义（OpenAI function calling 格式）────────────────────────────
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索内部知识库，返回相关文档片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回条数", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ── LLM 客户端（单例）─────────────────────────────────────────────────
_llm_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
            timeout=settings.llm_timeout,
            max_retries=0,  # 重试由 tenacity 控制
        )
    return _llm_client


# ══════════════════════════════════════════════════════════════════════
# 工具执行层（替换为真实实现）
# ══════════════════════════════════════════════════════════════════════

async def _execute_tool(name: str, arguments: dict) -> Any:
    """
    工具分发器。
    生产中每个工具应独立模块，此处为示例骨架。
    """
    logger.info("tool_execute", tool=name, args=arguments)

    if name == "search_knowledge_base":
        # TODO: 接入真实向量数据库
        await asyncio.sleep(0.3)  # 模拟 IO
        return {
            "results": [
                {"doc_id": "doc_001", "content": "相关文档内容示例", "score": 0.92},
            ]
        }

    if name == "get_current_time":
        import datetime
        return {"time": datetime.datetime.now().isoformat()}

    return {"error": f"未知工具: {name}"}


# ══════════════════════════════════════════════════════════════════════
# Agent 主逻辑
# ══════════════════════════════════════════════════════════════════════

class AgentService:

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_llm_stream(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ):
        """调用 LLM 流式接口，带重试。"""
        try:
            kwargs: dict[str, Any] = {
                "model": settings.llm_model,
                "messages": messages,
                "max_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "stream": True,
            }
            if use_tools and TOOLS:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"

            return await get_llm_client().chat.completions.create(**kwargs)

        except APITimeoutError as e:
            logger.error("llm_timeout")
            raise LLMTimeoutError() from e
        except APIConnectionError as e:
            logger.error("llm_connection_error", error=str(e))
            raise LLMUnavailableError(str(e)) from e
        except APIStatusError as e:
            logger.error("llm_status_error", status=e.status_code, error=str(e))
            raise LLMUnavailableError(f"HTTP {e.status_code}") from e

    async def run(
        self,
        session_id: str,
        user_message: str,
        history: list[HistoryMessage],
    ) -> AsyncGenerator[dict, None]:
        """
        Agent 主入口，返回事件流。

        调用方通过 async for 迭代事件，逐一发送给客户端。
        事件类型：thinking / tool_call / token / done
        """
        return self._agent_loop(session_id, user_message, history)

    async def _agent_loop(
        self,
        session_id: str,
        user_message: str,
        history: list[HistoryMessage],
    ) -> AsyncGenerator[dict, None]:
        """
        ReAct 循环：
        用户消息 → LLM → [工具调用 → 工具结果 → LLM] → 最终回答
        最多循环 5 次防止死循环。
        """
        # 构建消息列表
        messages: list[dict] = self._build_messages(history, user_message)

        total_tokens = 0
        max_iterations = 5

        for iteration in range(max_iterations):
            logger.debug("agent_iteration", session_id=session_id, iteration=iteration)

            # 收集本轮的完整 assistant 消息（用于下一轮）
            assistant_content = ""
            tool_calls_buffer: dict[int, dict] = {}  # index → tool_call

            stream = await self._call_llm_stream(messages)

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # ── 普通 token ─────────────────────────────────────
                if delta.content:
                    assistant_content += delta.content
                    yield {"type": "token", "content": delta.content}

                # ── 工具调用（增量累积）────────────────────────────
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name or "" if tc.function else "",
                                "arguments": "",
                            }
                        if tc.function and tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id

                # 统计 token 用量
                if chunk.usage:
                    total_tokens += (chunk.usage.total_tokens or 0)

            # ── 无工具调用：生成完毕，结束循环 ────────────────────
            if not tool_calls_buffer:
                break

            # ── 有工具调用：执行工具，追加结果，继续循环 ──────────
            # 把 assistant 的工具调用消息加入历史
            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_buffer.values()
                ],
            })

            # 逐个执行工具
            for tc in tool_calls_buffer.values():
                tool_name = tc["name"]
                try:
                    arguments = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                yield {"type": "tool_call", "tool": tool_name, "status": "running"}

                try:
                    result = await _execute_tool(tool_name, arguments)
                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "status": "done",
                        "result": result,
                    }
                except Exception as e:
                    logger.error("tool_execute_error", tool=tool_name, error=str(e))
                    result = {"error": str(e)}
                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "status": "error",
                        "result": result,
                    }

                # 工具结果追加到消息列表
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

        yield {
            "type": "done",
            "session_id": session_id,
            "usage": {"total_tokens": total_tokens},
        }

    @staticmethod
    def _build_messages(
        history: list[HistoryMessage],
        user_message: str,
    ) -> list[dict]:
        """将历史 + 当前消息转换为 LLM messages 格式。"""
        system_prompt = (
            "你是一个智能助手，回答清晰、简洁、准确。"
            "如果需要搜索资料，请使用提供的工具。"
        )
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": h.role.value, "content": h.content})
        messages.append({"role": "user", "content": user_message})
        return messages
        
        
11-websocket.py

"""
app/api/websocket.py
────────────────────
WebSocket 路由与连接生命周期管理。

职责（单一职责：只处理网络层）：
- 接受连接 / 注销连接
- 消息解析与校验（交给 Pydantic）
- 分发消息到对应处理函数
- 心跳保活
- 优雅的错误处理与日志

业务逻辑（Agent 推理、会话存储）全部委托给 Service 层。
"""

import asyncio
import json
import time

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.exceptions import (
    AgentBaseError,
    InvalidMessageError,
    MessageTooLargeError,
    RateLimitedError,
)
from app.core.logging import get_logger
from app.models.messages import (
    AbortMessage,
    ChatMessage,
    ClientMessage,
    DoneEvent,
    ErrorEvent,
    MessageRole,
    PongEvent,
    SessionCreatedEvent,
)
from app.services.agent import AgentService
from app.services.connection import ConnectionState, connection_manager
from app.services.session import SessionService
from app.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════
# 依赖注入工厂
# ══════════════════════════════════════════════════════════════════════

async def get_redis() -> aioredis.Redis:
    """从 app.state 获取 Redis 连接池（在 main.py 中初始化）。"""
    from main import app
    return app.state.redis


# ══════════════════════════════════════════════════════════════════════
# 主 WebSocket 端点
# ══════════════════════════════════════════════════════════════════════

@router.websocket("/api/v1/chat")
async def chat_endpoint(websocket: WebSocket):
    """
    唯一 WebSocket 入口，网页端和小程序端统一接入。

    生命周期：
    1. 鉴权 + 限流检查
    2. 建立连接，注册到 ConnectionManager
    3. 启动心跳任务
    4. 消息循环（接收 → 校验 → 分发）
    5. 断开连接，清理资源
    """
    # ── 获取客户端 IP（兼容代理）──────────────────────────────────────
    client_ip = _get_client_ip(websocket)
    client_type = websocket.headers.get("X-Client-Type", "unknown")

    redis: aioredis.Redis = await get_redis()
    rate_limiter = RateLimiter(redis)
    session_svc = SessionService(redis)
    agent_svc = AgentService()

    # ── 限流：IP 并发连接检查（握手前）──────────────────────────────
    try:
        await rate_limiter.check_connection_limit(client_ip)
    except RateLimitedError:
        # 拒绝握手，直接关闭（HTTP 429 等价）
        await websocket.close(code=4029, reason="Too Many Connections")
        return

    # ── 建立连接 ───────────────────────────────────────────────────────
    state = await connection_manager.connect(websocket, client_ip, client_type)
    await rate_limiter.increment_connection(client_ip)

    # ── 启动后台任务 ──────────────────────────────────────────────────
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(state),
        name=f"heartbeat-{state.connection_id}",
    )

    try:
        await _message_loop(state, session_svc, agent_svc, rate_limiter)
    except WebSocketDisconnect as e:
        logger.info(
            "ws_client_disconnected",
            connection_id=state.connection_id,
            code=e.code,
        )
    except Exception as e:
        logger.error(
            "ws_unexpected_error",
            connection_id=state.connection_id,
            error=str(e),
            exc_info=True,
        )
    finally:
        heartbeat_task.cancel()
        await connection_manager.disconnect(state.connection_id)
        await rate_limiter.decrement_connection(client_ip)
        logger.info("ws_cleanup_done", connection_id=state.connection_id)


# ══════════════════════════════════════════════════════════════════════
# 消息循环
# ══════════════════════════════════════════════════════════════════════

async def _message_loop(
    state: ConnectionState,
    session_svc: SessionService,
    agent_svc: AgentService,
    rate_limiter: RateLimiter,
) -> None:
    """持续接收并分发客户端消息，直到连接断开。"""

    while True:
        # 接收原始文本
        try:
            raw = await asyncio.wait_for(
                state.websocket.receive_text(),
                timeout=settings.ws_connection_timeout,
            )
        except asyncio.TimeoutError:
            logger.info("ws_idle_timeout", connection_id=state.connection_id)
            await state.websocket.close(code=4000, reason="Idle timeout")
            return

        # 消息大小校验
        if len(raw.encode()) > settings.ws_max_message_size:
            await _send_error(state, MessageTooLargeError(len(raw.encode()), settings.ws_max_message_size))
            continue

        # 消息频率限流
        try:
            await rate_limiter.check_message_rate(state.connection_id)
        except RateLimitedError as e:
            await _send_error(state, e)
            continue

        # 解析消息
        try:
            data = json.loads(raw)
            msg = _parse_message(data)
        except (json.JSONDecodeError, InvalidMessageError) as e:
            await _send_error(state, InvalidMessageError(str(e)))
            continue

        state.touch()

        # 分发
        if isinstance(msg, ChatMessage):
            await _handle_chat(state, msg, session_svc, agent_svc)
        elif isinstance(msg, AbortMessage):
            await _handle_abort(state, msg)
        else:
            # ping
            await connection_manager.send(state, PongEvent().model_dump())


# ══════════════════════════════════════════════════════════════════════
# 消息处理器
# ══════════════════════════════════════════════════════════════════════

async def _handle_chat(
    state: ConnectionState,
    msg: ChatMessage,
    session_svc: SessionService,
    agent_svc: AgentService,
) -> None:
    """处理 chat 消息：管理会话 + 运行 Agent + 流式推送事件。"""

    # ── 1. 会话管理 ───────────────────────────────────────────────────
    if msg.session_id:
        try:
            await session_svc.validate_session(msg.session_id)
            session_id = msg.session_id
        except AgentBaseError as e:
            await _send_error(state, e)
            return
    else:
        session_id = await session_svc.create_session(
            client_type=msg.meta.client.value
        )

    state.session_id = session_id
    state.reset_abort()

    # 通知客户端 session_id（新建或确认）
    await connection_manager.send(
        state,
        SessionCreatedEvent(session_id=session_id).model_dump(),
    )

    # ── 2. 加载历史 ───────────────────────────────────────────────────
    history = await session_svc.get_history(session_id)

    # ── 3. 持久化用户消息 ──────────────────────────────────────────────
    await session_svc.append_message(session_id, MessageRole.USER, msg.message)

    # ── 4. 运行 Agent，流式推送 ────────────────────────────────────────
    full_reply = []
    start_time = time.perf_counter()

    try:
        event_stream = await agent_svc.run(session_id, msg.message, history)

        async for event in event_stream:
            # 检查是否被中断
            if state.should_abort():
                from app.models.messages import AbortedEvent
                await connection_manager.send(
                    state,
                    AbortedEvent(session_id=session_id).model_dump(),
                )
                logger.info("agent_aborted", session_id=session_id)
                return

            # 收集 token 以便存入历史
            if event.get("type") == "token":
                full_reply.append(event.get("content", ""))

            sent = await connection_manager.send(state, event)
            if not sent:
                # 连接已断开，停止生成
                return

    except AgentBaseError as e:
        logger.error(
            "agent_error",
            session_id=session_id,
            code=e.code,
            error=e.message,
        )
        await _send_error(state, e)
        return
    except Exception as e:
        logger.error(
            "agent_unexpected_error",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        from app.core.exceptions import AgentBaseError, ErrorCode
        await _send_error(
            state,
            AgentBaseError("Agent 内部错误", ErrorCode.AGENT_ERROR),
        )
        return

    # ── 5. 持久化 Assistant 回复 ──────────────────────────────────────
    if full_reply:
        reply_text = "".join(full_reply)
        await session_svc.append_message(
            session_id, MessageRole.ASSISTANT, reply_text
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "chat_done",
        session_id=session_id,
        elapsed=round(elapsed, 3),
        reply_len=len(full_reply),
    )


async def _handle_abort(state: ConnectionState, msg: AbortMessage) -> None:
    """处理中断请求。"""
    if state.session_id == msg.session_id:
        state.request_abort()
        logger.info(
            "abort_requested",
            connection_id=state.connection_id,
            session_id=msg.session_id,
        )
    else:
        logger.warning(
            "abort_session_mismatch",
            connection_id=state.connection_id,
            requested=msg.session_id,
            actual=state.session_id,
        )


# ══════════════════════════════════════════════════════════════════════
# 心跳
# ══════════════════════════════════════════════════════════════════════

async def _heartbeat_loop(state: ConnectionState) -> None:
    """
    服务端定时主动发送 ping，防止：
    - 移动端网关 idle 超时断连
    - 小程序后台切换导致连接假死
    """
    while True:
        await asyncio.sleep(settings.ws_heartbeat_interval)
        sent = await connection_manager.send(state, {"type": "ping"})
        if not sent:
            break


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _parse_message(data: dict) -> ChatMessage | AbortMessage:
    """将原始 dict 解析为类型安全的消息对象。"""
    try:
        from pydantic import TypeAdapter
        adapter = TypeAdapter(ClientMessage)
        return adapter.validate_python(data)
    except ValidationError as e:
        raise InvalidMessageError(str(e))


async def _send_error(state: ConnectionState, error: AgentBaseError) -> None:
    """发送错误事件给客户端。"""
    await connection_manager.send(
        state,
        ErrorEvent(code=int(error.code), message=error.message).model_dump(),
    )


def _get_client_ip(websocket: WebSocket) -> str:
    """兼容 X-Forwarded-For 和直连两种情况。"""
    forwarded_for = websocket.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if websocket.client:
        return websocket.client.host
    return "unknown"
    
    
12-main.py

"""
main.py
───────
FastAPI 应用入口。

职责：
- 应用工厂
- 生命周期管理（Redis 连接池、日志初始化）
- 路由注册
- 健康检查 / 指标端点
- 全局中间件（CORS、请求日志）
"""

import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.api.websocket import router as ws_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.connection import connection_manager

# 日志必须在所有模块导入后、应用启动前初始化
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════
# Prometheus 指标
# ══════════════════════════════════════════════════════════════════════

ws_connections_total = Counter(
    "agent_ws_connections_total",
    "WebSocket 连接总数",
    ["client_type"],
)
ws_connections_active = Gauge(
    "agent_ws_connections_active",
    "当前活跃 WebSocket 连接数",
)
chat_requests_total = Counter(
    "agent_chat_requests_total",
    "对话请求总数",
    ["status"],  # success | error | aborted
)


# ══════════════════════════════════════════════════════════════════════
# 生命周期
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 / 关闭时的资源管理。"""

    # ── 启动 ──────────────────────────────────────────────────────────
    logger.info("app_starting", env=settings.app_env, model=settings.llm_model)

    # 初始化 Redis 连接池
    redis_kwargs: dict = {
        "decode_responses": False,
        "socket_keepalive": True,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True,
        "health_check_interval": 30,
    }
    if settings.redis_password:
        redis_kwargs["password"] = settings.redis_password

    app.state.redis = aioredis.from_url(
        settings.redis_url,
        **redis_kwargs,
    )

    # 验证 Redis 连通性
    try:
        await app.state.redis.ping()
        logger.info("redis_connected", url=settings.redis_url)
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        raise RuntimeError(f"Redis 连接失败: {e}") from e

    logger.info("app_started")
    yield

    # ── 关闭 ──────────────────────────────────────────────────────────
    logger.info("app_shutting_down")
    await app.state.redis.aclose()
    logger.info("app_shutdown_complete")


# ══════════════════════════════════════════════════════════════════════
# 应用工厂
# ══════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    _app = FastAPI(
        title="AI Agent WebSocket Server",
        version="1.0.0",
        docs_url="/docs" if settings.is_dev else None,   # 生产关闭 Swagger
        redoc_url="/redoc" if settings.is_dev else None,
        lifespan=lifespan,
    )

    # ── 中间件 ────────────────────────────────────────────────────────
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件
    @_app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start = asyncio.get_event_loop().time()
        response: Response = await call_next(request)
        elapsed = asyncio.get_event_loop().time() - start
        if not request.url.path.startswith("/health"):
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                elapsed=round(elapsed, 4),
            )
        return response

    # ── 路由 ──────────────────────────────────────────────────────────
    _app.include_router(ws_router)

    # ── 健康检查 ──────────────────────────────────────────────────────
    @_app.get("/health", include_in_schema=False)
    async def health():
        """K8s liveness probe。"""
        try:
            await _app.state.redis.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

        status = "ok" if redis_ok else "degraded"
        code = 200 if redis_ok else 503
        return JSONResponse(
            status_code=code,
            content={
                "status": status,
                "redis": "ok" if redis_ok else "error",
                "active_connections": connection_manager.active_count,
            },
        )

    @_app.get("/ready", include_in_schema=False)
    async def ready():
        """K8s readiness probe。"""
        return {"status": "ready"}

    # ── Prometheus 指标 ───────────────────────────────────────────────
    @_app.get("/metrics", include_in_schema=False)
    async def metrics():
        ws_connections_active.set(connection_manager.active_count)
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return _app


app = create_app()


# ══════════════════════════════════════════════════════════════════════
# 直接运行入口（开发用）
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
        log_level=settings.app_log_level.lower(),
        # 生产建议用多 worker + gunicorn 管理进程
        # workers=1 时 WebSocket 状态在内存中是安全的
        workers=1,
        ws_ping_interval=settings.ws_heartbeat_interval,
        ws_ping_timeout=settings.ws_heartbeat_interval * 2,
    )
    
    
13-ws_client_test.py

"""
scripts/ws_client_test.py
──────────────────────────
本地 WebSocket 测试客户端。

用法：
    python scripts/ws_client_test.py
    python scripts/ws_client_test.py --url ws://localhost:8000/api/v1/chat --msg "你好"
    python scripts/ws_client_test.py --test-abort   # 测试中断功能
"""

import argparse
import asyncio
import json
import sys
import time

import websockets


async def chat(url: str, message: str, session_id: str | None = None):
    """发起一次完整对话，打印所有事件。"""
    headers = {"X-Client-Type": "test-script"}

    print(f"\n{'='*60}")
    print(f"连接: {url}")
    print(f"消息: {message}")
    print(f"{'='*60}\n")

    async with websockets.connect(url, additional_headers=headers) as ws:
        payload = {
            "type": "chat",
            "message": message,
            "session_id": session_id,
            "meta": {"client": "unknown"},
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))

        reply_tokens = []
        start = time.perf_counter()

        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "session_created":
                print(f"[SESSION] {event['session_id']}")

            elif event_type == "thinking":
                print(f"[THINKING] {event['content']}")

            elif event_type == "tool_call":
                status = event["status"]
                tool = event["tool"]
                if status == "running":
                    print(f"[TOOL ?] {tool}")
                elif status == "done":
                    print(f"[TOOL ?] {tool} → {json.dumps(event.get('result'), ensure_ascii=False)}")
                else:
                    print(f"[TOOL ?] {tool}")

            elif event_type == "token":
                content = event["content"]
                reply_tokens.append(content)
                print(content, end="", flush=True)

            elif event_type == "done":
                elapsed = time.perf_counter() - start
                print(f"\n\n[DONE] 耗时={elapsed:.2f}s  用量={event.get('usage')}")
                return event.get("session_id"), "".join(reply_tokens)

            elif event_type == "error":
                print(f"\n[ERROR] code={event['code']} msg={event['message']}")
                return None, None

            elif event_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))

            elif event_type == "aborted":
                print(f"\n[ABORTED] session={event['session_id']}")
                return None, None


async def test_abort(url: str):
    """测试中断功能：发出消息后 1 秒发送 abort。"""
    headers = {"X-Client-Type": "test-script"}

    async with websockets.connect(url, additional_headers=headers) as ws:
        payload = {
            "type": "chat",
            "message": "请写一篇关于人工智能的长文章，至少2000字",
            "session_id": None,
            "meta": {"client": "unknown"},
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))

        session_id = None
        token_count = 0

        async def receiver():
            nonlocal session_id, token_count
            async for raw in ws:
                event = json.loads(raw)
                t = event.get("type")
                if t == "session_created":
                    session_id = event["session_id"]
                    print(f"[SESSION] {session_id}")
                elif t == "token":
                    token_count += 1
                    print(event["content"], end="", flush=True)
                elif t in ("done", "aborted", "error"):
                    print(f"\n[{t.upper()}]")
                    return

        async def aborter():
            await asyncio.sleep(1.0)  # 1 秒后中断
            if session_id:
                print(f"\n\n>>> 发送 abort ...")
                await ws.send(json.dumps({
                    "type": "abort",
                    "session_id": session_id,
                }))

        await asyncio.gather(receiver(), aborter())
        print(f"\n共收到 {token_count} 个 token")


async def multi_turn(url: str):
    """多轮对话测试。"""
    session_id = None

    questions = [
        "你好，请介绍一下你自己",
        "你刚才说的第一句话是什么",  # 测试记忆
        "再见",
    ]

    for q in questions:
        sid, _ = await chat(url, q, session_id)
        if sid:
            session_id = sid
        await asyncio.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Agent WebSocket 测试客户端")
    parser.add_argument("--url", default="ws://localhost:8000/api/v1/chat")
    parser.add_argument("--msg", default="今天天气怎么样？")
    parser.add_argument("--test-abort", action="store_true", help="测试中断功能")
    parser.add_argument("--multi-turn", action="store_true", help="多轮对话测试")
    args = parser.parse_args()

    if args.test_abort:
        asyncio.run(test_abort(args.url))
    elif args.multi_turn:
        asyncio.run(multi_turn(args.url))
    else:
        asyncio.run(chat(args.url, args.msg))


if __name__ == "__main__":
    main()
    
    
14-test-websocket.py

"""
tests/test_websocket.py
───────────────────────
WebSocket 接口集成测试。
使用 FastAPI TestClient + pytest-asyncio。

运行：
    pytest tests/ -v
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect


# ── Fixture：mock Redis，避免测试依赖真实 Redis ──────────────────────

@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.hset = AsyncMock(return_value=True)
    redis.hgetall = AsyncMock(return_value={
        b"created_at": b"1700000000.0",
        b"client_type": b"web",
        b"message_count": b"0",
    })
    redis.lrange = AsyncMock(return_value=[])
    redis.rpush = AsyncMock(return_value=1)
    redis.ltrim = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    redis.hincrby = AsyncMock(return_value=1)
    redis.zremrangebyscore = AsyncMock(return_value=0)
    redis.zcard = AsyncMock(return_value=0)
    redis.zadd = AsyncMock(return_value=1)
    redis.incr = AsyncMock(return_value=1)
    redis.decr = AsyncMock(return_value=0)
    redis.pipeline = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(),
        __aexit__=AsyncMock(),
        execute=AsyncMock(return_value=[True, 0, 1, True]),
        hset=MagicMock(return_value=AsyncMock()),
        expire=MagicMock(return_value=AsyncMock()),
        rpush=MagicMock(return_value=AsyncMock()),
        ltrim=MagicMock(return_value=AsyncMock()),
        hincrby=MagicMock(return_value=AsyncMock()),
        zremrangebyscore=MagicMock(return_value=AsyncMock()),
        zcard=MagicMock(return_value=AsyncMock()),
        zadd=MagicMock(return_value=AsyncMock()),
        incr=MagicMock(return_value=AsyncMock()),
    ))
    return redis


@pytest.fixture
def app_with_mock_redis(mock_redis):
    from main import app
    app.state.redis = mock_redis
    return app


# ── 消息模型测试 ──────────────────────────────────────────────────────

class TestMessageParsing:

    def test_chat_message_valid(self):
        from app.models.messages import ChatMessage
        msg = ChatMessage(type="chat", message="你好")
        assert msg.message == "你好"
        assert msg.session_id is None

    def test_chat_message_strips_whitespace(self):
        from app.models.messages import ChatMessage
        msg = ChatMessage(type="chat", message="  hello  ")
        assert msg.message == "hello"

    def test_chat_message_empty_fails(self):
        from pydantic import ValidationError
        from app.models.messages import ChatMessage
        with pytest.raises(ValidationError):
            ChatMessage(type="chat", message="")

    def test_chat_message_too_long_fails(self):
        from pydantic import ValidationError
        from app.models.messages import ChatMessage
        with pytest.raises(ValidationError):
            ChatMessage(type="chat", message="x" * 4097)

    def test_discriminated_union_chat(self):
        from pydantic import TypeAdapter
        from app.models.messages import ClientMessage
        adapter = TypeAdapter(ClientMessage)
        msg = adapter.validate_python({"type": "chat", "message": "test"})
        from app.models.messages import ChatMessage
        assert isinstance(msg, ChatMessage)

    def test_discriminated_union_abort(self):
        from pydantic import TypeAdapter
        from app.models.messages import ClientMessage, AbortMessage
        adapter = TypeAdapter(ClientMessage)
        msg = adapter.validate_python({"type": "abort", "session_id": "abc"})
        assert isinstance(msg, AbortMessage)

    def test_discriminated_union_ping(self):
        from pydantic import TypeAdapter
        from app.models.messages import ClientMessage, PingMessage
        adapter = TypeAdapter(ClientMessage)
        msg = adapter.validate_python({"type": "ping"})
        assert isinstance(msg, PingMessage)


# ── 异常模型测试 ──────────────────────────────────────────────────────

class TestExceptions:

    def test_error_to_dict(self):
        from app.core.exceptions import InvalidMessageError
        err = InvalidMessageError("测试错误")
        d = err.to_dict()
        assert d["type"] == "error"
        assert d["code"] == 4000
        assert "测试错误" in d["message"]

    def test_session_not_found(self):
        from app.core.exceptions import SessionNotFoundError, ErrorCode
        err = SessionNotFoundError("abc-123")
        assert err.code == ErrorCode.SESSION_NOT_FOUND
        assert "abc-123" in err.message


# ── WebSocket 集成测试 ────────────────────────────────────────────────

class TestWebSocketEndpoint:

    def test_health_check(self, app_with_mock_redis):
        client = TestClient(app_with_mock_redis)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ready_check(self, app_with_mock_redis):
        client = TestClient(app_with_mock_redis)
        resp = client.get("/ready")
        assert resp.status_code == 200

    @patch("app.services.agent.AgentService.run")
    def test_new_session_created(self, mock_run, app_with_mock_redis):
        """连接后发送 chat，应收到 session_created 事件。"""

        async def fake_stream(*args, **kwargs):
            async def _gen():
                yield {"type": "token", "content": "你好"}
                yield {"type": "done", "session_id": "mock-session", "usage": {}}
            return _gen()

        mock_run.side_effect = fake_stream

        client = TestClient(app_with_mock_redis)
        with client.websocket_connect("/api/v1/chat") as ws:
            ws.send_json({"type": "chat", "message": "你好", "session_id": None})

            events = []
            for _ in range(5):
                try:
                    data = ws.receive_json()
                    events.append(data)
                    if data.get("type") == "done":
                        break
                except Exception:
                    break

        event_types = [e["type"] for e in events]
        assert "session_created" in event_types

    def test_ping_pong(self, app_with_mock_redis):
        """发送 ping 应收到 pong。"""
        client = TestClient(app_with_mock_redis)
        with client.websocket_connect("/api/v1/chat") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_invalid_json_returns_error(self, app_with_mock_redis):
        """发送非法 JSON 应返回 error 事件。"""
        client = TestClient(app_with_mock_redis)
        with client.websocket_connect("/api/v1/chat") as ws:
            ws.send_text("not-valid-json")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == 4000

    def test_invalid_message_type_returns_error(self, app_with_mock_redis):
        """未知的 type 字段应返回 error。"""
        client = TestClient(app_with_mock_redis)
        with client.websocket_connect("/api/v1/chat") as ws:
            ws.send_json({"type": "unknown_type"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            
