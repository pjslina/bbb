"""
app/core/exceptions.py
统一异常体系 — 所有业务异常继承自 AgentBaseError，
FastAPI 全局 handler 将其转换为标准 JSON 响应。
"""
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


# ── 自定义异常 ────────────────────────────────────────────────────────────────

class AgentBaseError(Exception):
    """所有业务异常的基类。"""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: Optional[Any] = None):
        self.message = message
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                **({"detail": self.detail} if self.detail else {}),
            }
        }


class ToolExecutionError(AgentBaseError):
    """工具调用失败。"""
    status_code = 502
    error_code = "TOOL_EXECUTION_ERROR"


class ToolNotFoundError(AgentBaseError):
    """请求了未注册的工具。"""
    status_code = 400
    error_code = "TOOL_NOT_FOUND"


class LLMError(AgentBaseError):
    """LLM API 调用失败。"""
    status_code = 502
    error_code = "LLM_ERROR"


class MaxIterationsError(AgentBaseError):
    """Agent 超过最大迭代次数。"""
    status_code = 422
    error_code = "MAX_ITERATIONS_EXCEEDED"


class ValidationError(AgentBaseError):
    """请求参数校验失败。"""
    status_code = 400
    error_code = "VALIDATION_ERROR"


class DatabaseError(AgentBaseError):
    """数据库操作失败。"""
    status_code = 502
    error_code = "DATABASE_ERROR"


# ── FastAPI 全局 Handler ──────────────────────────────────────────────────────

async def agent_exception_handler(request: Request, exc: AgentBaseError) -> JSONResponse:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.error(
        "agent_error",
        error_code=exc.error_code,
        message=exc.message,
        path=str(request.url),
    )
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.exception("unhandled_error", path=str(request.url), exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}},
    )
