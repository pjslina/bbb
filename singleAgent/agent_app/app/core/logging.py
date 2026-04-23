"""
app/core/logging.py
结构化日志 — 生产环境输出 JSON，开发环境输出带颜色的可读格式。
用法：
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("tool_called", tool="weather", args={"city": "Beijing"})
"""
import logging
import sys

import structlog

from app.core.config import settings


def setup_logging() -> None:
    """在应用启动时调用一次。"""
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,          # 携带请求级 context（trace_id 等）
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        # 生产：JSON，便于 ELK/Loki 采集
        renderer = structlog.processors.JSONRenderer()
    else:
        # 开发：彩色可读格式
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # 降低 uvicorn access log 噪音
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
