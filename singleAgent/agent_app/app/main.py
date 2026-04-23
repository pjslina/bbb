"""
app/main.py
FastAPI 应用入口。

关键设计：
- lifespan 管理数据库连接池的创建与关闭（取代 on_startup/on_shutdown）
- 全局异常 handler 统一错误响应格式
- CORS 中间件按环境配置
- 工具模块在启动时 import（触发自动注册）
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import AgentBaseError, agent_exception_handler, unhandled_exception_handler
from app.core.logging import get_logger, setup_logging

# 初始化日志（必须在所有其他 import 之前）
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动初始化 → yield → 关闭清理。"""
    # ── 启动阶段 ──────────────────────────────────────────────────────────
    logger.info("app_starting", env=settings.app_env, model=settings.llm_model)

    # 初始化 MySQL 连接池
    from app.db.mysql import create_db_pool
    await create_db_pool()

    # 导入工具模块，触发 tool_registry.register()
    import app.tools.weather_tool    # noqa: F401
    import app.tools.database_tool  # noqa: F401

    logger.info("app_started", tools=[t.name for t in __import__('app.tools.base', fromlist=['tool_registry']).tool_registry.get_all()])

    yield  # 应用运行中

    # ── 关闭阶段 ──────────────────────────────────────────────────────────
    from app.db.mysql import close_db_pool
    await close_db_pool()
    logger.info("app_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Single Agent API",
        description="生产级 AI Agent，支持工具调用与流式对话",
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 全局异常 Handler ──────────────────────────────────────────────────
    app.add_exception_handler(AgentBaseError, agent_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── 路由注册 ──────────────────────────────────────────────────────────
    from app.api.v1.chat import router as chat_router
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
        log_config=None,  # 使用 structlog，禁用 uvicorn 默认日志配置
        workers=1,        # 开发环境单进程；生产用 gunicorn 多进程
    )
