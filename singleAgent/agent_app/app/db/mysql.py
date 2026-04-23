"""
app/db/mysql.py
异步 MySQL 连接池管理。
- 使用 aiomysql 连接池，避免每次请求新建连接
- 单例模式：整个应用共享一个 pool
- 提供 lifespan 钩子，在 FastAPI 启动/关闭时管理连接池生命周期
"""
from typing import Optional

import aiomysql

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_pool: Optional[aiomysql.Pool] = None


async def create_db_pool() -> aiomysql.Pool:
    """创建连接池，在应用启动时调用。"""
    global _pool
    logger.info(
        "creating_db_pool",
        host=settings.mysql_host,
        port=settings.mysql_port,
        database=settings.mysql_database,
        pool_size=settings.mysql_pool_size,
    )
    _pool = await aiomysql.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        db=settings.mysql_database,
        minsize=2,
        maxsize=settings.mysql_pool_size,
        autocommit=True,
        connect_timeout=10,
        # 定期回收连接，防止 MySQL 8h timeout 断连
        pool_recycle=settings.mysql_pool_recycle,
        charset="utf8mb4",
        cursorclass=aiomysql.DictCursor,
    )
    logger.info("db_pool_created")
    return _pool


async def close_db_pool() -> None:
    """关闭连接池，在应用关闭时调用。"""
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("db_pool_closed")


async def get_db_pool() -> aiomysql.Pool:
    """获取连接池。工具层通过此函数获取 pool。"""
    if _pool is None:
        raise RuntimeError("Database pool is not initialized. Call create_db_pool() first.")
    return _pool
