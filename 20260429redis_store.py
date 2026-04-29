"""Redis会话记忆：多轮对话历史存储 + 摘要压缩。

支持三种部署模式:
- standalone: 单机模式
- cluster: Redis Cluster 集群模式
- sentinel: Redis Sentinel 高可用模式
"""
from __future__ import annotations
import json
import os
from enum import Enum
from typing import Any
import redis.asyncio as aioredis
from redis.asyncio import Redis, ConnectionPool, ClusterNode
from redis.asyncio.sentinel import Sentinel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
    messages_from_dict, messages_to_dict,
)
from app.core.config import settings
from app.core.exceptions import MemoryError
from app.core.logging import get_logger

logger = get_logger(__name__)

_KEY_MSGS    = "session:{sid}:messages"
_KEY_SUMMARY = "session:{sid}:summary"
_KEY_META    = "session:{sid}:meta"

SUMMARISE_PROMPT = """你是一个对话摘要引擎。
用3-5句话概括以下对话的关键信息、用户意图和已完成的操作。
只输出摘要，不要前缀。

对话：
{conversation}"""


class RedisMode(Enum):
    """Redis 部署模式枚举"""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class RedisMemoryStore:
    _instance: "RedisMemoryStore | None" = None

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._mode: RedisMode = RedisMode.STANDALONE
        self._sentinel: Sentinel | None = None
        self._cluster_nodes: list[ClusterNode] = []

    @classmethod
    def get_instance(cls) -> "RedisMemoryStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _parse_redis_mode(self) -> RedisMode:
        """解析 Redis 部署模式"""
        mode_str = os.environ.get("REDIS_MODE", "standalone").lower().strip()
        try:
            return RedisMode(mode_str)
        except ValueError:
            logger.warning(
                f"未知的Redis模式 '{mode_str}', 默认使用 standalone",
                mode=mode_str
            )
            return RedisMode.STANDALONE

    def _build_standalone_connection_pool(
        self,
        url: str,
        password: str | None,
        max_connections: int,
    ) -> ConnectionPool:
        """构建单机模式的连接池"""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        return aioredis.ConnectionPool.from_url(
            url,
            password=password or None,
            max_connections=max_connections,
            decode_responses=True,
        )

    def _build_cluster_connection_pool(
        self,
        nodes: str,
        password: str | None,
        max_connections: int,
    ) -> list[ClusterNode]:
        """构建集群模式的节点列表"""
        cluster_nodes = []
        node_list = [n.strip() for n in nodes.split(",") if n.strip()]
        
        for node_str in node_list:
            if ":" in node_str:
                host, port_str = node_str.rsplit(":", 1)
                port = int(port_str)
            else:
                host = node_str
                port = 6379
            
            cluster_nodes.append(
                ClusterNode(host=host, port=port, password=password)
            )
        
        return cluster_nodes

    def _build_sentinel_connection_pool(
        self,
        sentinel_nodes: str,
        sentinel_password: str | None,
        master_name: str,
        max_connections: int,
    ) -> tuple[Sentinel, str]:
        """构建 Sentinel 模式的连接池
        
        Returns:
            tuple: (Sentinel实例, Master名称)
        """
        sentinels = []
        node_list = [n.strip() for n in sentinel_nodes.split(",") if n.strip()]
        
        for node_str in node_list:
            if ":" in node_str:
                host, port_str = node_str.rsplit(":", 1)
                port = int(port_str)
            else:
                host = node_str
                port = 26379  # Sentinel 默认端口
            
            sentinels.append((host, port))
        
        sentinel = Sentinel(
            sentinels,
            sentinel_password=sentinel_password,
        )
        
        return sentinel, master_name

    async def init(self) -> None:
        """根据配置初始化 Redis 连接"""
        try:
            self._mode = self._parse_redis_mode()
            
            if self._mode == RedisMode.STANDALONE:
                await self._init_standalone()
            elif self._mode == RedisMode.CLUSTER:
                await self._init_cluster()
            elif self._mode == RedisMode.SENTINEL:
                await self._init_sentinel()
            
            await self._redis.ping()
            logger.info(
                f"Redis记忆存储已连接",
                mode=self._mode.value
            )
        except Exception as exc:
            raise MemoryError(f"Redis连接失败: {exc}") from exc

    async def _init_standalone(self) -> None:
        """初始化单机模式"""
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        password = os.environ.get("REDIS_PASSWORD") or None
        max_conn = int(os.environ.get("REDIS_MAX_CONNECTIONS", "20"))
        
        pool = self._build_standalone_connection_pool(
            url, password, max_conn
        )
        self._redis = aioredis.Redis(connection_pool=pool)
        logger.info("Redis 单机模式已配置", url=url)

    async def _init_cluster(self) -> None:
        """初始化集群模式"""
        cluster_nodes_str = os.environ.get("REDIS_CLUSTER_NODES", "")
        if not cluster_nodes_str:
            raise MemoryError(
                "REDIS_CLUSTER_NODES 未配置，请检查环境变量"
            )
        
        password = os.environ.get("REDIS_CLUSTER_PASSWORD") or None
        max_conn = int(os.environ.get("REDIS_MAX_CONNECTIONS", "20"))
        
        self._cluster_nodes = self._build_cluster_connection_pool(
            cluster_nodes_str, password, max_conn
        )
        
        # Redis Cluster 使用特殊的方式创建连接
        self._redis = aioredis.Redis(
            cluster_init=True,
            startup_nodes=self._cluster_nodes,
            password=password,
            max_connections_per_node=max_conn,
            decode_responses=True,
        )
        logger.info("Redis 集群模式已配置", nodes=cluster_nodes_str)

    async def _init_sentinel(self) -> None:
        """初始化 Sentinel 模式"""
        sentinel_nodes_str = os.environ.get("REDIS_SENTINEL_NODES", "")
        if not sentinel_nodes_str:
            raise MemoryError(
                "REDIS_SENTINEL_NODES 未配置，请检查环境变量"
            )
        
        sentinel_password = os.environ.get("REDIS_SENTINEL_PASSWORD") or None
        master_name = os.environ.get("REDIS_MASTER_NAME", "mymaster")
        max_conn = int(os.environ.get("REDIS_MAX_CONNECTIONS", "20"))
        
        self._sentinel, master_name = self._build_sentinel_connection_pool(
            sentinel_nodes_str, sentinel_password, master_name, max_conn
        )
        
        # 从 Sentinel 获取 Master 节点
        master = self._sentinel.master_for(
            master_name,
            redis_class=aioredis.Redis,
            password=sentinel_password,
            max_connections=max_conn,
            decode_responses=True,
        )
        self._redis = master
        logger.info(
            "Redis Sentinel 模式已配置",
            master=master_name,
            sentinels=sentinel_nodes_str
        )

    async def close(self) -> None:
        """关闭 Redis 连接"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        
        if self._sentinel:
            # Sentinel 连接池会在 Redis 关闭时自动清理
            self._sentinel = None

    # ── 公共API ──────────────────────────────────────────────────────────────

    async def load_history(self, session_id: str) -> list[BaseMessage]:
        self._ensure()
        messages: list[BaseMessage] = []
        summary = await self._get_summary(session_id)
        if summary:
            messages.append(SystemMessage(content=f"[历史摘要]\n{summary}"))
        raw = await self._redis.get(_KEY_MSGS.format(sid=session_id))  # type: ignore
        if raw:
            try:
                messages.extend(messages_from_dict(json.loads(raw)))
            except Exception as exc:
                logger.warning("历史反序列化失败，重置", session_id=session_id, error=str(exc))
        return messages

    async def append_turn(self, session_id: str, human: HumanMessage, ai: AIMessage) -> None:
        self._ensure()
        existing = await self._load_raw(session_id)
        existing.extend([human, ai])
        await self._save_msgs(session_id, existing)

    async def update_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        self._ensure()
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]
        await self._save_msgs(session_id, filtered)

    async def set_summary(self, session_id: str, summary: str) -> None:
        self._ensure()
        ttl = int(os.environ.get("SESSION_TTL", "86400"))
        pipe = self._redis.pipeline()  # type: ignore
        pipe.set(_KEY_SUMMARY.format(sid=session_id), summary)
        pipe.expire(_KEY_SUMMARY.format(sid=session_id), ttl)
        await pipe.execute()

    async def get_message_count(self, session_id: str) -> int:
        return len(await self._load_raw(session_id))

    async def session_exists(self, session_id: str) -> bool:
        self._ensure()
        return bool(await self._redis.exists(_KEY_MSGS.format(sid=session_id)))  # type: ignore

    async def delete_session(self, session_id: str) -> None:
        self._ensure()
        await self._redis.delete(  # type: ignore
            _KEY_MSGS.format(sid=session_id),
            _KEY_SUMMARY.format(sid=session_id),
            _KEY_META.format(sid=session_id),
        )

    async def get_meta(self, session_id: str) -> dict[str, Any]:
        self._ensure()
        raw = await self._redis.get(_KEY_META.format(sid=session_id))  # type: ignore
        return json.loads(raw) if raw else {}

    async def set_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        self._ensure()
        ttl = int(os.environ.get("SESSION_TTL", "86400"))
        pipe = self._redis.pipeline()  # type: ignore
        pipe.set(_KEY_META.format(sid=session_id), json.dumps(meta, ensure_ascii=False))
        pipe.expire(_KEY_META.format(sid=session_id), ttl)
        await pipe.execute()

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    async def _load_raw(self, session_id: str) -> list[BaseMessage]:
        raw = await self._redis.get(_KEY_MSGS.format(sid=session_id))  # type: ignore
        if not raw:
            return []
        try:
            return messages_from_dict(json.loads(raw))
        except Exception:
            return []

    async def _save_msgs(self, session_id: str, msgs: list[BaseMessage]) -> None:
        ttl = int(os.environ.get("SESSION_TTL", "86400"))
        pipe = self._redis.pipeline()  # type: ignore
        pipe.set(_KEY_MSGS.format(sid=session_id),
                 json.dumps(messages_to_dict(msgs), ensure_ascii=False))
        pipe.expire(_KEY_MSGS.format(sid=session_id), ttl)
        await pipe.execute()

    async def _get_summary(self, session_id: str) -> str:
        return await self._redis.get(_KEY_SUMMARY.format(sid=session_id)) or ""  # type: ignore

    def _ensure(self) -> None:
        if not self._redis:
            raise MemoryError("RedisMemoryStore 未初始化，请先调用 init()")

    @property
    def mode(self) -> RedisMode:
        """获取当前 Redis 部署模式"""
        return self._mode


async def maybe_compress_session(session_id: str, store: RedisMemoryStore, llm: Any) -> bool:
    """若消息数超过阈值，自动生成摘要并裁剪历史。"""
    threshold = int(os.environ.get("SUMMARY_THRESHOLD", "20"))
    count = await store.get_message_count(session_id)
    if count < threshold:
        return False
    messages = await store._load_raw(session_id)
    keep_n = threshold // 2
    to_summarise, to_keep = messages[:-keep_n], messages[-keep_n:]
    lines = []
    for m in to_summarise:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        lines.append(f"{role}: {m.content}")
    resp = await llm.ainvoke([HumanMessage(
        content=SUMMARISE_PROMPT.format(conversation="\n".join(lines))
    )])
    summary = resp.content if hasattr(resp, "content") else str(resp)
    await store.set_summary(session_id, summary)
    await store.update_messages(session_id, to_keep)
    logger.info("会话已压缩", session_id=session_id, before=count, after=len(to_keep))
    return True


memory_store = RedisMemoryStore.get_instance()