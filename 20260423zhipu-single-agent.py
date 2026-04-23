# 生产级单Agent智能体应用 — 完整实现

## 一、架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI + SSE                            │
│                    POST /api/v1/chat/stream                     │
└─────────────┬───────────────────────────────────────────────────┘
              │ SSE Stream
┌─────────────▼───────────────────────────────────────────────────┐
│                      LangGraph Workflow                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Step1:意图识别 │───?│ Step2:工具A   │───?│ Step3:工具B ∥ 工具C│  │
│  │  +参数提取    │    │ (查询用户)    │    │ (订单∥库存)       │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                   │             │
│          ┌──────────────┐    ┌──────────────┐     │             │
│          │ Step5:生成报告 │?───│ Step4:逻辑组装 │?────┘             │
│          │ (流式输出)    │    │              │                   │
│          └──────────────┘    └──────────────┘                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Checkpointer (会话持久化)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
              │                    │
    ┌─────────▼──────┐   ┌────────▼───────┐
    │  LLM Factory   │   │  DB Manager    │
    │  ┌───────────┐ │   │ ┌────────────┐ │
    │  │  Qwen     │ │   │ │ main_db    │ │
    │  │  GLM      │ │   │ │ order_db   │ │
    │  └───────────┘ │   │ │ inventory_ │ │
    └────────────────┘   │ └────────────┘ │
                         └────────────────┘
```

## 二、项目结构

```
agent_app/
├── app/
│   ├── __init__.py
│   ├── main.py                 # 应用入口 & 生命周期管理
│   ├── config.py               # 配置管理 (pydantic-settings)
│   ├── models.py               # 请求/响应 Pydantic 模型
│   ├── db/
│   │   ├── __init__.py
│   │   └── manager.py          # 多数据库连接池管理
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py          # LLM 工厂 (Qwen / GLM)
│   ├── tools/
│   │   ├── __init__.py
│   │   └── definitions.py      # SQL 模板 & 工具定义
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py            # Agent 状态定义
│   │   ├── nodes.py            # 图节点实现 (5步)
│   │   └── graph.py            # LangGraph 工作流构建
│   └── api/
│       ├── __init__.py
│       └── routes.py           # SSE 路由
├── requirements.txt
├── .env.example
└── README.md
```

## 三、完整代码实现

### 3.1 依赖文件

```txt
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
sse-starlette==2.1.0
pydantic==2.7.4
pydantic-settings==2.3.4
langchain-core==0.2.27
langchain-openai==0.1.39
langgraph==0.2.2
langgraph-checkpoint==1.0.10
asyncpg==0.29.0
tenacity==8.4.1
python-dotenv==1.0.1
orjson==3.10.6
```

### 3.2 环境变量模板

```bash
# .env.example

# ==================== LLM Configuration ====================
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZHIPU_MODEL=glm-4-plus

# 默认使用的模型
DEFAULT_LLM_PROVIDER=qwen

# ==================== Database Configuration ====================
# 主数据库 (用户信息)
DB_MAIN_DSN=postgresql://postgres:password@localhost:5432/main_db
# 订单数据库
DB_ORDER_DSN=postgresql://postgres:password@localhost:5432/order_db
# 库存数据库 (OpenGauss 兼容 PostgreSQL 协议)
DB_INVENTORY_DSN=postgresql://gaussdb:password@localhost:5432/inventory_db

# 连接池配置
DB_POOL_MIN_SIZE=2
DB_POOL_MAX_SIZE=10
DB_POOL_MAX_INACTIVE_CONNECTION_LIFETIME=300.0

# ==================== App Configuration ====================
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=["*"]
```

### 3.3 配置管理

```python
# app/config.py
"""
集中式配置管理 - 使用 pydantic-settings 从环境变量 / .env 文件加载配置
支持类型校验、默认值、环境变量前缀等
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """单个数据库连接配置"""
    dsn: str = Field(..., description="数据库连接字符串 postgresql://user:pass@host:port/db")
    pool_min_size: int = Field(default=2, ge=1, description="连接池最小连接数")
    pool_max_size: int = Field(default=10, ge=1, description="连接池最大连接数")


class LLMProviderConfig(BaseSettings):
    """LLM 供应商配置"""
    api_key: str = Field(..., description="API Key")
    base_url: str = Field(..., description="OpenAI 兼容 API Base URL")
    model: str = Field(..., description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    streaming: bool = Field(default=True, description="是否启用流式输出")


class AppConfig(BaseSettings):
    """应用全局配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- LLM ----------
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    qwen_temperature: float = 0.7
    qwen_max_tokens: int = 4096

    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4-plus"
    zhipu_temperature: float = 0.7
    zhipu_max_tokens: int = 4096

    default_llm_provider: str = "qwen"

    # ---------- Database ----------
    db_main_dsn: str = "postgresql://postgres:password@localhost:5432/main_db"
    db_order_dsn: str = "postgresql://postgres:password@localhost:5432/order_db"
    db_inventory_dsn: str = "postgresql://gaussdb:password@localhost:5432/inventory_db"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # ---------- App ----------
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = '["*"]'

    # ---------- 派生属性 ----------
    @property
    def qwen_config(self) -> LLMProviderConfig:
        return LLMProviderConfig(
            api_key=self.qwen_api_key,
            base_url=self.qwen_base_url,
            model=self.qwen_model,
            temperature=self.qwen_temperature,
            max_tokens=self.qwen_max_tokens,
        )

    @property
    def zhipu_config(self) -> LLMProviderConfig:
        return LLMProviderConfig(
            api_key=self.zhipu_api_key,
            base_url=self.zhipu_base_url,
            model=self.zhipu_model,
            temperature=self.zhipu_temperature,
            max_tokens=self.zhipu_max_tokens,
        )

    @property
    def database_configs(self) -> Dict[str, DatabaseConfig]:
        return {
            "main_db": DatabaseConfig(
                dsn=self.db_main_dsn,
                pool_min_size=self.db_pool_min_size,
                pool_max_size=self.db_pool_max_size,
            ),
            "order_db": DatabaseConfig(
                dsn=self.db_order_dsn,
                pool_min_size=self.db_pool_min_size,
                pool_max_size=self.db_pool_max_size,
            ),
            "inventory_db": DatabaseConfig(
                dsn=self.db_inventory_dsn,
                pool_min_size=self.db_pool_min_size,
                pool_max_size=self.db_pool_max_size,
            ),
        }

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)


# 全局单例
settings = AppConfig()
```

### 3.4 Pydantic 模型

```python
# app/models.py
"""
API 请求/响应的 Pydantic 模型定义
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    QWEN = "qwen"
    ZHIPU = "zhipu"


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, max_length=10000, description="用户消息")
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话ID, 同一ID支持多轮对话",
    )
    model: LLMProvider = Field(default=LLMProvider.QWEN, description="使用的LLM模型")


class SSEEventType(str, Enum):
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_DATA = "step_data"
    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"
    DONE = "done"


class SSEEvent(BaseModel):
    """SSE 事件统一格式"""
    type: SSEEventType
    step: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_sse(self) -> str:
        import orjson
        return orjson.dumps(
            {"type": self.type.value, "step": self.step, "data": self.data},
            ensure_ascii=False,
        ).decode("utf-8")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    databases: Dict[str, str] = Field(default_factory=dict)
    llm_providers: List[str] = Field(default_factory=list)


class IntentType(str, Enum):
    """意图类型"""
    QUERY_REPORT = "query_report"        # 综合查询报告
    QUERY_USER = "query_user"            # 查询用户信息
    QUERY_ORDER = "query_order"          # 查询订单信息
    QUERY_INVENTORY = "query_inventory"  # 查询库存信息
    CHAT = "chat"                        # 一般对话


class IntentResult(BaseModel):
    """意图识别结果 - 用于 LLM 结构化输出"""
    intent: IntentType = Field(description="识别出的用户意图")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="提取的参数, 如 user_id, start_date, end_date 等",
    )
    sql_template_keys: List[str] = Field(
        default_factory=list,
        description="需要执行的 SQL 模板键列表",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="意图识别置信度",
    )
```

### 3.5 多数据库连接管理

```python
# app/db/manager.py
"""
多数据库连接池管理器
- 支持 PostgreSQL / OpenGauss (兼容 PostgreSQL 协议)
- 基于 asyncpg 的异步连接池
- 应用生命周期内管理连接池的创建与销毁
- 内置重试机制与连接健康检查
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Record

import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import DatabaseConfig, settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    多数据库连接池管理器 (单例模式)
    
    用法:
        db_manager = DatabaseManager()
        await db_manager.init()
        records = await db_manager.execute("main_db", "SELECT * FROM users WHERE id = $1", "U001")
        await db_manager.close()
    """

    def __init__(self) -> None:
        self._pools: Dict[str, asyncpg.Pool] = {}
        self._configs: Dict[str, DatabaseConfig] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def pool_aliases(self) -> List[str]:
        return list(self._pools.keys())

    async def init(self, configs: Optional[Dict[str, DatabaseConfig]] = None) -> None:
        """
        初始化所有数据库连接池
        
        Args:
            configs: 数据库配置字典, key 为别名, value 为 DatabaseConfig.
                     如不传则从全局 settings 读取.
        """
        if self._initialized:
            logger.warning("DatabaseManager already initialized, skipping.")
            return

        self._configs = configs or settings.database_configs

        for alias, config in self._configs.items():
            try:
                pool = await self._create_pool(alias, config)
                self._pools[alias] = pool
                logger.info(
                    f"Database pool [{alias}] created: "
                    f"min={config.pool_min_size}, max={config.pool_max_size}"
                )
            except Exception as e:
                logger.error(f"Failed to create pool [{alias}]: {e}")
                raise

        self._initialized = True
        logger.info(f"DatabaseManager initialized with {len(self._pools)} pool(s).")

    async def _create_pool(self, alias: str, config: DatabaseConfig) -> asyncpg.Pool:
        """创建单个连接池"""
        return await asyncpg.create_pool(
            dsn=config.dsn,
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            # 连接初始化 SQL (设置时区等)
            setup=self._connection_setup,
            # 连接重用前的健康检查
            connection_class=asyncpg.Connection,
        )

    @staticmethod
    async def _connection_setup(conn: asyncpg.Connection) -> None:
        """连接初始化回调: 设置时区、编码等"""
        await conn.execute("SET timezone = 'UTC';")
        await conn.execute("SET client_encoding = 'UTF8';")

    @retry(
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def execute(
        self,
        db_alias: str,
        query: str,
        *args: Any,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        执行参数化 SQL 查询并返回结果列表
        
        Args:
            db_alias: 数据库别名 (如 "main_db")
            query: SQL 语句, 使用 $1, $2 占位符
            *args: SQL 参数
            timeout: 查询超时时间(秒)
            
        Returns:
            查询结果列表, 每行为一个字典
            
        Raises:
            KeyError: 数据库别名不存在
            asyncpg.PostgresError: SQL 执行错误
        """
        pool = self._get_pool(db_alias)
        async with pool.acquire(timeout=timeout) as conn:
            rows: List[Record] = await conn.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]

    @retry(
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def execute_one(
        self,
        db_alias: str,
        query: str,
        *args: Any,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """执行查询并返回单条结果"""
        pool = self._get_pool(db_alias)
        async with pool.acquire(timeout=timeout) as conn:
            row: Optional[Record] = await conn.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None

    async def execute_many(
        self,
        db_alias: str,
        query: str,
        args_list: List[tuple],
        timeout: float = 30.0,
    ) -> str:
        """批量执行 (INSERT/UPDATE/DELETE)"""
        pool = self._get_pool(db_alias)
        async with pool.acquire(timeout=timeout) as conn:
            result = await conn.executemany(query, args_list, timeout=timeout)
            return result

    async def health_check(self) -> Dict[str, str]:
        """检查所有数据库连接状态"""
        result = {}
        for alias, pool in self._pools.items():
            try:
                async with pool.acquire(timeout=5.0) as conn:
                    val = await conn.fetchval("SELECT 1")
                    result[alias] = "ok" if val == 1 else "error"
            except Exception as e:
                result[alias] = f"error: {e}"
        return result

    def _get_pool(self, db_alias: str) -> asyncpg.Pool:
        """获取指定别名的连接池"""
        if db_alias not in self._pools:
            raise KeyError(
                f"Database pool '{db_alias}' not found. "
                f"Available: {list(self._pools.keys())}"
            )
        return self._pools[db_alias]

    async def close(self) -> None:
        """关闭所有连接池"""
        for alias, pool in self._pools.items():
            try:
                await pool.close()
                logger.info(f"Database pool [{alias}] closed.")
            except Exception as e:
                logger.error(f"Error closing pool [{alias}]: {e}")
        self._pools.clear()
        self._initialized = False
        logger.info("DatabaseManager shutdown complete.")


# 全局单例
db_manager = DatabaseManager()
```

### 3.6 LLM 工厂

```python
# app/llm/factory.py
"""
LLM 工厂 - 统一创建不同供应商的 LLM 实例
基于 langchain-openai 的 ChatOpenAI, 利用 OpenAI 兼容协议支持 Qwen / GLM
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import LLMProviderConfig, settings
from app.models import LLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """LLM 实例工厂"""

    _instances: Dict[str, BaseChatModel] = {}

    @classmethod
    def create(
        cls,
        provider: LLMProvider | str = LLMProvider.QWEN,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
    ) -> BaseChatModel:
        """
        创建 LLM 实例
        
        Args:
            provider: 模型供应商
            temperature: 生成温度
            max_tokens: 最大 token 数
            streaming: 是否启用流式输出
            
        Returns:
            BaseChatModel 实例
        """
        provider = LLMProvider(provider)
        cache_key = f"{provider.value}_{temperature}_{max_tokens}_{streaming}"

        if cache_key in cls._instances:
            return cls._instances[cache_key]

        if provider == LLMProvider.QWEN:
            config = settings.qwen_config
        elif provider == LLMProvider.ZHIPU:
            config = settings.zhipu_config
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=temperature or config.temperature,
            max_tokens=max_tokens or config.max_tokens,
            streaming=streaming,
            # 请求超时
            request_timeout=60.0,
            # 重试
            max_retries=3,
        )

        cls._instances[cache_key] = llm
        logger.info(
            f"Created LLM instance: provider={provider.value}, "
            f"model={config.model}, streaming={streaming}"
        )
        return llm

    @classmethod
    def create_structured_llm(
        cls,
        provider: LLMProvider | str = LLMProvider.QWEN,
        schema=None,
        temperature: float = 0.1,
    ):
        """
        创建支持结构化输出的 LLM (用于意图识别等场景)
        
        Args:
            provider: 模型供应商
            schema: Pydantic 模型类 (如 IntentResult)
            temperature: 低温度保证确定性输出
            
        Returns:
            支持结构化输出的 Runnable
        """
        llm = cls.create(provider, temperature=temperature, streaming=False)
        if schema:
            return llm.with_structured_output(schema)
        return llm

    @classmethod
    def clear_cache(cls) -> None:
        """清除缓存 (测试用)"""
        cls._instances.clear()
```

### 3.7 SQL 模板 & 工具定义

```python
# app/tools/definitions.py
"""
SQL 模板系统 & 工具定义
- 预定义 SQL 模板, 防止 SQL 注入
- 参数化查询, 安全可控
- 意图 -> SQL 模板映射
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models import IntentType


# =====================================================
# SQL 模板定义
# =====================================================

@dataclass
class SQLTemplate:
    """SQL 模板"""
    key: str
    db_alias: str
    sql: str
    param_order: List[str]
    description: str
    # 参数默认值 (可选)
    param_defaults: Dict[str, Any] = field(default_factory=dict)


SQL_TEMPLATES: Dict[str, SQLTemplate] = {
    # ---------- Tool A: 用户信息查询 ----------
    "query_user_profile": SQLTemplate(
        key="query_user_profile",
        db_alias="main_db",
        sql="""
            SELECT user_id, username, email, phone, status, created_at
            FROM users
            WHERE user_id = $1
            LIMIT 1
        """,
        param_order=["user_id"],
        description="查询用户基本信息",
    ),

    # ---------- Tool B: 订单统计查询 ----------
    "query_order_stats": SQLTemplate(
        key="query_order_stats",
        db_alias="order_db",
        sql="""
            SELECT
                COUNT(*)          AS order_count,
                COALESCE(SUM(amount), 0) AS total_amount,
                COALESCE(AVG(amount), 0) AS avg_amount,
                MIN(created_at)   AS first_order_date,
                MAX(created_at)   AS last_order_date
            FROM orders
            WHERE user_id = $1
              AND created_at >= $2
              AND created_at <= $3
        """,
        param_order=["user_id", "start_date", "end_date"],
        description="查询用户订单统计",
        param_defaults={"start_date": "2024-01-01", "end_date": "2099-12-31"},
    ),

    # ---------- Tool C: 库存查询 ----------
    "query_product_inventory": SQLTemplate(
        key="query_product_inventory",
        db_alias="inventory_db",
        sql="""
            SELECT
                i.product_id,
                i.product_name,
                i.stock_quantity,
                i.unit_price,
                i.category
            FROM inventory i
            WHERE i.product_id IN (
                SELECT DISTINCT oi.product_id
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE o.user_id = $1
            )
            ORDER BY i.stock_quantity ASC
        """,
        param_order=["user_id"],
        description="查询用户相关产品库存信息",
    ),

    # ---------- 额外模板: 用户订单明细 ----------
    "query_order_details": SQLTemplate(
        key="query_order_details",
        db_alias="order_db",
        sql="""
            SELECT
                o.id          AS order_id,
                o.amount,
                o.status      AS order_status,
                o.created_at,
                json_agg(
                    json_build_object(
                        'product_id', oi.product_id,
                        'product_name', oi.product_name,
                        'quantity', oi.quantity,
                        'price', oi.price
                    )
                ) AS items
            FROM orders o
            LEFT JOIN order_items oi ON o.id = oi.order_id
            WHERE o.user_id = $1
              AND o.created_at >= $2
              AND o.created_at <= $3
            GROUP BY o.id
            ORDER BY o.created_at DESC
            LIMIT 20
        """,
        param_order=["user_id", "start_date", "end_date"],
        description="查询用户订单明细",
        param_defaults={"start_date": "2024-01-01", "end_date": "2099-12-31"},
    ),
}


# =====================================================
# 意图 -> 步骤映射
# =====================================================

# 每个 Intent 对应哪个 Step 需要执行哪些 SQL 模板
INTENT_STEP_MAPPING: Dict[IntentType, Dict[str, List[str]]] = {
    IntentType.QUERY_REPORT: {
        "tool_a": ["query_user_profile"],
        "tool_b": ["query_order_stats"],
        "tool_c": ["query_product_inventory"],
    },
    IntentType.QUERY_USER: {
        "tool_a": ["query_user_profile"],
        "tool_b": [],
        "tool_c": [],
    },
    IntentType.QUERY_ORDER: {
        "tool_a": ["query_user_profile"],
        "tool_b": ["query_order_stats", "query_order_details"],
        "tool_c": [],
    },
    IntentType.QUERY_INVENTORY: {
        "tool_a": [],
        "tool_b": [],
        "tool_c": ["query_product_inventory"],
    },
    IntentType.CHAT: {
        "tool_a": [],
        "tool_b": [],
        "tool_c": [],
    },
}


def get_template(key: str) -> SQLTemplate:
    """获取 SQL 模板"""
    if key not in SQL_TEMPLATES:
        raise KeyError(f"SQL template '{key}' not found. Available: {list(SQL_TEMPLATES.keys())}")
    return SQL_TEMPLATES[key]


def resolve_params(template: SQLTemplate, extracted_params: Dict[str, Any]) -> list:
    """
    将提取的参数按 SQL 模板的 param_order 排列, 并填充默认值
    
    Args:
        template: SQL 模板
        extracted_params: LLM 提取的参数字典
        
    Returns:
        按序排列的参数列表, 对应 $1, $2, ...
    """
    merged = {**template.param_defaults, **extracted_params}
    params = []
    for key in template.param_order:
        if key not in merged:
            raise ValueError(
                f"Missing required parameter '{key}' for template '{template.key}'. "
                f"Required: {template.param_order}"
            )
        params.append(merged[key])
    return params
```

### 3.8 Agent 状态定义

```python
# app/agent/state.py
"""
LangGraph Agent 状态定义
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.models import IntentType


class AgentState(TypedDict):
    """
    Agent 工作流状态
    
    注意: 使用 Annotated + reducer 的字段会自动合并而非覆盖
    """
    # ---- 对话历史 (自动追加) ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- Step 1: 意图识别 ----
    intent: Optional[IntentType]
    params: Optional[Dict[str, Any]]
    sql_template_keys: Optional[List[str]]

    # ---- Step 2: 工具 A 结果 ----
    tool_a_result: Optional[Dict[str, Any]]

    # ---- Step 3: 工具 B/C 结果 ----
    tool_b_result: Optional[Dict[str, Any]]
    tool_c_result: Optional[Dict[str, Any]]

    # ---- Step 4: 逻辑组装 ----
    assembled_data: Optional[Dict[str, Any]]

    # ---- Step 5: 生成报告 ----
    report: Optional[str]

    # ---- 通用 ----
    current_step: str
    errors: Annotated[list[str], operator.add]  # 错误信息 (自动追加)
```

### 3.9 图节点实现（核心逻辑）

```python
# app/agent/nodes.py
"""
LangGraph 图节点实现 - 5 步工作流
Step1: 意图识别与参数提取
Step2: 工具调用 A
Step3: 工具调用 B & C (并行)
Step4: 逻辑组装
Step5: 生成报告
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.db.manager import db_manager
from app.llm.factory import LLMFactory
from app.models import IntentResult, IntentType, LLMProvider
from app.tools.definitions import (
    INTENT_STEP_MAPPING,
    SQL_TEMPLATES,
    get_template,
    resolve_params,
)

logger = logging.getLogger(__name__)

# =====================================================
# System Prompts
# =====================================================

INTENT_RECOGNITION_SYSTEM_PROMPT = """\
你是一个专业的意图识别引擎。根据用户的输入，准确识别意图并提取结构化参数。

## 可识别的意图类型：
1. query_report - 综合查询报告：需要查询用户信息 + 订单统计 + 库存信息
2. query_user - 查询用户信息：仅需查询用户基本信息
3. query_order - 查询订单信息：需要查询订单统计和明细
4. query_inventory - 查询库存信息：需要查询产品库存
5. chat - 一般对话：无需查询数据库的闲聊或知识问答

## 参数提取规则：
- user_id: 用户ID，如 "U001"、"U002" 等
- start_date: 起始日期，格式 "YYYY-MM-DD"，如无明确时间则默认 "2024-01-01"
- end_date: 结束日期，格式 "YYYY-MM-DD"，如无明确时间则默认 "2099-12-31"

## 示例：
用户: "帮我查一下用户U001今年的消费情况"
→ intent: query_report, params: {user_id: "U001", start_date: "2024-01-01", end_date: "2099-12-31"}

用户: "你好"
→ intent: chat, params: {}

请根据用户输入识别意图并提取参数。"""

REPORT_GENERATION_SYSTEM_PROMPT = """\
你是一个资深数据分析报告生成专家。根据提供的查询数据，生成专业、清晰、结构化的分析报告。

## 报告要求：
1. 使用 Markdown 格式
2. 包含数据摘要、关键指标、趋势分析
3. 如有异常数据需重点标注
4. 给出合理的业务建议
5. 语言简洁专业，避免冗余

## 输出格式：
### ? 数据分析报告
**查询时间**: {当前时间}

#### 一、用户概况
...

#### 二、订单分析
...

#### 三、库存状态
...

#### 四、综合建议
...
"""

CHAT_SYSTEM_PROMPT = """\
你是一个友好、专业的AI助手。请根据用户的输入给出有帮助的回答。
如果用户询问数据相关的问题，建议他们使用查询功能。"""


# =====================================================
# Step 1: 意图识别与参数提取
# =====================================================

async def intent_recognition_node(state: AgentState) -> dict:
    """
    Step 1: 意图识别与参数提取
    使用 LLM 结构化输出识别意图类型和提取参数
    """
    logger.info("[Step1] Intent recognition started.")

    llm_provider = state.get("llm_provider", "qwen")
    structured_llm = LLMFactory.create_structured_llm(
        provider=llm_provider,
        schema=IntentResult,
        temperature=0.1,
    )

    # 构造消息
    messages = state["messages"]
    # 取最近的用户消息
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    prompt_messages = [
        SystemMessage(content=INTENT_RECOGNITION_SYSTEM_PROMPT),
        HumanMessage(content=last_user_msg),
    ]

    try:
        result: IntentResult = await structured_llm.ainvoke(prompt_messages)
        logger.info(
            f"[Step1] Intent recognized: {result.intent.value}, "
            f"params={result.params}, confidence={result.confidence}"
        )

        # 获取对应步骤的 SQL 模板键
        step_mapping = INTENT_STEP_MAPPING.get(result.intent, {})
        all_template_keys = []
        for step_keys in step_mapping.values():
            all_template_keys.extend(step_keys)

        return {
            "intent": result.intent,
            "params": result.params,
            "sql_template_keys": all_template_keys,
            "current_step": "intent_recognition",
        }

    except Exception as e:
        logger.error(f"[Step1] Intent recognition failed: {e}")
        return {
            "intent": IntentType.CHAT,
            "params": {},
            "sql_template_keys": [],
            "current_step": "intent_recognition",
            "errors": [f"意图识别异常: {str(e)}"],
        }


# =====================================================
# Step 2: 工具调用 A (查询用户信息等)
# =====================================================

async def tool_call_a_node(state: AgentState) -> dict:
    """
    Step 2: 执行工具 A
    根据意图执行对应的 SQL 查询 (通常为查询用户基本信息)
    """
    logger.info("[Step2] Tool call A started.")
    intent = state.get("intent")
    params = state.get("params", {})

    step_mapping = INTENT_STEP_MAPPING.get(intent, {})
    template_keys = step_mapping.get("tool_a", [])

    if not template_keys:
        logger.info("[Step2] No SQL templates for tool_a, skipping.")
        return {"tool_a_result": None, "current_step": "tool_call_a"}

    results = {}
    for key in template_keys:
        try:
            template = get_template(key)
            sql_params = resolve_params(template, params)

            logger.info(
                f"[Step2] Executing SQL template '{key}' on '{template.db_alias}' "
                f"with params={sql_params}"
            )
            rows = await db_manager.execute(template.db_alias, template.sql, *sql_params)
            results[key] = {
                "template": key,
                "description": template.description,
                "row_count": len(rows),
                "data": rows,
            }
            logger.info(f"[Step2] SQL template '{key}' returned {len(rows)} rows.")

        except Exception as e:
            logger.error(f"[Step2] SQL template '{key}' failed: {e}")
            results[key] = {"template": key, "error": str(e), "data": []}

    return {"tool_a_result": results, "current_step": "tool_call_a"}


# =====================================================
# Step 3: 工具调用 B (订单查询) & C (库存查询) — 并行执行
# =====================================================

async def tool_call_b_node(state: AgentState) -> dict:
    """
    Step 3a: 执行工具 B
    根据意图执行订单相关的 SQL 查询
    """
    logger.info("[Step3b] Tool call B started.")
    intent = state.get("intent")
    params = state.get("params", {})

    step_mapping = INTENT_STEP_MAPPING.get(intent, {})
    template_keys = step_mapping.get("tool_b", [])

    if not template_keys:
        logger.info("[Step3b] No SQL templates for tool_b, skipping.")
        return {"tool_b_result": None, "current_step": "tool_call_b"}

    results = {}
    for key in template_keys:
        try:
            template = get_template(key)
            sql_params = resolve_params(template, params)

            logger.info(
                f"[Step3b] Executing SQL template '{key}' on '{template.db_alias}'"
            )
            rows = await db_manager.execute(template.db_alias, template.sql, *sql_params)
            results[key] = {
                "template": key,
                "description": template.description,
                "row_count": len(rows),
                "data": rows,
            }
            logger.info(f"[Step3b] SQL template '{key}' returned {len(rows)} rows.")

        except Exception as e:
            logger.error(f"[Step3b] SQL template '{key}' failed: {e}")
            results[key] = {"template": key, "error": str(e), "data": []}

    return {"tool_b_result": results, "current_step": "tool_call_b"}


async def tool_call_c_node(state: AgentState) -> dict:
    """
    Step 3b: 执行工具 C
    根据意图执行库存相关的 SQL 查询
    """
    logger.info("[Step3c] Tool call C started.")
    intent = state.get("intent")
    params = state.get("params", {})

    step_mapping = INTENT_STEP_MAPPING.get(intent, {})
    template_keys = step_mapping.get("tool_c", [])

    if not template_keys:
        logger.info("[Step3c] No SQL templates for tool_c, skipping.")
        return {"tool_c_result": None, "current_step": "tool_call_c"}

    results = {}
    for key in template_keys:
        try:
            template = get_template(key)
            sql_params = resolve_params(template, params)

            logger.info(
                f"[Step3c] Executing SQL template '{key}' on '{template.db_alias}'"
            )
            rows = await db_manager.execute(template.db_alias, template.sql, *sql_params)
            results[key] = {
                "template": key,
                "description": template.description,
                "row_count": len(rows),
                "data": rows,
            }
            logger.info(f"[Step3c] SQL template '{key}' returned {len(rows)} rows.")

        except Exception as e:
            logger.error(f"[Step3c] SQL template '{key}' failed: {e}")
            results[key] = {"template": key, "error": str(e), "data": []}

    return {"tool_c_result": results, "current_step": "tool_call_c"}


# =====================================================
# Step 4: 逻辑组装
# =====================================================

async def logic_assembly_node(state: AgentState) -> dict:
    """
    Step 4: 逻辑组装
    将工具 A/B/C 的结果进行清洗、格式化、逻辑整合
    """
    logger.info("[Step4] Logic assembly started.")

    assembled: Dict[str, Any] = {
        "intent": state.get("intent", "").value if state.get("intent") else "unknown",
        "query_time": datetime.now().isoformat(),
        "user_params": state.get("params", {}),
        "tool_a": _sanitize_tool_result(state.get("tool_a_result")),
        "tool_b": _sanitize_tool_result(state.get("tool_b_result")),
        "tool_c": _sanitize_tool_result(state.get("tool_c_result")),
        "errors": state.get("errors", []),
    }

    # 逻辑处理: 衍生计算
    _compute_derived_metrics(assembled)

    logger.info("[Step4] Logic assembly completed.")
    return {"assembled_data": assembled, "current_step": "logic_assembly"}


def _sanitize_tool_result(result: Optional[Dict]) -> Dict:
    """清洗工具结果, 移除过大字段、处理空值等"""
    if result is None:
        return {"status": "skipped", "data": None}

    sanitized = {}
    for key, val in result.items():
        entry = {
            "status": "error" if "error" in val else "success",
            "description": val.get("description", ""),
            "row_count": val.get("row_count", 0),
        }
        if "error" in val:
            entry["error"] = val["error"]
        else:
            # 截断过大的数据 (防止 LLM 上下文溢出)
            data = val.get("data", [])
            if len(data) > 50:
                entry["data"] = data[:50]
                entry["truncated"] = True
                entry["total_rows"] = len(data)
            else:
                entry["data"] = data
        sanitized[key] = entry
    return sanitized


def _compute_derived_metrics(assembled: Dict) -> None:
    """计算衍生指标"""
    tool_b = assembled.get("tool_b", {})
    
    for key, val in tool_b.items():
        if val.get("status") != "success" or not val.get("data"):
            continue
        for row in val["data"]:
            # 订单相关衍生指标
            if "total_amount" in row and "order_count" in row and row["order_count"] > 0:
                row["avg_order_amount"] = round(row["total_amount"] / row["order_count"], 2)
    
    # 库存预警
    tool_c = assembled.get("tool_c", {})
    low_stock_items = []
    for key, val in tool_c.items():
        if val.get("status") != "success" or not val.get("data"):
            continue
        for row in val["data"]:
            if row.get("stock_quantity", 999) < 10:
                low_stock_items.append({
                    "product_id": row.get("product_id"),
                    "product_name": row.get("product_name"),
                    "stock_quantity": row.get("stock_quantity"),
                })
    
    assembled["derived"] = {
        "low_stock_items": low_stock_items,
        "low_stock_count": len(low_stock_items),
    }


# =====================================================
# Step 5: 生成报告
# =====================================================

async def report_generation_node(state: AgentState) -> dict:
    """
    Step 5: 生成报告 (流式输出)
    使用 LLM 根据组装数据生成自然语言报告
    """
    logger.info("[Step5] Report generation started.")

    llm_provider = state.get("llm_provider", "qwen")
    intent = state.get("intent")
    assembled = state.get("assembled_data", {})

    # 根据意图选择不同的提示
    if intent == IntentType.CHAT:
        return await _handle_chat(state, llm_provider)
    
    return await _handle_report(state, llm_provider, assembled)


async def _handle_chat(state: AgentState, llm_provider: str) -> dict:
    """处理一般对话"""
    llm = LLMFactory.create(provider=llm_provider, streaming=True)
    
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + state["messages"]
    response = await llm.ainvoke(messages)
    
    return {
        "report": response.content,
        "messages": [AIMessage(content=response.content)],
        "current_step": "report_generation",
    }


async def _handle_report(state: AgentState, llm_provider: str, assembled: Dict) -> dict:
    """处理数据报告生成"""
    llm = LLMFactory.create(provider=llm_provider, streaming=True, temperature=0.5)

    # 构造数据上下文
    data_context = json.dumps(assembled, ensure_ascii=False, indent=2, default=str)
    
    # 取用户原始问题
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    prompt_messages = [
        SystemMessage(content=REPORT_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=f"""
## 用户问题
{user_question}

## 查询数据
```json
{data_context}
```

请根据以上数据生成分析报告。
"""),
    ]

    # 调用 LLM (流式输出通过 astream_events 捕获)
    response = await llm.ainvoke(prompt_messages)

    return {
        "report": response.content,
        "messages": [AIMessage(content=response.content)],
        "current_step": "report_generation",
    }


# =====================================================
# 路由函数
# =====================================================

def route_by_intent(state: AgentState) -> str:
    """
    根据意图决定下一步
    - chat → 直接跳到报告生成
    - 其他 → 走完整工具调用链
    """
    intent = state.get("intent")
    if intent == IntentType.CHAT:
        return "report_generation"
    return "tool_call_a"
```

### 3.10 LangGraph 工作流构建

```python
# app/agent/graph.py
"""
LangGraph 工作流构建
定义 5 步流水线: 意图识别 → 工具A → 工具B∥C → 逻辑组装 → 报告生成
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    intent_recognition_node,
    logic_assembly_node,
    report_generation_node,
    route_by_intent,
    tool_call_a_node,
    tool_call_b_node,
    tool_call_c_node,
)
from app.agent.state import AgentState
from app.models import LLMProvider

logger = logging.getLogger(__name__)

# 有效的节点名称集合 (用于 SSE 事件过滤)
NODE_NAMES = {
    "intent_recognition",
    "tool_call_a",
    "tool_call_b",
    "tool_call_c",
    "logic_assembly",
    "report_generation",
}


def build_agent_graph(
    checkpointer: Optional[MemorySaver] = None,
) -> StateGraph:
    """
    构建 Agent 工作流图
    
    架构:
        START → intent_recognition ──[条件路由]──┬→ tool_call_a → [tool_call_b ∥ tool_call_c] → logic_assembly → report_generation → END
                                                 └→ report_generation → END (chat 意图)
    
    Args:
        checkpointer: 状态检查点 (用于持久化对话历史)
        
    Returns:
        编译后的 LangGraph 图
    """
    # 创建状态图
    graph = StateGraph(AgentState)

    # ---- 添加节点 ----
    graph.add_node("intent_recognition", intent_recognition_node)
    graph.add_node("tool_call_a", tool_call_a_node)
    graph.add_node("tool_call_b", tool_call_b_node)
    graph.add_node("tool_call_c", tool_call_c_node)
    graph.add_node("logic_assembly", logic_assembly_node)
    graph.add_node("report_generation", report_generation_node)

    # ---- 添加边 ----
    # 入口
    graph.set_entry_point("intent_recognition")

    # 条件路由: 意图识别后决定走工具链还是直接对话
    graph.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "tool_call_a": "tool_call_a",
            "report_generation": "report_generation",
        },
    )

    # 工具调用链: A → [B ∥ C] → 逻辑组装
    graph.add_edge("tool_call_a", "tool_call_b")
    graph.add_edge("tool_call_a", "tool_call_c")

    # 扇入: B 和 C 都完成后才进入逻辑组装
    graph.add_edge("tool_call_b", "logic_assembly")
    graph.add_edge("tool_call_c", "logic_assembly")

    # 逻辑组装 → 报告生成
    graph.add_edge("logic_assembly", "report_generation")

    # 报告生成 → 结束
    graph.add_edge("report_generation", END)

    # ---- 编译 ----
    cp = checkpointer or MemorySaver()
    compiled = graph.compile(
        checkpointer=cp,
        # 可选: 递归限制, 防止无限循环
        recursive_limit=25,
    )

    logger.info("Agent graph compiled successfully.")
    return compiled


# 全局编译图实例
_compiled_graph = None


def get_compiled_graph() -> StateGraph:
    """获取编译后的图 (懒加载单例)"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph
```

### 3.11 SSE 路由

```python
# app/api/routes.py
"""
API 路由 - SSE 流式接口
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_compiled_graph, NODE_NAMES
from app.agent.state import AgentState
from app.db.manager import db_manager
from app.models import ChatRequest, HealthResponse, LLMProvider, SSEEvent, SSEEventType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Agent"])


# =====================================================
# 辅助函数
# =====================================================

def _format_sse(event_type: str, step: Optional[str] = None, data: Optional[dict] = None) -> str:
    """格式化 SSE 事件数据"""
    payload = {"type": event_type}
    if step:
        payload["step"] = step
    if data:
        payload["data"] = data
    return json.dumps(payload, ensure_ascii=False, default=str)


async def _stream_agent_events(
    graph: CompiledStateGraph,
    inputs: dict,
    config: dict,
) -> AsyncIterator[str]:
    """
    从 LangGraph astream_events 生成 SSE 事件流
    
    过滤策略:
    1. on_chain_start/end + langgraph_node ∈ NODE_NAMES → 步骤进度
    2. on_chat_model_stream → LLM Token 流式输出
    3. on_tool_start/end → 工具调用追踪
    """
    try:
        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node")
            event_data = event.get("data", {})

            # ---- 步骤开始 ----
            if kind == "on_chain_start" and node_name in NODE_NAMES:
                yield _format_sse("step_start", step=node_name, data={})

            # ---- 步骤结束 ----
            elif kind == "on_chain_end" and node_name in NODE_NAMES:
                output = event_data.get("output", {})
                # 提取关键信息, 避免发送整个状态 (过大)
                step_output = {}
                if isinstance(output, dict):
                    for key in ("intent", "params", "current_step", "report"):
                        if key in output and output[key] is not None:
                            val = output[key]
                            # 枚举类型转字符串
                            if hasattr(val, "value"):
                                val = val.value
                            step_output[key] = val
                    # 工具结果摘要
                    for key in ("tool_a_result", "tool_b_result", "tool_c_result", "assembled_data"):
                        if key in output and output[key] is not None:
                            step_output[key + "_summary"] = _summarize_tool_result(output[key])

                yield _format_sse("step_end", step=node_name, data=step_output)

            # ---- LLM Token 流式输出 ----
            elif kind == "on_chat_model_stream":
                chunk = event_data.get("chunk")
                if chunk and hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
                    yield _format_sse("token", step=node_name, data={"token": chunk.content})

        # 流结束
        yield _format_sse("done", data={})

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield _format_sse("error", data={"message": str(e)})
        yield _format_sse("done", data={})


def _summarize_tool_result(result) -> dict:
    """生成工具结果摘要 (避免 SSE 传输过大数据)"""
    if result is None:
        return {"status": "skipped"}
    if isinstance(result, dict):
        summary = {}
        for key, val in result.items():
            if isinstance(val, dict):
                summary[key] = {
                    "status": "error" if "error" in val else "success",
                    "row_count": val.get("row_count", 0),
                }
                if "error" in val:
                    summary[key]["error"] = val["error"]
            else:
                summary[key] = str(val)[:100]
        return summary
    return {"type": type(result).__name__}


# =====================================================
# API 端点
# =====================================================

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口 (POST + SSE)
    
    请求体:
    - message: 用户消息
    - thread_id: 会话ID (同一ID支持多轮对话)
    - model: 使用的LLM (qwen / zhipu)
    
    SSE 事件格式:
    - step_start: 步骤开始
    - step_end: 步骤结束 (含关键输出)
    - token: LLM 流式 Token
    - error: 错误
    - done: 流结束
    """
    graph = get_compiled_graph()

    # 构造输入
    inputs: dict = {
        "messages": [HumanMessage(content=request.message)],
        "llm_provider": request.model.value,
    }

    # 配置 (含 thread_id 用于多轮对话状态持久化)
    config: dict = {
        "configurable": {
            "thread_id": request.thread_id,
        },
    }

    logger.info(
        f"Chat stream request: thread_id={request.thread_id}, "
        f"model={request.model.value}, message={request.message[:50]}..."
    )

    async def event_generator():
        async for sse_data in _stream_agent_events(graph, inputs, config):
            yield sse_data

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping=15,  # 心跳间隔(秒)
    )


@router.post("/chat/invoke")
async def chat_invoke(request: ChatRequest):
    """
    非流式聊天接口 (同步等待完整响应)
    """
    graph = get_compiled_graph()

    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "llm_provider": request.model.value,
    }
    config = {
        "configurable": {
            "thread_id": request.thread_id,
        },
    }

    try:
        result = await graph.ainvoke(inputs, config=config)
        return {
            "thread_id": request.thread_id,
            "intent": result.get("intent", "").value if result.get("intent") else None,
            "report": result.get("report"),
            "errors": result.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Invoke error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    db_status = await db_manager.health_check()
    return HealthResponse(
        status="ok",
        databases=db_status,
        llm_providers=["qwen", "zhipu"],
    )


@router.delete("/chat/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """删除会话 (清除对话历史)"""
    # MemorySaver 不支持主动删除, 生产环境使用 PostgresSaver 时可实现
    return {"message": f"Thread {thread_id} deletion requested (depends on checkpointer impl)"}
```

### 3.12 应用入口

```python
# app/main.py
"""
应用入口 - FastAPI 应用配置 & 生命周期管理
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.db.manager import db_manager

# =====================================================
# 日志配置
# =====================================================

def _setup_logging() -> None:
    """配置日志格式"""
    log_format = (
        "%(asctime)s | %(levelname)-7s | %(name)-20s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger(__name__)


# =====================================================
# 应用生命周期
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理: 启动时初始化, 关闭时清理"""
    # ---- Startup ----
    logger.info("=" * 60)
    logger.info("Agent Application Starting...")
    logger.info(f"LLM Providers: qwen({settings.qwen_model}), zhipu({settings.zhipu_model})")
    logger.info(f"Default Provider: {settings.default_llm_provider}")
    logger.info("=" * 60)

    # 初始化数据库连接池
    try:
        await db_manager.init()
        logger.info("Database pools initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # 非致命错误, 允许应用启动 (但查询会失败)

    yield

    # ---- Shutdown ----
    logger.info("Agent Application Shutting Down...")
    await db_manager.close()
    logger.info("Shutdown complete.")


# =====================================================
# FastAPI 应用实例
# =====================================================

app = FastAPI(
    title="Agent API",
    description="生产级单 Agent 智能体应用 - 流式对话 + 多数据库 + 多 LLM",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


# =====================================================
# 根路径
# =====================================================

@app.get("/")
async def root():
    return {
        "service": "Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "chat_stream": "POST /api/v1/chat/stream",
    }


# =====================================================
# 直接运行
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,  # 生产环境关闭 reload
        workers=1,     # LangGraph 有状态, 建议单 worker + 异步
        log_level=settings.log_level.lower(),
        access_log=True,
    )
```

### 3.13 包初始化文件

```python
# app/__init__.py
# app/db/__init__.py
# app/llm/__init__.py
# app/tools/__init__.py
# app/agent/__init__.py
# app/api/__init__.py
# 以上文件均为空，仅需创建即可
```

## 四、启动与使用

### 4.1 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际的 API Key 和数据库 DSN

# 启动服务
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4.2 测试 SSE 流式接口

```python
# test_client.py
import httpx
import json

URL = "http://localhost:8000/api/v1/chat/stream"

def test_stream():
    payload = {
        "message": "帮我查询用户U001在2024年的消费情况和库存情况",
        "thread_id": "test_conv_001",
        "model": "qwen"
    }
    
    with httpx.stream("POST", URL, json=payload, timeout=120) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                evt_type = data.get("type")
                
                if evt_type == "step_start":
                    print(f"\n? [{data['step']}] 开始执行...")
                    
                elif evt_type == "step_end":
                    print(f"? [{data['step']}] 执行完成")
                    if "data" in data and data["data"]:
                        for k, v in data["data"].items():
                            print(f"  └ {k}: {v}")
                    
                elif evt_type == "token":
                    print(data["data"]["token"], end="", flush=True)
                    
                elif evt_type == "error":
                    print(f"\n? 错误: {data['data']['message']}")
                    
                elif evt_type == "done":
                    print("\n\n=== 流结束 ===")

if __name__ == "__main__":
    test_stream()
```

**输出示例：**

```
? [intent_recognition] 开始执行...
? [intent_recognition] 执行完成
  └ intent: query_report
  └ params: {'user_id': 'U001', 'start_date': '2024-01-01', 'end_date': '2024-12-31'}

? [tool_call_a] 开始执行...
? [tool_call_a] 执行完成
  └ tool_a_result_summary: {'query_user_profile': {'status': 'success', 'row_count': 1}}

? [tool_call_b] 开始执行...
? [tool_call_c] 开始执行...
? [tool_call_b] 执行完成
  └ tool_b_result_summary: {'query_order_stats': {'status': 'success', 'row_count': 1}}
? [tool_call_c] 执行完成
  └ tool_c_result_summary: {'query_product_inventory': {'status': 'success', 'row_count': 3}}

? [logic_assembly] 开始执行...
? [logic_assembly] 执行完成

? [report_generation] 开始执行...
### ? 数据分析报告
**查询时间**: 2024-12-15T10:30:00

#### 一、用户概况
用户 U001 (张三) 为活跃用户，注册于 2023-06-15...

#### 二、订单分析
2024年累计下单 47 笔，总消费 ?28,650.00...

#### 三、库存状态
相关产品中有 2 项库存不足(低于10件)...

#### 四、综合建议
建议及时补充库存...

=== 流结束 ===
```

### 4.3 cURL 测试

```bash
# 流式请求
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查询用户U001的消费报告",
    "thread_id": "conv_001",
    "model": "qwen"
  }'

# 非流式请求
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好",
    "thread_id": "conv_002",
    "model": "zhipu"
  }'

# 健康检查
curl http://localhost:8000/api/v1/health
```

### 4.4 多轮对话测试

```python
# 多轮对话: 同一 thread_id 保持上下文
import httpx, json

URL = "http://localhost:8000/api/v1/chat/stream"

def chat(message: str, thread_id: str, model: str = "qwen"):
    full_response = ""
    with httpx.stream("POST", URL, json={
        "message": message, "thread_id": thread_id, "model": model
    }, timeout=120) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    full_response += data["data"]["token"]
                elif data.get("type") == "done":
                    break
    return full_response

# 第一轮
r1 = chat("查询用户U001的消费情况", "multi_turn_001")
print(f"第一轮: {r1[:100]}...")

# 第二轮 (引用上下文)
r2 = chat("把时间范围缩小到2024年第三季度重新查询", "multi_turn_001")
print(f"第二轮: {r2[:100]}...")

# 第三轮 (闲聊)
r3 = chat("谢谢，还有其他建议吗？", "multi_turn_001")
print(f"第三轮: {r3[:100]}...")
```

## 五、关键设计决策与生产考量

### 5.1 架构决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 流式协议 | SSE (非 WebSocket) | 单向推送场景，HTTP 原生支持，自动重连，CDN 友好 |
| 工作流引擎 | LangGraph StateGraph | 原生支持条件路由、扇入扇出、状态持久化 |
| 数据库驱动 | asyncpg | 高性能异步 PostgreSQL 驱动，OpenGauss 兼容 |
| LLM 集成 | ChatOpenAI (统一) | Qwen/GLM 均支持 OpenAI 兼容协议，一套代码适配多模型 |
| SQL 执行 | 模板 + 参数化查询 | 杜绝 SQL 注入，LLM 只选模板不生成 SQL |

### 5.2 安全性

```
? SQL 注入防护: 预定义模板 + asyncpg 参数化查询 ($1, $2)
? 数据脱敏: SSE 传输工具结果摘要，不传原始大字段
? 数据截断: 超过 50 行自动截断，防止 LLM 上下文溢出
? 输入校验: Pydantic 严格校验请求参数 (min_length, max_length, enum)
? CORS 控制: 可配置允许的域名
? 生产环境需加: API Key 认证 / Rate Limiting / 请求签名
```

### 5.3 可靠性

```
? 连接池: asyncpg Pool 自动管理连接生命周期
? 重试: tenacity 指数退避重试 (数据库连接、LLM 调用)
? 降级: 意图识别失败时降级为 chat 模式
? 错误隔离: 单个工具失败不影响其他工具，错误记录在 state.errors
? 递归限制: LangGraph recursive_limit=25 防止无限循环
```

### 5.4 生产环境升级清单

```python
# 1. Checkpointer: MemorySaver → PostgresSaver (持久化对话历史)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(DSN) as checkpointer:
    await checkpointer.setup()
    graph = build_agent_graph(checkpointer=checkpointer)

# 2. 认证中间件
from fastapi import Depends, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()
async def verify_token(credentials = Security(security)):
    # JWT 验证逻辑
    pass

@router.post("/chat/stream", dependencies=[Depends(verify_token)])

# 3. Rate Limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
@router.post("/chat/stream")
@limiter.limit("10/minute")

# 4. 结构化日志
import structlog
logger = structlog.get_logger()

# 5. 分布式追踪
from opentelemetry import trace
tracer = trace.get_tracer("agent-app")

# 6. 监控指标
from prometheus_client import Counter, Histogram
REQUEST_COUNT = Counter("agent_requests_total", "Total requests")
LLM_LATENCY = Histogram("agent_llm_latency_seconds", "LLM call latency")
```

### 5.5 数据库初始化 (测试用)

```sql
-- main_db: 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id   VARCHAR(20) PRIMARY KEY,
    username  VARCHAR(100) NOT NULL,
    email     VARCHAR(200),
    phone     VARCHAR(20),
    status    VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO users VALUES ('U001', '张三', 'zhangsan@example.com', '13800138000', 'active', '2023-06-15');

-- order_db: 订单表
CREATE TABLE IF NOT EXISTS orders (
    id         SERIAL PRIMARY KEY,
    user_id    VARCHAR(20) NOT NULL,
    amount     DECIMAL(10,2) NOT NULL,
    status     VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_items (
    id           SERIAL PRIMARY KEY,
    order_id     INTEGER REFERENCES orders(id),
    product_id   VARCHAR(20) NOT NULL,
    product_name VARCHAR(200),
    quantity     INTEGER DEFAULT 1,
    price        DECIMAL(10,2) NOT NULL
);

-- inventory_db: 库存表
CREATE TABLE IF NOT EXISTS inventory (
    product_id    VARCHAR(20) PRIMARY KEY,
    product_name  VARCHAR(200) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    unit_price    DECIMAL(10,2) NOT NULL,
    category      VARCHAR(50)
);
```

这套实现覆盖了从 HTTP 接口到多数据库查询到 LLM 多轮对话的完整链路，代码结构清晰、可扩展，满足生产级部署要求。