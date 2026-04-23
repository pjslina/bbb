"""
app/core/config.py
统一配置管理 — 使用 pydantic-settings，支持 .env 文件与环境变量。
所有模块通过 `from app.core.config import settings` 获取配置，
禁止散落的 os.getenv() 调用。
"""
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API Key")
    llm_model: str = Field("claude-sonnet-4-20250514")
    llm_max_tokens: int = Field(4096, ge=1, le=8192)
    llm_max_iterations: int = Field(10, ge=1, le=30, description="ReAct 最大循环次数")

    # ── MySQL ────────────────────────────────────────────────────────────────
    mysql_host: str = Field("127.0.0.1")
    mysql_port: int = Field(3306, ge=1, le=65535)
    mysql_user: str = Field("agent")
    mysql_password: str = Field(...)
    mysql_database: str = Field("agent_db")
    mysql_pool_size: int = Field(10, ge=1, le=100)
    mysql_pool_recycle: int = Field(3600)

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = Field("development")
    app_log_level: str = Field("INFO")
    app_cors_origins: List[str] = Field(default=["*"])

    # ── 外部 API ─────────────────────────────────────────────────────────────
    weather_api_key: str = Field(default="")
    weather_api_base_url: str = Field(default="https://api.openweathermap.org/data/2.5")

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def mysql_dsn(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例，整个进程只初始化一次。"""
    return Settings()


# 快捷导入
settings = get_settings()
