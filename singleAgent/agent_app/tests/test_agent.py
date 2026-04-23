"""
tests/test_agent.py
单元测试与集成测试示例。
使用 pytest + pytest-asyncio，Mock 掉外部依赖。
"""
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import ChatMessage, MessageRole, SSEEventType


# ── 工具注册测试 ──────────────────────────────────────────────────────────────

def test_tool_registry_has_tools():
    """验证工具已正确注册。"""
    # 触发注册
    import app.tools.weather_tool   # noqa
    import app.tools.database_tool  # noqa
    from app.tools.base import tool_registry

    tools = tool_registry.get_all()
    tool_names = [t.name for t in tools]
    assert "get_weather" in tool_names
    assert "query_database" in tool_names


def test_tool_to_anthropic_format():
    """验证工具 schema 符合 Anthropic API 格式。"""
    import app.tools.weather_tool  # noqa
    from app.tools.base import tool_registry

    tool_defs = tool_registry.to_anthropic_tools()
    for td in tool_defs:
        assert "name" in td
        assert "description" in td
        assert "input_schema" in td
        assert td["input_schema"]["type"] == "object"


# ── 工具执行测试 ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_weather_tool_success():
    """Mock HTTP 调用，测试天气工具正常路径。"""
    import app.tools.weather_tool  # noqa
    from app.tools.base import tool_registry

    mock_response = {
        "name": "Beijing",
        "sys": {"country": "CN"},
        "main": {"temp": 25.5, "feels_like": 24.0, "humidity": 60},
        "weather": [{"description": "晴"}],
        "wind": {"speed": 3.2},
    }

    with patch("app.tools.weather_tool.httpx.AsyncClient") as mock_client_cls:
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value = mock_client

        result = await tool_registry.execute("get_weather", "tool_001", {"city": "北京"})

    assert not result.is_error
    assert result.content["city"] == "Beijing"
    assert result.content["temperature"] == 25.5


@pytest.mark.asyncio
async def test_weather_tool_city_not_found():
    """测试城市不存在时的错误处理。"""
    import httpx
    import app.tools.weather_tool  # noqa
    from app.tools.base import tool_registry

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("app.tools.weather_tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)
        )
        mock_client_cls.return_value = mock_client

        result = await tool_registry.execute("get_weather", "tool_002", {"city": "不存在城市xyz"})

    assert result.is_error
    assert "未找到" in result.content


@pytest.mark.asyncio
async def test_tool_timeout():
    """测试工具超时保护。"""
    import asyncio
    import app.tools.weather_tool  # noqa
    from app.tools.base import tool_registry

    weather_tool = tool_registry.get("get_weather")
    original_execute = weather_tool.execute

    async def slow_execute(**kwargs):
        await asyncio.sleep(100)  # 模拟超时

    weather_tool.execute = slow_execute

    with patch("app.tools.base.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await tool_registry.execute("get_weather", "tool_003", {"city": "北京"})

    weather_tool.execute = original_execute
    assert result.is_error
    assert "timed out" in result.content


# ── DB 工具参数校验测试 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_db_tool_missing_required_param():
    """action=order_status 缺少 order_id 时应返回错误。"""
    import app.tools.database_tool  # noqa
    from app.tools.base import tool_registry

    result = await tool_registry.execute(
        "query_database", "tool_004", {"action": "order_status"}
    )
    assert result.is_error
    assert "order_id" in result.content


# ── SSE 序列化测试 ────────────────────────────────────────────────────────────

def test_sse_event_serialization():
    """验证 SSE 帧格式符合协议规范。"""
    from app.models.schemas import SSEEvent, SSEEventType, TextDeltaData

    event = SSEEvent(
        type=SSEEventType.TEXT_DELTA,
        data=TextDeltaData(text="hello").model_dump(),
    )
    sse_str = event.to_sse()

    assert sse_str.startswith("data: ")
    assert sse_str.endswith("\n\n")

    payload = json.loads(sse_str[len("data: "):].strip())
    assert payload["type"] == "text_delta"
    assert payload["data"]["text"] == "hello"
