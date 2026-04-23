"""
app/tools/weather_tool.py
示例工具：调用 OpenWeatherMap API 查询天气。
演示了：带重试的 HTTP 调用、参数校验、错误处理。
"""
from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import BaseTool, ToolResult, tool_registry

logger = get_logger(__name__)


class WeatherTool(BaseTool):
    name = "get_weather"
    description = (
        "查询指定城市的实时天气信息，包括温度、湿度、天气状况和风速。"
        "当用户询问天气时使用此工具。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，支持中文或英文，例如：北京、Shanghai",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "default": "metric",
                "description": "温度单位：metric=摄氏度，imperial=华氏度",
            },
        },
        "required": ["city"],
    }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def _fetch_weather(self, city: str, units: str) -> Dict[str, Any]:
        params = {
            "q": city,
            "appid": settings.weather_api_key,
            "units": units,
            "lang": "zh_cn",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.weather_api_base_url}/weather",
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def execute(self, city: str, units: str = "metric") -> ToolResult:
        try:
            data = await self._fetch_weather(city, units)
            result = {
                "city": data.get("name", city),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"] if data.get("weather") else "",
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "unit": "°C" if units == "metric" else "°F",
            }
            return ToolResult(content=result)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return ToolResult(content=f"城市 '{city}' 未找到，请检查城市名称是否正确。", is_error=True)
            elif e.response.status_code == 401:
                return ToolResult(content="天气 API 密钥无效。", is_error=True)
            return ToolResult(content=f"天气 API 请求失败：HTTP {e.response.status_code}", is_error=True)
        except httpx.TimeoutException:
            return ToolResult(content="天气 API 请求超时，请稍后重试。", is_error=True)
        except Exception as e:
            logger.exception("weather_tool_error", exc_info=e)
            return ToolResult(content=f"获取天气信息时发生错误：{e}", is_error=True)


# 自动注册
tool_registry.register(WeatherTool())
