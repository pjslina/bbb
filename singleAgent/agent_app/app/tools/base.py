"""
app/tools/base.py
工具基类与全局注册器。

设计原则：
- 每个工具是一个独立的类，继承 BaseTool
- 工具通过 @tool_registry.register 装饰器自动注册
- AgentTool 协议与 Anthropic tool_use API 对齐，无需额外转换
- 执行失败时返回 ToolResult(is_error=True)，不抛异常，让 Agent 自行决策重试
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from app.core.logging import get_logger

logger = get_logger(__name__)


class ToolResult:
    """工具执行结果。is_error=True 时 content 为错误描述，Agent 可据此决定重试。"""

    def __init__(self, content: Any, is_error: bool = False):
        self.content = content
        self.is_error = is_error

    def to_str(self) -> str:
        import json
        if isinstance(self.content, (dict, list)):
            return json.dumps(self.content, ensure_ascii=False, default=str)
        return str(self.content)

    def __repr__(self) -> str:
        return f"ToolResult(is_error={self.is_error}, content={self.content!r})"


class BaseTool(ABC):
    """所有工具的抽象基类。"""

    # 工具名称（唯一），与 LLM tool_use 中的 name 对应
    name: str
    # 工具描述，传给 LLM，直接影响模型何时调用此工具
    description: str
    # JSON Schema，定义工具入参
    input_schema: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具逻辑。参数来自 LLM tool_use 的 input 字段。"""
        ...

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为 Anthropic API 所需的 tool 定义格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """全局工具注册器，单例模式。"""

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, BaseTool] = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例。"""
        if tool.name in self._tools:
            logger.warning("tool_already_registered", tool_name=tool.name)
        self._tools[tool.name] = tool
        logger.info("tool_registered", tool_name=tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """返回所有工具的 Anthropic API 格式列表，直接传入 client.messages.create(tools=...)。"""
        return [t.to_anthropic_tool() for t in self._tools.values()]

    async def execute(self, tool_name: str, tool_id: str, tool_input: Dict[str, Any]) -> ToolResult:
        """
        执行指定工具。
        - 自动绑定 structlog contextvars（tool_name, tool_id）
        - 超时保护：单个工具最多执行 30 秒
        """
        tool = self.get(tool_name)
        if tool is None:
            from app.core.exceptions import ToolNotFoundError
            raise ToolNotFoundError(f"Tool '{tool_name}' is not registered.")

        logger.info("tool_executing", tool_name=tool_name, tool_id=tool_id, input=tool_input)
        try:
            result = await asyncio.wait_for(tool.execute(**tool_input), timeout=30.0)
            if result.is_error:
                logger.warning("tool_returned_error", tool_name=tool_name, error=result.content)
            else:
                logger.info("tool_succeeded", tool_name=tool_name)
            return result
        except asyncio.TimeoutError:
            logger.error("tool_timeout", tool_name=tool_name)
            return ToolResult(content=f"Tool '{tool_name}' timed out after 30 seconds.", is_error=True)
        except Exception as e:
            logger.exception("tool_unexpected_error", tool_name=tool_name, exc_info=e)
            return ToolResult(content=f"Unexpected error in tool '{tool_name}': {e}", is_error=True)


# 全局单例
tool_registry = ToolRegistry()
