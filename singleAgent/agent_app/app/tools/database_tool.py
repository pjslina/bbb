"""
app/tools/database_tool.py
示例工具：MySQL 查询工具，按业务场景路由不同 SQL。

设计模式：
- 单一工具类，通过 action 参数区分查询场景
- SQL 集中在类内统一管理，禁止外部传入原始 SQL（防止注入）
- 参数化查询，所有用户输入通过绑定参数传递
- 查询结果自动限制行数，防止超大结果集
"""
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.db.mysql import get_db_pool
from app.tools.base import BaseTool, ToolResult, tool_registry

logger = get_logger(__name__)

MAX_ROWS = 50  # 单次查询最大返回行数


class DatabaseQueryTool(BaseTool):
    name = "query_database"
    description = (
        "查询业务数据库，支持以下场景：\n"
        "- order_status：根据订单号查询订单状态\n"
        "- user_orders：查询某用户的最近订单列表\n"
        "- product_stock：查询商品库存\n"
        "- order_stats：查询订单统计（按日期范围）\n"
        "当用户询问订单、库存、用户信息等业务数据时使用此工具。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["order_status", "user_orders", "product_stock", "order_stats"],
                "description": "查询场景类型",
            },
            "order_id": {
                "type": "string",
                "description": "订单号，action=order_status 时必填",
            },
            "user_id": {
                "type": "string",
                "description": "用户 ID，action=user_orders 时必填",
            },
            "product_id": {
                "type": "string",
                "description": "商品 ID，action=product_stock 时必填",
            },
            "start_date": {
                "type": "string",
                "description": "开始日期 YYYY-MM-DD，action=order_stats 时使用",
            },
            "end_date": {
                "type": "string",
                "description": "结束日期 YYYY-MM-DD，action=order_stats 时使用",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "返回记录数上限",
            },
        },
        "required": ["action"],
    }

    # ── SQL 集中管理（禁止外部传入原始 SQL）─────────────────────────────────
    _QUERIES: Dict[str, str] = {
        "order_status": """
            SELECT
                o.order_id,
                o.status,
                o.total_amount,
                o.created_at,
                o.updated_at,
                u.username AS customer_name
            FROM orders o
            LEFT JOIN users u ON o.user_id = u.user_id
            WHERE o.order_id = %(order_id)s
            LIMIT 1
        """,

        "user_orders": """
            SELECT
                o.order_id,
                o.status,
                o.total_amount,
                o.item_count,
                o.created_at
            FROM orders o
            WHERE o.user_id = %(user_id)s
            ORDER BY o.created_at DESC
            LIMIT %(limit)s
        """,

        "product_stock": """
            SELECT
                p.product_id,
                p.name AS product_name,
                p.sku,
                i.quantity AS stock_quantity,
                i.warehouse_location,
                p.price
            FROM products p
            LEFT JOIN inventory i ON p.product_id = i.product_id
            WHERE p.product_id = %(product_id)s
            LIMIT 1
        """,

        "order_stats": """
            SELECT
                DATE(created_at)     AS date,
                COUNT(*)             AS total_orders,
                SUM(total_amount)    AS total_revenue,
                AVG(total_amount)    AS avg_order_value,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_orders
            FROM orders
            WHERE created_at BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT %(limit)s
        """,
    }

    # ── 参数校验（调用前） ────────────────────────────────────────────────────
    _REQUIRED_PARAMS: Dict[str, List[str]] = {
        "order_status": ["order_id"],
        "user_orders":  ["user_id"],
        "product_stock": ["product_id"],
        "order_stats": [],  # start_date/end_date 有默认值
    }

    def _build_params(self, action: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """构造参数字典，补充默认值。"""
        params = {k: v for k, v in kwargs.items() if v is not None}
        params.setdefault("limit", min(kwargs.get("limit", 10), MAX_ROWS))

        if action == "order_stats":
            params.setdefault("start_date", "2024-01-01")
            params.setdefault("end_date", "2099-12-31")

        return params

    def _validate_params(self, action: str, params: Dict[str, Any]) -> Optional[str]:
        """返回 None 表示合法，否则返回错误消息。"""
        required = self._REQUIRED_PARAMS.get(action, [])
        missing = [r for r in required if not params.get(r)]
        if missing:
            return f"action='{action}' 缺少必填参数：{', '.join(missing)}"
        return None

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        sql = self._QUERIES.get(action)
        if not sql:
            return ToolResult(content=f"未知的 action：'{action}'", is_error=True)

        params = self._build_params(action, kwargs)
        err = self._validate_params(action, params)
        if err:
            return ToolResult(content=err, is_error=True)

        logger.info("db_query", action=action, params={k: v for k, v in params.items() if k != "password"})

        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql, params)
                    columns = [col[0] for col in cursor.description] if cursor.description else []
                    rows = await cursor.fetchall()

            if not rows:
                return ToolResult(content={"message": "未查询到相关数据", "action": action, "rows": []})

            result_rows = [dict(zip(columns, row)) for row in rows]
            # 日期/Decimal 等类型序列化
            for row in result_rows:
                for k, v in row.items():
                    if hasattr(v, "isoformat"):   # datetime / date
                        row[k] = v.isoformat()
                    elif hasattr(v, "__float__"):  # Decimal
                        row[k] = float(v)

            logger.info("db_query_success", action=action, row_count=len(result_rows))
            return ToolResult(content={"action": action, "rows": result_rows, "count": len(result_rows)})

        except Exception as e:
            logger.exception("db_query_error", action=action, exc_info=e)
            return ToolResult(content=f"数据库查询失败：{e}", is_error=True)


# 自动注册
tool_registry.register(DatabaseQueryTool())
