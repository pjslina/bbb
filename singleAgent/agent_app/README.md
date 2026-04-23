# Single Agent Application — Production Best Practices

## 项目结构

```
agent_app/
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── core/
│   │   ├── config.py            # 统一配置（Pydantic Settings）
│   │   ├── logging.py           # 结构化日志
│   │   └── exceptions.py        # 全局异常处理
│   ├── api/
│   │   └── v1/
│   │       ├── router.py        # 路由聚合
│   │       └── chat.py          # SSE 流式接口
│   ├── agents/
│   │   ├── base.py              # Agent 基类
│   │   └── chat_agent.py        # 核心 Agent 实现（ReAct 循环）
│   ├── tools/
│   │   ├── base.py              # Tool 基类 & 注册器
│   │   ├── weather_tool.py      # 示例：HTTP API 工具
│   │   └── database_tool.py     # 示例：MySQL 查询工具
│   ├── models/
│   │   └── schemas.py           # 请求/响应 Pydantic 模型
│   └── db/
│       └── mysql.py             # 异步 MySQL 连接池
├── tests/
│   └── test_agent.py
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

## 核心架构决策

| 关注点 | 选型 | 理由 |
|--------|------|------|
| Web 框架 | FastAPI | 原生 async、SSE 支持、自动 OpenAPI |
| LLM 调用 | Anthropic SDK (claude-sonnet-4) | Tool use 原生支持 |
| 流式输出 | SSE (Server-Sent Events) | 标准协议，前端兼容性好 |
| 数据库 | aiomysql + SQLAlchemy Core | 异步，不阻塞事件循环 |
| 配置管理 | pydantic-settings | 类型安全，支持 .env |
| 日志 | structlog | 结构化 JSON 日志，便于采集 |
| 错误处理 | 全局 ExceptionHandler + 自定义异常 | 统一响应格式 |

## 快速启动

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```
