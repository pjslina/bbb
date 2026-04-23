架构核心设计
? ReAct 循环（chat_agent.py）

用户消息 → LLM → stop_reason=="tool_use"
    → 执行工具 → tool_result 追加 messages
    → 再次调用 LLM → ... → stop_reason=="end_turn"
    
每一轮都 yield SSEEvent，前端实时感知进度。
? SSE 事件类型分层

事件含义前端用途text_delta流式文字打字机效果tool_use_start工具开始显示"正在查询..."tool_use_end工具完成折叠显示工具结果message_stop全部结束Token 计数、关闭连接error错误友好提示

? 扩展新工具（3步）

继承 BaseTool，填写 name / description / input_schema / execute()
文件末尾 tool_registry.register(MyTool())
在 main.py lifespan 中 import app.tools.my_tool  — 完成

? 生产级关键实践

SQL 注入防护：所有 SQL 集中在工具类内，参数化查询，禁止外部传入原始 SQL
超时保护：每个工具 asyncio.wait_for(timeout=30) 硬限制
最大迭代：LLM_MAX_ITERATIONS=10 防止 Agent 无限循环
连接池：aiomysql 异步连接池，兼容 openGauss（PostgreSQL 协议兼容）
结构化日志：structlog 绑定 session_id + trace_id，生产输出 JSON 便于采集