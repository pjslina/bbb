场景                    SSE        WebSocket
───────────────────────────────────────────
网页端开发体验          ? 优      ? 良
小程序开发体验          ? 模拟   ? 原生支持
双端统一协议            ? 不能   ? 能
服务端实现复杂度        低        中
运维复杂度              低        中

前端要求统一用 WebSocket，完全合理，直接支持。调整方案如下。

调整后架构
┌──────────────────────────┐
  网页端  ──WS──────? │                          │
                      │   统一 WebSocket 接口     │ ──? Agent / LLM / Tools
  小程序  ──WS──────? │   ws://host/api/v1/chat  │
                      │                          │
                      └──────────────────────────┘
                      
唯一接口：
WS  /api/v1/chat      # 双端统一，全程 WebSocket

消息协议设计（重点）
WebSocket 是双向的，必须定义清楚上下行消息格式，否则后期会很乱。
客户端 → 服务端（上行）
// 发起对话
{
  "type": "chat",
  "session_id": "uuid-or-null",   // null 表示新会话
  "message": "用户输入内容",
  "meta": {
    "client": "web" | "miniprogram"
  }
}

// 中断生成（用户点停止）
{
  "type": "abort",
  "session_id": "uuid"
}

// 心跳
{
  "type": "ping"
}


服务端 → 客户端（下行）
// 会话已建立
{ "type": "session_created", "session_id": "uuid" }

// 思考/工具调用状态
{ "type": "thinking",  "content": "正在搜索..." }
{ "type": "tool_call", "tool": "search", "status": "running" }
{ "type": "tool_call", "tool": "search", "status": "done", "result": {} }

// 流式 token（最高频）
{ "type": "token", "content": "你" }
{ "type": "token", "content": "好" }

// 本轮结束
{ "type": "done", "session_id": "uuid", "usage": { "tokens": 128 } }

// 错误
{ "type": "error", "code": 4001, "message": "session 不存在" }

// 心跳回应
{ "type": "pong" }

Python 实现（FastAPI + WebSocket）

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import AsyncGenerator
import asyncio, json, uuid

app = FastAPI()

# ── 核心 Agent 逻辑（与协议无关，可复用）──────────────────────────────
async def run_agent(session_id: str, message: str) -> AsyncGenerator[dict, None]:
    """yield 事件字典，不关心下层是 SSE 还是 WebSocket"""
    
    yield {"type": "thinking", "content": "分析问题中..."}
    
    # 模拟工具调用
    yield {"type": "tool_call", "tool": "search", "status": "running"}
    await asyncio.sleep(0.5)
    yield {"type": "tool_call", "tool": "search", "status": "done", "result": {"hits": 3}}
    
    # 流式 token 输出
    answer = "这是 Agent 根据搜索结果生成的回答，逐字输出。"
    for char in answer:
        yield {"type": "token", "content": char}
        await asyncio.sleep(0.02)
    
    yield {"type": "done", "session_id": session_id, "usage": {"tokens": len(answer)}}


# ── WebSocket 连接管理 ────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        # session_id -> WebSocket
        self.active: dict[str, WebSocket] = {}
        # session_id -> cancel flag
        self.abort_flags: dict[str, bool] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws
        self.abort_flags[session_id] = False

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)
        self.abort_flags.pop(session_id, None)

    def request_abort(self, session_id: str):
        self.abort_flags[session_id] = True

    def should_abort(self, session_id: str) -> bool:
        return self.abort_flags.get(session_id, False)


manager = ConnectionManager()


# ── WebSocket 主接口 ──────────────────────────────────────────────────
@app.websocket("/api/v1/chat")
async def chat_ws(websocket: WebSocket):
    conn_id = str(uuid.uuid4())   # 连接级别 ID
    await manager.connect(conn_id, websocket)
    
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # 心跳
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # 中断当前生成
            if msg_type == "abort":
                session_id = msg.get("session_id", conn_id)
                manager.request_abort(session_id)
                continue

            # 正常对话
            if msg_type == "chat":
                session_id = msg.get("session_id") or str(uuid.uuid4())
                manager.abort_flags[session_id] = False

                await websocket.send_json({
                    "type": "session_created",
                    "session_id": session_id
                })

                # 流式推送 Agent 输出
                async for event in run_agent(session_id, msg["message"]):
                    if manager.should_abort(session_id):
                        await websocket.send_json({
                            "type": "aborted",
                            "session_id": session_id
                        })
                        break
                    await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "code": 500, "message": str(e)})
    finally:
        manager.disconnect(conn_id)
        
前端对接示意
网页端
const ws = new WebSocket('wss://your-api.com/api/v1/chat')

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'chat',
    session_id: null,
    message: userInput,
    meta: { client: 'web' }
  }))
}

ws.onmessage = ({ data }) => {
  const event = JSON.parse(data)
  switch (event.type) {
    case 'session_created': saveSessionId(event.session_id); break
    case 'token':           appendToken(event.content); break
    case 'tool_call':       showToolStatus(event); break
    case 'done':            finishMessage(); break
    case 'error':           showError(event.message); break
  }
}

小程序端（原生支持，无需任何模拟）
const ws = wx.connectSocket({
  url: 'wss://your-api.com/api/v1/chat'
})

ws.onOpen(() => {
  ws.send({ data: JSON.stringify({
    type: 'chat',
    session_id: null,
    message: userInput,
    meta: { client: 'miniprogram' }
  })})
})

ws.onMessage(({ data }) => {
  const event = JSON.parse(data)
  // 与网页端完全一致的处理逻辑
  handleAgentEvent(event)
})

两端事件处理逻辑完全相同，可以抽成公共模块复用。

需要额外注意的生产细节
Nginx 配置（必须升级协议头）

location /api/v1/chat {
    proxy_pass         http://backend;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;     # 关键
    proxy_set_header   Connection "upgrade";       # 关键
    proxy_read_timeout 300s;    # Agent 响应慢，超时要放长
    proxy_send_timeout 300s;
}

心跳保活（防止连接被中间件断开）
# 服务端定期发 ping，防止小程序/网关 idle 超时断连
async def heartbeat(websocket: WebSocket):
    while True:
        await asyncio.sleep(30)
        try:
            await websocket.send_json({"type": "ping"})
        except Exception:
            break

# 在 connect 时启动
asyncio.create_task(heartbeat(websocket))

最终决策表
决策点方案接口数量统一一个 ws://host/api/v1/chat协议WebSocket（双端原生支持，前端开发体验好）消息格式JSON，type 字段区分事件类型中断支持abort 消息 + 服务端 cancel flag会话管理session_id 由服务端生成，双端通用Nginx必须配置 Upgrade 头 + 放长超时心跳服务端每 30s 主动 ping
前端同事的判断完全正确，WebSocket 在这个场景下是更合适的团队选择。

