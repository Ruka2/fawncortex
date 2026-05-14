"""
Deerberry Web UI 后端服务器（server.py）
=========================================
基于 FastAPI + WebSocket 实现，职责：
1. 托管前端静态文件（deerberry/components/webui/static/）
2. 提供 WebSocket 实时通信通道
3. 桥接 DeerberryEngine 事件系统与 WebSocket 客户端

启动方式：
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload

或：
    python server.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from main8_server import DeerberryEngine, create_engine


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(title="FawnCortex智能体对话系统", version="1.0.0")

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "deerberry" / "components" / "webui" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 全局引擎实例（单例）
_engine: Optional[DeerberryEngine] = None
_engine_lock = asyncio.Lock()


async def get_engine() -> DeerberryEngine:
    """获取或初始化全局引擎实例。"""
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                _engine = await create_engine()
                await _engine.start()
    return _engine


# =============================================================================
# HTTP 路由
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回主页面 HTML。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse(content="<h1>FawnCortex智能体对话系统</h1><p>index.html not found</p>", status_code=404)


# =============================================================================
# WebSocket 路由
# =============================================================================

class ConnectionManager:
    """WebSocket 连接管理器。"""

    def __init__(self):
        self.active_connection: Optional[WebSocket] = None
        self._engine_handlers: list[str] = []

    async def connect(self, websocket: WebSocket):
        if self.active_connection is not None:
            # 只允许一个客户端连接（简化设计）
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "data": {"message": "另一个客户端已连接，请关闭后重试。"}
            })
            await websocket.close(code=1008)
            return False

        await websocket.accept()
        self.active_connection = websocket
        return True

    def disconnect(self):
        self.active_connection = None

    async def send_json(self, message: dict):
        if self.active_connection is not None:
            try:
                await self.active_connection.send_json(message)
            except Exception as e:
                print(f"[WebSocket] send error: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 主入口。

    客户端消息格式：
        {"type": "user_input", "text": "你好"}

    服务端消息格式：
        {"type": "chat_message", "data": {"text": "...", ...}}
    """
    connected = await manager.connect(websocket)
    if not connected:
        return

    engine = await get_engine()

    # 注册引擎事件处理器，将事件转发到 WebSocket
    async def on_engine_event(event: dict):
        await manager.send_json(event)

    # 订阅所有感兴趣的事件
    event_types = [
        "user_message",
        "chat_message",
        "emotion_update",
        "brain_snapshot",
        "brain_summary",
        "midway_message",
        "reflection_judgment",
        "output_scheduled",
        "tts_text",
        "tts_audio",
        "interrupt",
        "round_start",
        "round_end",
        "error",
        "system",
        "chat_context",
    ]
    for et in event_types:
        engine.on(et, on_engine_event)

    try:
        # 通知客户端已连接
        await manager.send_json({
            "type": "system",
            "data": {"status": "connected", "message": "✅ 已连接到对话系统"}
        })

        while True:
            # 接收客户端消息
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json({
                    "type": "error",
                    "data": {"message": "无效的 JSON 格式"}
                })
                continue

            msg_type = msg.get("type", "")

            if msg_type == "user_input":
                text = msg.get("text", "").strip()
                if text:
                    await engine.send_user_input(text)
                else:
                    await manager.send_json({
                        "type": "error",
                        "data": {"message": "输入不能为空"}
                    })

            elif msg_type == "reset":
                await engine.reset_memory()
                await manager.send_json({
                    "type": "system",
                    "data": {"status": "reset_ack", "message": "记忆已清空"}
                })

            elif msg_type == "ping":
                await manager.send_json({"type": "pong", "data": {}})

            elif msg_type == "set_names":
                agent_name = msg.get("agent_name", "").strip()
                user_name = msg.get("user_name", "").strip()
                if agent_name and user_name:
                    await engine.update_names(agent_name, user_name)
                    await manager.send_json({
                        "type": "system",
                        "data": {
                            "status": "names_updated",
                            "message": f"名称已更新: Agent={agent_name}, User={user_name}",
                        },
                    })
                else:
                    await manager.send_json({
                        "type": "error",
                        "data": {"message": "Agent 名称和用户名不能为空"},
                    })

            else:
                await manager.send_json({
                    "type": "error",
                    "data": {"message": f"未知消息类型: {msg_type}"}
                })

    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[WebSocket] 异常: {e}")
    finally:
        # 【关键修复】注销所有事件处理器，防止刷新页面后消息重复
        for et in event_types:
            engine.emitter.off(et, on_engine_event)
        manager.disconnect()


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DEERBERRY_PORT", "8259"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
