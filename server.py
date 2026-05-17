"""
FawnCortex Web UI 后端服务器（server.py）
=========================================
基于 FastAPI + WebSocket 实现，职责：
1. 托管前端静态文件（fawncortex/components/webui/static/）
2. 提供 WebSocket 实时通信通道（支持多客户端广播）
3. 桥接 FawnCortexEngine 事件系统与 WebSocket 客户端
4. 新增 /live 页面路由（OBS 推流 + 聊天交互）

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

from main8_server import FawnCortexEngine, create_engine

# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(title="FawnCortex智能体对话系统", version="1.0.0")

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "fawncortex" / "components" / "webui" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 全局引擎实例（单例）
_engine: Optional[FawnCortexEngine] = None
_engine_lock = asyncio.Lock()


async def get_engine() -> FawnCortexEngine:
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
    """返回主页面 HTML（监控面板）。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse(content="<h1>FawnCortex智能体对话系统</h1><p>index.html not found</p>", status_code=404)


@app.get("/live", response_class=HTMLResponse)
async def live():
    """返回直播交互页面 HTML（聊天 + OBS 推流画面）。"""
    html_path = STATIC_DIR / "live.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse(content="<h1>live.html not found</p>", status_code=404)


# =============================================================================
# WebSocket 路由（支持多客户端广播）
# =============================================================================

class ConnectionManager:
    """WebSocket 连接管理器（支持多客户端广播）。"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: dict):
        """广播消息到所有已连接的客户端。"""
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception as e:
                print(f"[WebSocket] send error: {e}")
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 主入口。

    客户端消息格式：
        {"type": "user_input", "text": "你好"}

    服务端消息格式：
        {"type": "chat_message", "data": {"text": "...", ...}}
    """
    await manager.connect(websocket)

    engine = await get_engine()

    # 注册引擎事件处理器，只转发给当前 WebSocket 连接
    # 【关键修复】之前用 manager.send_json() 广播给所有连接，
    # 导致多窗口时每个事件被重复发送。改为只发给当前连接。
    async def on_engine_event(event: dict):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

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
        "tts_started",
        "tts_finished",
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
        await websocket.send_json({
            "type": "system",
            "data": {"status": "connected", "message": "✅ 已连接到对话系统"}
        })

        while True:
            # 接收客户端消息
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
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
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "输入不能为空"}
                    })

            elif msg_type == "reset":
                await engine.reset_memory()
                await websocket.send_json({
                    "type": "system",
                    "data": {"status": "reset_ack", "message": "记忆已清空"}
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "data": {}})

            elif msg_type == "set_names":
                agent_name = msg.get("agent_name", "").strip()
                user_name = msg.get("user_name", "").strip()
                if agent_name and user_name:
                    await engine.update_names(agent_name, user_name)
                    await websocket.send_json({
                        "type": "system",
                        "data": {
                            "status": "names_updated",
                            "message": f"名称已更新: Agent={agent_name}, User={user_name}",
                        },
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Agent 名称和用户名不能为空"},
                    })

            else:
                await websocket.send_json({
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
        manager.disconnect(websocket)


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FAWNCORTEX_PORT", "8259"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
