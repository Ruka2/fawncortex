"""
聊天室控制器（Chatroom Controller）
====================================
Event-Driven Multi-Agent Chatroom 的核心控制层。

本文件实现三大基础设施：
1. EventBus       — 轻量级 Pub-Sub 异步事件总线
2. BackgroundBrainAgent — BrainAgent 的后台常驻包装器（支持打断）
3. ReflectionAgent — 元认知审判官，控制全场对话时机

【架构标注说明】
- 【基础设施】：已完整实现，可直接使用
- 【策略占位】：仅提供规则骨架/启发式，需你后续替换为 LLM 驱动或精细化规则
- 【扩展点】：标注了你未来可能扩展的位置
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel


# =============================================================================
# 【基础设施】事件类型定义
# =============================================================================

@dataclass
class UserInputEvent:
    """用户输入事件：由 EventBus 广播给所有订阅 Agent。"""
    topic: str = "user.input"
    msg: Optional[Msg] = None
    round_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThoughtEvent:
    """BrainAgent 思考完成事件。"""
    topic: str = "brain.thought"
    content: str = ""               # 思考文本摘要（BrainAgent 洞察文本）
    raw_data: dict = field(default_factory=dict)  # BrainAgent 原始输出数据（含 insight 字段）
    round_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterventionEvent:
    """ReflectionAgent 发布的干预事件。

    action 枚举：
    - "none"      : 不干预
    - "inject"    : 将 Brain 思考结果注入 ChatAgent，触发插话/补充
    - "stop_brain": Brain 过度思考，强制打断
    - "clarify"   : 请求 ChatAgent 向用户发起澄清追问
    """
    topic: str = "reflection.intervention"
    action: str = "none"
    target: str = ""                # 目标 Agent 名称（如 "BrainAgent", "ChatAgent"）
    payload: str = ""               # 携带的指令/内容
    round_id: int = 0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# 【基础设施】EventBus — 轻量级异步事件总线（Pub-Sub）
# =============================================================================

class EventBus:
    """轻量级异步事件总线。

    每个 Agent 注册一个专属 asyncio.Queue，EventBus 按 topic 路由事件。
    与 AgentScope MsgHub 的区别：
    - MsgHub 是同步上下文内的顺序发言广播；
    - EventBus 是真正的异步、跨上下文、持久化的 Pub-Sub。
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._subscribers: dict[str, list[str]] = defaultdict(list)
        self._running = True

    # ── 注册与订阅 ──

    def register_agent(self, agent_name: str) -> asyncio.Queue:
        """为指定 Agent 注册一个专属接收队列，返回该队列供 Agent 自行消费。"""
        q = asyncio.Queue()
        self._queues[agent_name] = q
        return q

    def subscribe(self, agent_name: str, topics: list[str]) -> None:
        """让 agent_name 订阅指定 topic 列表。"""
        for topic in topics:
            if agent_name not in self._subscribers[topic]:
                self._subscribers[topic].append(agent_name)

    def unsubscribe(self, agent_name: str, topics: list[str]) -> None:
        """取消订阅。"""
        for topic in topics:
            if agent_name in self._subscribers[topic]:
                self._subscribers[topic].remove(agent_name)

    # ── 发布 ──

    async def publish(self, topic: str, event: Any) -> None:
        """向所有订阅该 topic 的 Agent 发送事件（非阻塞）。"""
        for agent_name in self._subscribers.get(topic, []):
            if agent_name in self._queues:
                await self._queues[agent_name].put(event)

    async def stop(self) -> None:
        """停止总线（不会清空已有队列，只是标记状态）。"""
        self._running = False


# =============================================================================
# 【基础设施】BackgroundBrainAgent — BrainAgent 的后台常驻包装器
# =============================================================================

class BackgroundBrainAgent:
    """BrainAgent 的后台常驻包装器。

    【设计意图】
    BrainAgent 本身是"调用-等待返回"的同步式 Agent（reply() → 阻塞直到思考完成）。
    本包装器将其转化为：
    1. 常驻后台的 asyncio.Task（通过 run() 启动）
    2. 通过 EventBus 接收 UserInputEvent（非阻塞投递）
    3. 支持 cancel() 外部打断（ReflectionAgent 调用）

    【打断机制】
    - 用户输入新消息时，自动取消上一轮未完成的思考
    - ReflectionAgent 判定过度思考时，也可调用 cancel()
    - 取消通过 asyncio.Task.cancel() 实现，触发 BrainAgent.interrupt()
    """

    def __init__(self, brain_agent: Any, bus: EventBus) -> None:
        self.brain = brain_agent          # 原始的 BrainAgent 实例
        self.bus = bus
        self.input_queue = bus.register_agent("BrainAgent")
        bus.subscribe("BrainAgent", ["user.input"])

        # 输出队列：外部通过 await self.output_queue.get() 获取思考结果
        self.output_queue: asyncio.Queue = asyncio.Queue()

        self.current_task: Optional[asyncio.Task] = None
        self.running = True
        self.last_thought: Optional[ThoughtEvent] = None


    # ── 后台主循环 ──
    async def run(self) -> None:
        """常驻后台主循环：持续监听 EventBus 输入。"""
        while self.running:
            event = await self.input_queue.get()

            if isinstance(event, UserInputEvent):
                # 【扩展点】未来可在此加入更多事件类型（如 CognitiveControlEvent）
                await self._on_user_input(event)


    async def _on_user_input(self, event: UserInputEvent) -> None:
        """处理新用户输入：取消旧思考，启动新思考。"""
        # 1. 安全取消上一轮未完成的思考
        await self._cancel_current_think()

        # 2. 启动新一轮思考任务
        self.current_task = asyncio.create_task(
            self._think(event),
            name=f"brain_think_r{event.round_id}",
        )


    # ── 单次思考（可被 cancel 打断）──
    async def _think(self, event: UserInputEvent) -> None:
        """单次深度思考任务。

        注意：此函数运行在独立的 asyncio.Task 中，可以被外部 cancel() 打断。
        被打断时会抛出 asyncio.CancelledError，需正确捕获并清理。
        """
        try:
            # 调用 BrainAgent.think() 获取完整输出（含 insight + retrieved_memories）
            data = await self.brain.think(event.msg)

            # BrainAgent 输出自然语言洞察文本（含相关记忆）
            text = data.get("insight", "")
            raw_data = data  # 直接使用完整字典

            # 构造 ThoughtEvent
            thought = ThoughtEvent(
                content=text,
                raw_data=raw_data,
                round_id=event.round_id,
            )
            self.last_thought = thought

            # 【核心】通知外部：思考完成
            await self.output_queue.put({
                "status": "completed",
                "thought": thought,
            })

            # 【核心】向总线广播，供其他 Agent 监听
            await self.bus.publish("brain.thought", thought)

        except asyncio.CancelledError:
            # 被打断时的清理逻辑
            await self.output_queue.put({
                "status": "cancelled",
                "thought": None,
            })
            # 【扩展点】可在此保存"部分中间结果"用于后续恢复
            raise  # 必须重新抛出，让 asyncio 正确回收 Task


    # ── 内部打断接口 ──
    async def _cancel_current_think(self) -> None:
        """安全取消当前思考任务（内部使用）。"""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
            finally:
                self.current_task = None


    # ── 外部打断接口 ──
    def cancel(self) -> None:
        """外部调用：立即打断当前思考（由 ReflectionAgent 调用）。

        注意：这是同步接口，直接对 Task 调用 cancel()，不会等待 Task 真正结束。
        如果需要等待清理完成，请使用 await self._cancel_current_think()。
        """
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()



    @property
    def status(self) -> str:
        """当前状态：idle | thinking | completed（供 ReflectionAgent 查询）。"""
        if self.current_task is None:
            return "idle"
        if self.current_task.done():
            return "completed"
        return "thinking"

    async def stop(self) -> None:
        """优雅停止后台循环。"""
        self.running = False
        await self._cancel_current_think()


