"""
任务执行器（TaskExecutor）
==========================
负责任务计划（TaskPlan）的顺序执行，处理阻塞/非阻塞节点、
中断、重排（replan）等复杂编排逻辑。

与 AgentScope 的关系：
- 参考了 Pipeline 的并发/顺序执行语义
- 但增加了动态重排、中断、优先级插队等编排能力
"""

import json

import asyncio
from typing import Optional

from agentscope.agent import AgentBase
from agentscope.message import Msg

from .task_plan import TaskPlan, TaskNode, NodeType, ExecutionMode
from .output_scheduler import OutputScheduler, Priority
from ..shared.shared_context import SharedContext


class TaskExecutor:
    """任务执行引擎。

    Args:
        agents: Agent 字典，key 为 agent_name。
        scheduler: 输出调度器（负责 TTS/VTS）。
        shared_ctx: 公共信息域。
    """

    def __init__(
        self,
        agents: dict[str, AgentBase],
        scheduler: OutputScheduler,
        shared_ctx: SharedContext,
    ):
        self.agents = agents
        self.scheduler = scheduler
        self.shared_ctx = shared_ctx
        self._current_tasks: list[asyncio.Task] = []
        self._running = True
        self._last_quick_chat_reply: str = ""  # 记录本轮 quick_chat 的 assistant 输出

    async def execute(self, plan: TaskPlan, user_msg: Msg) -> None:
        """顺序执行任务计划。

        Args:
            plan: 任务计划。
            user_msg: 当前轮次的用户输入消息。
        """
        self._running = True
        self._last_quick_chat_reply = ""  # 每轮重置
        idx = 0

        while idx < len(plan.nodes) and self._running:
            node = plan.nodes[idx]

            # 执行当前节点
            if node.blocking:
                try:
                    await self._execute_node(node, user_msg)
                except asyncio.CancelledError:
                    print(f"🔇 节点 '{node.name}' 被打断")
                    break
            else:
                task = asyncio.create_task(self._execute_node(node, user_msg))
                self._current_tasks.append(task)

            idx += 1

        # 等待所有非阻塞后台任务完成
        if self._current_tasks:
            await asyncio.gather(*self._current_tasks, return_exceptions=True)
            self._current_tasks.clear()

        print("✅ 任务计划执行完毕")

    async def interrupt(self) -> None:
        """中断当前执行。

        用户新输入到达时调用：
        1. 停止接受新节点
        2. 取消所有正在运行的后台任务
        3. 打断输出调度器
        """
        self._running = False
        for task in self._current_tasks:
            if not task.done():
                task.cancel()
        if self._current_tasks:
            await asyncio.gather(*self._current_tasks, return_exceptions=True)
        self._current_tasks.clear()
        await self.scheduler.interrupt()
        print("🔴 任务执行器：已中断")

    async def _execute_node(self, node: TaskNode, user_msg: Msg) -> None:
        """执行单个任务节点。"""
        print(f"▶️  执行节点: {node.name} ({node.node_type.value}, blocking={node.blocking})")

        if node.node_type == NodeType.QUICK_CHAT:
            await self._exec_quick_chat(node, user_msg)
        elif node.node_type == NodeType.DEEP_THINK:
            await self._exec_deep_think(node, user_msg)
        elif node.node_type == NodeType.EMOTION_ACTION:
            await self._exec_emotion_action(node, user_msg)
        elif node.node_type == NodeType.SUMMARY_CHAT:
            await self._exec_summary_chat(node, user_msg)
        else:
            print(f"⚠️  未知节点类型: {node.node_type}")

    # -------------------------------------------------------------------------
    # 节点执行实现
    # -------------------------------------------------------------------------

    async def _exec_quick_chat(self, node: TaskNode, user_msg: Msg) -> None:
        """快速闲聊节点：对话Agent。"""
        chat_agent = self.agents.get("chat")
        
        # 如果没有启动闲聊Agent
        if not chat_agent:
            return
    
        reply = await chat_agent.reply(user_msg)
        chat_text = reply.get_text_content()
        
        # 用于快速对话场景拼凑prompt # fixme: 需要优化提示词
        ctx = self.shared_ctx.peek()
        insight_parts = []
        if ctx.get("retrieved_memories"):
            insight_parts.append(f"检索到的相关记忆：{ctx['retrieved_memories']}")
        if ctx.get("suggested_dialogue_strategy"):
            insight_parts.append(f"对话策略建议：{ctx['suggested_dialogue_strategy']}")
        if ctx.get("user_profile"):
            insight_parts.append(f"用户画像：{ctx['user_profile']}")
        if ctx.get("user_intent"):
            insight_parts.append(f"用户意图：{ctx['user_intent']}")

        if insight_parts and hasattr(chat_agent, "inject_context"):
            context_text = "\n".join(insight_parts)
            chat_agent.inject_context(context_text)
        
        
        # 记录本轮 quick_chat 的 assistant 输出，供 deep_think 判断是否需要预注入
        self._last_quick_chat_reply = chat_text

        # 使用 brain 建议的表情，或回退到默认
        emotion_text = self.shared_ctx.peek().get("suggested_emotion", "neutral")
        await self.scheduler.schedule(chat_text, emotion_text, "chat")

    async def _exec_emotion_action(self, node: TaskNode, user_msg: Msg) -> None:
        """表情动作节点：调用 emotion_agent 输出表情。"""
        emotion_agent = self.agents.get("emotion")
        if not emotion_agent:
            return
        reply = await emotion_agent(user_msg)
        emotion_text = reply.get_text_content()
        await self.scheduler.schedule("", emotion_text, "emotion_action")

    async def _exec_summary_chat(self, node: TaskNode, user_msg: Msg) -> None:
        """总结回复节点：基于大脑洞察后的正式回复。

        在 deep_think 之后执行，此时 SharedContext 已被更新。
        会将 brain 的洞察注入 chat_agent 的 prompt 中。
        """
        chat_agent = self.agents.get("chat")
        
        # 如果没有启动闲聊agent
        if not chat_agent:
            return

        # 如果不需要再响应
        if self.shared_ctx.ignore_requested:
            return

        # 组装 brain 洞察上下文
        ctx = self.shared_ctx.peek()
        insight_parts = []
        if ctx.get("retrieved_memories"):
            insight_parts.append(f"检索到的相关记忆：{ctx['retrieved_memories']}")
        if ctx.get("suggested_dialogue_strategy"):
            insight_parts.append(f"对话策略建议：{ctx['suggested_dialogue_strategy']}")
        if ctx.get("user_profile"):
            insight_parts.append(f"用户画像：{ctx['user_profile']}")
        if ctx.get("user_emotion"):
            insight_parts.append(f"用户情绪：{ctx['user_emotion']}")
        if ctx.get("user_intent"):
            insight_parts.append(f"用户意图：{ctx['user_intent']}")

        if insight_parts and hasattr(chat_agent, "inject_context"):
            context_text = "\n".join(insight_parts)
            chat_agent.inject_context(context_text)
            
        # ── Debug: 打印 quick_chat 调用前的 SharedContextData ──
        print(f"\n{'='*60}")
        print(f"[SharedContext DEBUG] summary_chat 调用大模型前")
        print(f"{'='*60}")
        for k, v in ctx.items():
            v_str = str(v)
            display = v_str[:300] + ("..." if len(v_str) > 300 else "")
            print(f"  {k}: {display}")
        print(f"{'='*60}\n")

        # ── Debug: 打印 quick_chat 调用前的 SharedContextData ──
        print(f"\n{'='*60}")
        print(f"[SharedContext DEBUG] summary_chat 调用大模型前")
        print(f"{'='*60}")
        for k, v in ctx.items():
            v_str = str(v)
            display = v_str[:300] + ("..." if len(v_str) > 300 else "")
            print(f"  {k}: {display}")
        print(f"{'='*60}\n")

        reply = await chat_agent.reply(user_msg)
        chat_text = reply.get_text_content()
        
        # 已完成brain_agent的summary节点，清空大脑智能体需要使用的上下文
        self._last_quick_chat_reply = ""

        # 使用 brain 建议的表情，或回退到默认
        emotion_text = ctx.get("suggested_emotion", "neutral")
        await self.scheduler.schedule(chat_text, emotion_text, "summary_chat")
        
        

    async def _exec_deep_think(self, node: TaskNode, user_msg: Msg) -> None:
        """大脑深度思考节点（阻塞）。"""
        brain_agent = self.agents.get("brain")
        if not brain_agent:
            return

        # 只有当 quick_chat 已经先响应完成，才把其 assistant 回复预注入 brain
        # 避免 deep_think 在 quick_chat 之前执行时读到旧数据
        assistant_text = self._last_quick_chat_reply

        # 使用 think_with_context：内部按 user → assistant 顺序预填充短期记忆，然后执行 ReAct
        result_dict = await brain_agent.think_with_context(user_msg, assistant_text)

        # 转回 JSON 字符串供 _parse_brain_output 解析
        insight_text = json.dumps(result_dict, ensure_ascii=False, indent=2)

        # 解析大脑输出（JSON格式）
        await self._parse_brain_output(insight_text)

    # -------------------------------------------------------------------------
    # 大脑输出解析
    # -------------------------------------------------------------------------

    async def _parse_brain_output(self, text: str) -> None:
        """解析大脑Agent的输出 JSON，更新 SharedContext。

        期望格式（与 brain_agent.py 同步）：
        {
            "clarification": {
                "clarification_reason": "...",
                "clarification_needed": false,
                "clarification_option": "ignore"
            },
            "user_info": {
                "user_profile": "...",
                "user_emotion": "...",
                "user_intent": "..."
            },
            "suggested": {
                "suggested_dialogue_strategy": "...",
                "suggested_emotion": "neutral"
            },
            "retrieved_memories": ["...", "..."]   # ReActAgent 工具调用中检索到的记忆
        }
        """

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"[WARN] BrainAgent 输出不是合法 JSON，已丢弃: {text[:200]}...")
            return

        # 按 brain_agent.py 最新嵌套格式提取字段
        clarification = data.get("clarification", {})
        user_info = data.get("user_info", {})
        suggested = data.get("suggested", {})

        await self.shared_ctx.update(
            version=self.shared_ctx.peek()["version"],
            # --- 澄清 / 任务编排字段 ---
            clarification_needed=clarification.get("clarification_needed", False),
            clarification_option=clarification.get("clarification_option", "ignore"),
            clarification_reason=clarification.get("clarification_reason", ""),
            # --- 用户信息字段 ---
            user_profile=user_info.get("user_profile", ""),
            user_emotion=user_info.get("user_emotion", ""),
            user_intent=user_info.get("user_intent", ""),
            # --- 对话策略字段 ---
            suggested_dialogue_strategy=suggested.get("suggested_dialogue_strategy", ""),
            suggested_emotion=suggested.get("suggested_emotion", ""),
            # --- ReActAgent 检索到的长期记忆（由 brain_agent.py think() 自动附加） ---
            retrieved_memories=data.get("retrieved_memories", []),
        )
