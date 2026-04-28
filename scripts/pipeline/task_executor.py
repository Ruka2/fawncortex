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

from agentscope.agent import AgentBase
from agentscope.message import Msg

from .task_plan import TaskPlan, TaskNode, NodeType, ExecutionMode    # todo: ExecutionMode暂时没用到，待办
from .output_scheduler import OutputScheduler
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
            results = await asyncio.gather(*self._current_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ 后台任务 #{i} 执行失败: {result}")
                    import traceback
                    traceback.print_exception(type(result), result, result.__traceback__)
            self._current_tasks.clear()

        # print("✅ 任务计划执行完毕")

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
        agent = self.agents.get(node.agents[0])  # fixme: 目前只能设定单智能体，后续补充多智能体序列
        if not agent:
            print(f"⚠️  节点 '{node.name}' 找不到 Agent '{node.agents}'")
            return

        # 先注入 brain 洞察（如果 SharedContext 中有数据）
        # 这样 quick_chat 回复时就能看到上一轮的深度分析结果
        ctx = self.shared_ctx.peek()
        if hasattr(agent, "inject_context"):
            agent.inject_context(ctx)

        reply = await agent.reply(user_msg)
        chat_text = reply.get_text_content()

        # 记录本轮 quick_chat 的 assistant 输出，供 deep_think 判断是否需要预注入
        self._last_quick_chat_reply = chat_text

        # 使用 brain 建议的表情，或回退到默认（无）
        emotion_text = ctx.get("suggested_emotion", "")
        await self.scheduler.schedule(chat_text, emotion_text, node.agents)

    async def _exec_emotion_action(self, node: TaskNode, user_msg: Msg) -> None:
        """表情动作节点：调用 emotion_agent 输出表情。"""
        agent = self.agents.get(node.agents[0])  # fixme: 目前只能设定单智能体，后续补充多智能体序列
        if not agent:
            print(f"⚠️  节点 '{node.name}' 找不到 Agent '{node.agents}'")
            return
        reply = await agent(user_msg)
        emotion_text = reply.get_text_content()
        await self.scheduler.schedule("", emotion_text, node.agents)


    async def _exec_summary_chat(self, node: TaskNode, user_msg: Msg) -> None:
        """总结回复节点：基于大脑洞察后的正式回复。

        在 deep_think 之后执行，此时 SharedContext 已被更新。
        会将 brain 的洞察注入 chat_agent 的 prompt 中。
        """
        agent = self.agents.get(node.agents[0])  # fixme: 目前只能设定单智能体，后续补充多智能体序列
        if not agent:
            print(f"⚠️  节点 '{node.name}' 找不到 Agent '{node.agents}'")
            return

        # 如果不需要再响应
        if self.shared_ctx.ignore_requested:
            return

        # 将 brain 洞察上下文注入对话智能体
        ctx = self.shared_ctx.peek()
        if hasattr(agent, "inject_context"):   # 如果是chat_agent的话才可以注入，因为只有他有inject_context函数
            agent.inject_context(ctx)
            
        # ── Debug: 打印 quick_chat 调用前的 SharedContextData ──
        # print(f"\n{'='*60}")
        # print(f"[SharedContext DEBUG] summary_chat 调用大模型前")
        # print(f"{'='*60}")
        # for k, v in ctx.items():
        #     v_str = str(v)
        #     display = v_str[:300] + ("..." if len(v_str) > 300 else "")
        #     print(f"  {k}: {display}")
        # print(f"{'='*60}\n")

        # ── Debug: 打印 quick_chat 调用前的 SharedContextData ──
        # print(f"\n{'='*60}")
        # print(f"[SharedContext DEBUG] summary_chat 调用大模型前")
        # print(f"{'='*60}")
        # for k, v in ctx.items():
        #     v_str = str(v)
        #     display = v_str[:300] + ("..." if len(v_str) > 300 else "")
        #     print(f"  {k}: {display}")
        # print(f"{'='*60}\n")

        reply = await agent.reply(user_msg)
        chat_text = reply.get_text_content()

        # 已完成brain_agent的summary节点，清空大脑智能体需要使用的上下文
        self._last_quick_chat_reply = ""

        # 使用 brain 建议的表情，或回退到默认
        emotion_text = ctx.get("suggested_emotion", "")
        await self.scheduler.schedule(chat_text, emotion_text, node.agents)
        
        

    async def _exec_deep_think(self, node: TaskNode, user_msg: Msg) -> None:
        """大脑深度思考节点（阻塞）。"""
        agent = self.agents.get(node.agents[0])  # fixme: 目前只能设定单智能体，后续补充多智能体序列
        if not agent:
            print(f"⚠️  节点 '{node.name}' 找不到 Agent '{node.agents}'")
            return

        # 只有当 quick_chat 已经先响应完成，才把其 assistant 回复预注入 brain
        # 避免 deep_think 在 quick_chat 之前执行时读到旧数据
        assistant_text = self._last_quick_chat_reply

        # 使用 think_with_context：内部按 user → assistant 顺序预填充短期记忆，然后执行 ReAct
        result_dict = await agent.think_with_context(user_msg, assistant_text)

        # 转回 JSON 字符串供 _parse_brain_output 解析
        insight_text = json.dumps(result_dict, ensure_ascii=False, indent=2)

        # 解析大脑输出（JSON格式）
        await self._parse_brain_output(insight_text)



    # -------------------------------------------------------------------------
    # 工具类
    
    # 大脑输出解析
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
