"""
自动任务编排主入口（Auto Task Orchestration）
=================================================
基于 AgentScope 框架的多智能体动态编排系统。

核心流程：
1. 用户输入 → 打断上一轮 → 清空状态
2. 编排智能体（Orchestrator）判断复杂度，生成任务队列
3. TaskExecutor 按序执行队列（阻塞/非阻塞/插队）
4. 大脑智能体（Brain）深度思考，支持反思重排（replan）
5. 输出调度器（OutputScheduler）统一管控 TTS/VTS + 打断

运行方式:
    python main5_planqueue.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import asyncio

import json

# =============================================================================
# AgentScope 基础组件
# =============================================================================
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

# =============================================================================
# 项目自定义组件
# =============================================================================
import config

from components.tts import SiliconFlowCosyVoice
from scripts.agent.memory import create_long_term_memory
from scripts.agent.chat_agent import ChatAgent
from scripts.agent.emotion_agent import EmotionAgent
from scripts.agent.brain_agent import BrainAgent
from scripts.agent.orchestrator_agent import OrchestratorAgent

from scripts.shared.shared_context import SharedContext
from scripts.pipeline.task_plan import TaskNode, TaskPlan
from scripts.pipeline.output_scheduler import OutputScheduler
from scripts.pipeline.task_executor import TaskExecutor

from scripts.tools.search_memory import set_memory_manager


AGENT_NAME = "小花"
USER_NAME = "fafa"

async def main() -> None:
    
    # 初始化输出调度器（TTS + 优先级队列）
    tts = SiliconFlowCosyVoice()
    scheduler = OutputScheduler(tts)
    asyncio.create_task(scheduler.run())
    print("[init] 输出调度器已启动")

    # 初始化长期记忆
    long_term_memory = create_long_term_memory(
        agent_name=AGENT_NAME,
        user_name=USER_NAME,
    )
    set_memory_manager(long_term_memory)
    print(f"[init] 长期记忆已初始化 （history_db: {config.MEM0_HISTORY_DB_PATH}）")

    # 初始化公共信息域
    shared_ctx = SharedContext()
    print(f"[init] 共享上下文信息域已创建")

    # 初始化大模型
    model = OpenAIChatModel(
        model_name=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        stream=config.STREAM,
        client_kwargs={"base_url": config.OPENAI_BASE_URL},
        generate_kwargs={"extra_body": {"enable_thinking": False}},
    )
    print(f"[init] 大模型已初始化 ({config.MODEL_NAME})")

    # 初始化各个智能体（共4个：对话、表情、大脑、任务编排）
    chat_agent = ChatAgent(model=model)
    emotion_agent = EmotionAgent(model=model)
    brain_agent = BrainAgent(model=model, long_term_memory=long_term_memory)
    orchestrator = OrchestratorAgent(model=model)

    agents = {
        "chat": chat_agent,
        "emotion": emotion_agent,
        "brain": brain_agent,
    }
    print(f"[init] 智能体已创建（{list(agents.keys())}）")

    # 初始化任务执行器
    executor = TaskExecutor(agents, scheduler, shared_ctx)
    print("[init] 任务执行器已创建")
    
    
    
    # -------------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------------
    round_num = 0

    while True:
        # 等待用户输入
        try:
            user_input = (
                await asyncio.get_event_loop().run_in_executor(None, input, "👤 你: ")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue


        round_num += 1
        msg = Msg(name="user", content=user_input, role="user")
        print(f"\n--- 第 {round_num} 轮 ---")

        # ── 打断上一轮（用户新输入到达时）──
        await executor.interrupt()
        # await shared_ctx.clear()
        await scheduler.interrupt()

        # 重置各 Agent 的 prompt/memory（每轮独立）
        chat_agent.reset_prompt()
        # 注：InMemoryMemory 会随轮次累积，如需严格每轮隔离可调用 .clear()

        # ── 编排智能体生成任务计划 ──
        print("🎛️  编排智能体正在规划任务队列...")
        plan_dict = await orchestrator.plan(msg)
        print(f"编排智能体输出: {json.dumps(plan_dict, ensure_ascii=False)}")

        # 装载编排智能体的输出结果
        # Orchestrator 返回格式: {"node_list": ["quick_chat", "deep_think", ...]}
        nodes_data = plan_dict.get("node_list", [])
        plan = TaskPlan.from_raw_list(
            raw_nodes=nodes_data,
            version=round_num,
            source="orchestrator",
        )
        print(f"   任务队列: {' → '.join(n.name for n in plan.nodes)}")
        

        # ── 执行任务计划 ──
        await executor.execute(plan, msg)
        print(f"--- 第 {round_num} 轮结束 ---\n")

    # 清理
    await scheduler.stop()
    print("=== 系统已关闭 ===")


if __name__ == "__main__":
    asyncio.run(main())
