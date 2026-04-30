

# 项目路径根目录定位
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 模型配置表
import config

# 核心基础依赖
import asyncio
import time

# AgentScope 基础依赖
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

# 自定义智能体工具依赖
from scripts.agent.chat_agent import ChatAgent
from scripts.agent.emotion_agent import EmotionAgent
from scripts.agent.brain_agent import BrainAgent
from scripts.agent.orchestrator_agent import OrchestratorAgent

from scripts.pipeline.task_plan import TaskPlan
from scripts.pipeline.output_scheduler import OutputScheduler
from scripts.pipeline.task_executor import TaskExecutor
from scripts.pipeline.shared_context import SharedContext

from scripts.agent.memory import create_long_term_memory

from scripts.tools.search_memory import set_memory_manager

# 外部非智能体执行工具
from components.tts import SiliconFlowCosyVoice

# 日志打印代码
from scripts.logger.latency_tracker import LatencyTracker
from scripts.logger.logger import enable_file_logging


# 基础用户配置
AGENT_NAME = "小花"
USER_NAME = "fafa"


async def main() -> None:
    
    # 启用终端日志收集（所有 print 自动写入 logs/ 目录）
    enable_file_logging()

    # 初始化延迟追踪器（可选依赖，传 None 则关闭计时）
    latency_tracker = LatencyTracker()

    # 初始化TTS
    tts = SiliconFlowCosyVoice(
        api_key=config.TTS_API_KEY,
        api_url=config.TTS_BASE_URL,
        model=config.TTS_MODEL_NAME,
        voice=config.TTS_VOICE,
    )
    print(f"[init] TTS已创建: {config.TTS_MODEL_NAME}, {config.TTS_VOICE}")
    
    # 初始化输出调度器
    scheduler = OutputScheduler(tts, latency_tracker=latency_tracker)
    asyncio.create_task(scheduler.run())
    print("[init] 输出调度器已启动")   # 包含tts和time_tracker

    # 初始化长期记忆
    long_term_memory = create_long_term_memory(
        agent_name=AGENT_NAME,
        user_name=USER_NAME,
        vector_store_path=config.MEM0_VECTOR_STORE_PATH,
        history_db_path=config.MEM0_HISTORY_DB_PATH,
        llm_model_name=config.LLM_MODEL_NAME,
        llm_api_key=config.LLM_API_KEY,
        llm_base_url=config.LLM_BASE_URL,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        embedding_api_key=config.EMBEDDING_API_KEY,
        embedding_base_url=config.EMBEDDING_BASE_URL,
    )
    set_memory_manager(long_term_memory)
    print(f"[init] 长期记忆已初始化: {config.MEM0_HISTORY_DB_PATH}")

    # 初始化公共信息域
    shared_ctx = SharedContext()
    print(f"[init] 共享上下文信息域已创建: {shared_ctx}")

    # 初始化大模型
    model = OpenAIChatModel(
        model_name=config.LLM_MODEL_NAME,
        api_key=config.LLM_API_KEY,
        stream=config.STREAM,
        client_kwargs={"base_url": config.LLM_BASE_URL},
        generate_kwargs={"extra_body": {"enable_thinking": False}},  # fixme: 追求时延，默认模型关闭think模式
    )
    print(f"[init] 大模型已初始化: {config.LLM_MODEL_NAME}")

    # 初始化各个智能体（共4个：对话、表情、大脑、任务编排）
    chat_agent = ChatAgent(model=model, agent_name=AGENT_NAME)
    emotion_agent = EmotionAgent(model=model)
    brain_agent = BrainAgent(model=model, long_term_memory=long_term_memory)
    orchestrator = OrchestratorAgent(model=model)

    agents = {
        "chat": chat_agent,
        "emotion": emotion_agent,
        "brain": brain_agent,
    }

    # 初始化任务执行器
    executor = TaskExecutor(agents, scheduler, shared_ctx, latency_tracker=latency_tracker)
    print(f"[init] 任务执行器已创建: {list(agents.keys())}")
    
    
    
    
    ##################################################################################
    # 主循环
    round_num = 0

    try:
        while True:
            user_input = (
                await asyncio.get_event_loop().run_in_executor(None, input, "")  # fixme: ""👤 你: "" 本来是这个调用的第三个输出， 因为有流式输出，会影响终端日志打印所以先去除
            ).strip()

            if not user_input:
                continue

            round_num += 1
            msg = Msg(name="user", content=user_input, role="user")
            print(f"\n=== 第 {round_num} 轮 ===")

            # 开始新一轮计时
            latency_tracker.start_round(round_num, user_input)

            # ── 打断上一轮，当用户新输入到达时，并对部分执行调度器做打断 ──
            await executor.interrupt()
            # await shared_ctx.clear()
            # await scheduler.interrupt()

            # 重置核心chat_agent的prompt（上轮的重置）
            chat_agent.reset_prompt()

            # ── 编排智能体生成任务计划 ──
            orch_start = time.perf_counter()
            plan_dict = await orchestrator.plan(msg)
            orch_end = time.perf_counter()
            latency_tracker.record_agent(
                agent_name="orchestrator",
                node_type="orchestrator",
                start_ts=orch_start,
                end_ts=orch_end,
            )

            # 装载编排智能体的输出结果
            # Orchestrator 返回格式: {"node_list": ["quick_chat", "deep_think", ...]}
            nodes_data = plan_dict.get("node_list", [])
            plan = TaskPlan.from_raw_list(
                raw_nodes=nodes_data,
                version=round_num,
                source="orchestrator",
            )
            # print(f"任务队列: {' → '.join(n.name for n in plan.nodes)}")
            
            # ── 执行任务计划 ──
            await executor.execute(plan, msg)
            
            print(f"=== 第 {round_num} 轮结束 ===")

    except asyncio.CancelledError:
        # asyncio.run() 被 Ctrl+C 取消时正常退出
        pass
    finally:
        # 优雅关闭资源
        print("[Shutdown] 正在关闭输出调度器...")
        await scheduler.stop()
        print("[Shutdown] 输出调度器已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Logger] 程序已正常中断")
    finally:
        # 兜底：先关闭日志文件，再立即退出进程
        # （避免解释器关闭 ThreadPoolExecutor 线程时再次收到 Ctrl+C 导致 fatal error）
        import sys
        from scripts.logger.logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        import os
        os._exit(0)
