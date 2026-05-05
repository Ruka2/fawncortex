

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
from agentscope.tool import Toolkit

# 自定义智能体工具依赖
from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.agent.brain_agent import BrainAgent
from deerberry.agent.orchestrator_agent import OrchestratorAgent

from deerberry.pipeline.task_plan import TaskPlan
from deerberry.pipeline.output_scheduler import OutputScheduler
from deerberry.pipeline.task_executor import TaskExecutor
from deerberry.pipeline.shared_context import SharedContext

# 自定义智能体记忆的实例类
from deerberry.base.memory import create_long_term_memory

# 外部引用工具
from deerberry.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
)
from deerberry.tools.arxiv_search import (
    create_arxiv_client,
    register_arxiv_tools,
    close_arxiv_client,
)

# 外部非智能体执行工具
from deerberry.components.voice.tts import SiliconFlowCosyVoice

# 日志打印代码
from deerberry.logger.latency_tracker import LatencyTracker
from deerberry.logger.logger import enable_file_logging


# 基础用户配置
AGENT_NAME = "Ruka"
USER_NAME = "鹿过"

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

    # 初始化长期记忆（使用 memory 角色专用配置）
    memory_cfg = config.LLM_ROLE_CONFIG.get("memory", {})
    long_term_memory = create_long_term_memory(
        agent_name=AGENT_NAME,
        user_name=USER_NAME,
        vector_store_path=config.MEM0_VECTOR_STORE_PATH,
        history_db_path=config.MEM0_HISTORY_DB_PATH,
        llm_model_name=memory_cfg.get("model_name") or config.LLM_MODEL_NAME,
        llm_api_key=memory_cfg.get("api_key") or config.LLM_API_KEY,
        llm_base_url=memory_cfg.get("base_url") or config.LLM_BASE_URL,
        llm_generate_kwargs=config.LLM_ROLE_GENERATE_KWARGS.get("memory"),
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        embedding_api_key=config.EMBEDDING_API_KEY,
        embedding_base_url=config.EMBEDDING_BASE_URL,
    )
    set_memory_manager(long_term_memory)
    print(f"[init] 长期记忆已初始化: {config.MEM0_HISTORY_DB_PATH}, Memory LLM: {memory_cfg.get('model_name') or config.LLM_MODEL_NAME}, Embedding Model: {config.EMBEDDING_MODEL_NAME}")

    # 初始化公共信息域
    shared_ctx = SharedContext()
    print(f"[init] 共享上下文信息域已创建: {shared_ctx}")


    # 按智能体角色创建专用大模型实例
    def build_model_for_role(role: str, stream: bool = True):
        """根据 config.LLM_ROLE_CONFIG 中的角色映射创建 OpenAIChatModel。"""
        cfg = config.LLM_ROLE_CONFIG.get(role, {})
        model_name = cfg.get("model_name") or config.LLM_MODEL_NAME
        api_key = cfg.get("api_key") or config.LLM_API_KEY
        base_url = cfg.get("base_url") or config.LLM_BASE_URL
        generate_kwargs = config.LLM_ROLE_GENERATE_KWARGS.get(role, {})

        return OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs=generate_kwargs,
        )
    
    chat_model = build_model_for_role("chat", stream=config.STREAM)
    emotion_model = build_model_for_role("emotion", stream=config.STREAM)
    brain_model = build_model_for_role("brain", stream=config.STREAM)
    orchestrator_model = build_model_for_role("orchestrator", stream=config.STREAM)

    # 打印各角色模型映射信息
    print("[init] 多角色LLM配置映射:")
    for role, model in [
        ("chat", chat_model),
        ("emotion", emotion_model),
        ("brain", brain_model),
        ("orchestrator", orchestrator_model),
    ]:
        cfg = config.LLM_ROLE_CONFIG.get(role, {})
        used_model = cfg.get("model_name") or config.LLM_MODEL_NAME
        used_base = cfg.get("base_url") or config.LLM_BASE_URL
        print(f"       {role:15s} model={used_model}, base_url={used_base}")

    # 补充 memory 记忆系统的配置映射
    memory_used_model = memory_cfg.get("model_name") or config.LLM_MODEL_NAME
    memory_used_base = memory_cfg.get("base_url") or config.LLM_BASE_URL
    print(f"       {'memory':15s} model={memory_used_model}, base_url={memory_used_base}")

    toolkit = Toolkit()
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
    schemas = toolkit.get_json_schemas()
    print(f"[init] Brain Agent Toolkit 已组装，共 {len(schemas)} 个工具:")
    for s in schemas:
        name = s.get("function", {}).get("name", "unknown")
        print(f"       {name}")

    # 初始化对话所使用到的核心智能体（核心智能体：对话、大脑、表情；流程控制：编排；工具类：记忆）
    chat_agent = ChatAgent(model=chat_model, agent_name=AGENT_NAME)
    emotion_agent = EmotionAgent(model=emotion_model)
    brain_agent = BrainAgent(
        model=brain_model,
        long_term_memory=long_term_memory,
        toolkit=toolkit,
    )
    orchestrator = OrchestratorAgent(model=orchestrator_model)

    agents = {
        "chat": chat_agent,
        "emotion": emotion_agent,
        "brain": brain_agent,
        "orchestrator": orchestrator
    }

    # 初始化任务执行器
    executor = TaskExecutor(agents, scheduler, shared_ctx, latency_tracker=latency_tracker)
    print(f"[init] 任务执行器已创建")
    
    
    
    
    ##################################################################################
    # 主循环
    round_num = 0

    try:
        while True:
            try:
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

                # ── 将 chat agent 的对话上下文同步到 orchestrator ──
                # 让编排智能体基于闲聊智能体的完整对话历史来规划任务
                chat_history = await chat_agent.memory.get_memory()
                await orchestrator.memory.clear()
                for m in chat_history:
                    await orchestrator.memory.add(m)
                # print(f"[sync] orchestrator 上下文已同步为 chat_agent 的 {len(chat_history)} 条消息")

                # ── 编排智能体生成任务计划 ──
                orch_start = time.perf_counter()
                try:
                    plan_dict = await orchestrator.plan(msg)
                except Exception as e:
                    print(f"❌ Orchestrator 任务规划异常: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    latency_tracker.record_agent(
                        agent_name="orchestrator",
                        node_type="orchestrator",
                        start_ts=orch_start,
                        end_ts=time.perf_counter(),
                    )
                    print(f"=== 第 {round_num} 轮异常结束 ===")
                    continue

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
                try:
                    await executor.execute(plan, msg)
                except Exception as e:
                    print(f"❌ 任务执行异常: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"=== 第 {round_num} 轮异常结束 ===")
                    continue
                
                print(f"=== 第 {round_num} 轮结束 ===")

            except asyncio.CancelledError:
                raise  # 重新抛出，让外层处理 Ctrl+C
            except Exception as e:
                # 单轮兜底：任何未捕获的异常只结束本轮，不退出程序
                print(f"❌ 第 {round_num} 轮发生未捕获异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                print(f"=== 第 {round_num} 轮异常结束 ===")
                continue

    except asyncio.CancelledError:
        # asyncio.run() 被 Ctrl+C 取消时正常退出
        pass
    except Exception as e:
        # 外层兜底：捕获主循环级别的致命异常
        print(f"💥 程序主循环发生致命异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 优雅关闭资源（注意：多个 StatefulClient 必须按 LIFO 顺序关闭）
        print("[Shutdown] 正在关闭输出调度器...")
        await scheduler.stop()
        print("[Shutdown] 输出调度器已关闭")
        print("[Shutdown] 正在关闭 MCP 服务...")
        await close_arxiv_client()
        print("[Shutdown] MCP 服务已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Logger] 程序已正常中断")
    finally:
        # 兜底：先关闭日志文件，再立即退出进程
        # （避免解释器关闭 ThreadPoolExecutor 线程时再次收到 Ctrl+C 导致 fatal error）
        import sys
        from deerberry.logger.logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        import os
        os._exit(0)
