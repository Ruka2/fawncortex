"""
聊天室入口（Chatroom Entry）
=============================
Event-Driven Multi-Agent Chatroom 的完整入口文件。

架构概览（参考 PROJECT_ANALYSIS_AND_ROADMAP.md 3.1.1）：
┌─────────────────────────────────────────────────────────────┐
│  用户输入 → EventBus → 前台并行轨道（System 1）               │
│                       → 后台认知轨道（System 2）               │
│                       → ReflectionAgent（审判官）             │
│                       → OutputScheduler（TTS/VTS 输出）       │
└─────────────────────────────────────────────────────────────┘

轨道说明：
- 前台轨道：ChatAgent + EmotionAgent 并行响应，谁先完成谁先打印/播报
- 后台轨道：BrainAgent 常驻后台，通过 EventBus 接收事件并深度思考
- 审判层：ReflectionAgent 在关键时机（前台完成、Brain 完成、超时）做干预决策

【架构标注说明】
- 【基础设施】：已完整实现，可直接运行
- 【策略占位】：仅提供骨架，需你后续精细化调整
- 【扩展点】：标注了你未来可能扩展的位置
"""

# 项目路径根目录定位
import sys
from pathlib import Path
from typing import Any, Optional
sys.path.insert(0, str(Path(__file__).parent))

# 模型配置表
import config

# 核心基础依赖
import asyncio
import time
# from datetime import datetime

# AgentScope 基础依赖
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit

# 自定义智能体依赖
from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.agent.brain_agent import BrainAgent
from deerberry.agent.reflection_agent import ReflectionAgent

# 自定义智能体记忆的实例类
from deerberry.base.memory import create_long_term_memory

# 外部引用工具
from deerberry.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
)
from deerberry.tools.paper_search import (
    search_papers,
    get_paper_details,
    search_authors,
    read_paper,
)
from deerberry.tools.get_current_time import get_current_time

# 外部非智能体执行工具
from deerberry.components.voice.tts import SiliconFlowCosyVoice
from deerberry.pipeline.output_scheduler import OutputScheduler

# 日志打印代码
from deerberry.logger.latency_tracker import LatencyTracker
from deerberry.logger.logger import enable_file_logging

# 分布式控制器
from deerberry.pipeline.event_controller import (
    EventBus,
    BackgroundBrainAgent,
    UserInputEvent
)

# 分布式管线控制
from deerberry.pipeline.back_stage_midway import midway_watcher, brain_summary
from deerberry.pipeline.front_stage_pipeline import FrontStagePipeline


# 基础用户配置
AGENT_NAME = "Ruka"
USER_NAME = "鹿过"


# =============================================================================
# 【基础设施】辅助函数
# =============================================================================

def build_model_for_role(role: str, stream: bool = True) -> OpenAIChatModel:
    """ 根据 config.LLM_ROLE_CONFIG 中的角色映射创建 OpenAIChatModel """
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







# =============================================================================
# 【基础设施】主函数
# =============================================================================
async def main() -> None:
    # ── 0. 初始化日志 ──
    enable_file_logging()
    latency_tracker = LatencyTracker()

    # ── 1. 初始化 TTS + OutputScheduler（语音/表情输出轨道）──
    tts = SiliconFlowCosyVoice(
        api_key=config.TTS_API_KEY,
        api_url=config.TTS_BASE_URL,
        model=config.TTS_MODEL_NAME,
        voice=config.TTS_VOICE,
    )
    scheduler = OutputScheduler(tts, latency_tracker=latency_tracker)
    asyncio.create_task(scheduler.run())
    print(f"[init] TTS 已创建: {config.TTS_MODEL_NAME}, {config.TTS_VOICE}")
    print("[init] OutputScheduler 已启动")

    # ── 2. 初始化长期记忆 ──
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
    print(f"[init] 长期记忆已初始化: {config.MEM0_HISTORY_DB_PATH}")


    # ── 3. 按角色创建专用大模型实例 ──
    chat_model = build_model_for_role("chat", stream=config.STREAM)
    emotion_model = build_model_for_role("emotion", stream=config.STREAM)
    brain_model = build_model_for_role("brain", stream=config.STREAM)
    reflection_model = build_model_for_role("reflection", stream=config.STREAM)

    print("[init] 多角色 LLM 配置映射:")
    for role, model in [
        ("chat", chat_model),
        ("emotion", emotion_model),
        ("brain", brain_model),
        ("reflection", reflection_model),
    ]:
        cfg = config.LLM_ROLE_CONFIG.get(role.replace("(reflection)", ""), {})
        used_model = cfg.get("model_name") or config.LLM_MODEL_NAME
        used_base = cfg.get("base_url") or config.LLM_BASE_URL
        print(f"       {role:25s} model={used_model}, base_url={used_base}")

    # ── 4. 初始化核心智能体 ──
    chat_agent = ChatAgent(model=chat_model, agent_name=AGENT_NAME)
    emotion_agent = EmotionAgent(model=emotion_model)

    toolkit = Toolkit()
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
    toolkit.register_tool_function(search_papers)
    toolkit.register_tool_function(read_paper)
    toolkit.register_tool_function(get_paper_details)
    toolkit.register_tool_function(search_authors)
    toolkit.register_tool_function(get_current_time)
    schemas = toolkit.get_json_schemas()
    print(f"[init] Brain Agent Toolkit 已组装，共 {len(schemas)} 个工具")

    brain_agent = BrainAgent(
        model=brain_model,
        long_term_memory=long_term_memory,
        toolkit=toolkit,
    )
    reflection_agent = ReflectionAgent(model=reflection_model)
    print("[init] 核心智能体集群已创建: ChatAgent, EmotionAgent, BrainAgent, ReflectionAgent")

    # ── 5. 初始化事件总线 + 后台 BrainAgent ──
    bus = EventBus()
    # BrainAgent 通过 EventBus 订阅 user.input topic
    bus.subscribe("BrainAgent", ["user.input"])

    brain_bg = BackgroundBrainAgent(brain_agent, bus)
    brain_task = asyncio.create_task(brain_bg.run())
    print("[init] BackgroundBrainAgent 已启动（后台常驻）")

    # ── 5.5 初始化前台并行管道 ──
    front_stage = FrontStagePipeline(
        chat_agent=chat_agent,
        emotion_agent=emotion_agent,
        scheduler=scheduler,
    )
    print("[init] FrontStagePipeline 前台并行管道已创建")

    # ── 6. 主循环：事件驱动的聊天室 ──
    round_num = 0
    # 中间汇报任务管理（每轮独立）
    current_midway_task: Optional[asyncio.Task] = None
    current_stop_event: Optional[asyncio.Event] = None
    # 当前轮次的前台表情（用于 midway TTS）
    current_emotion = "smile"

    try:
        while True:
            try:
                # ── 6.0 读取用户输入 ──
                user_input = (
                    await asyncio.get_event_loop().run_in_executor(None, input, "")
                ).strip()

                if not user_input:
                    continue

                # ── 取消上一轮的 midway_watcher ──
                if current_midway_task and not current_midway_task.done():
                    if current_stop_event:
                        current_stop_event.set()
                    current_midway_task.cancel()
                    try:
                        await current_midway_task
                    except asyncio.CancelledError:
                        pass
                    current_midway_task = None
                    current_stop_event = None



                round_num += 1
                msg = Msg(name="user", content=user_input, role="user")
                print(f"\n{'='*60}")
                print(f"🚀 第 {round_num} 轮  |  {user_input}")
                print(f"{'='*60}")

                round_start = time.perf_counter()
                latency_tracker.start_round(round_num, user_input)

                # ── 6.1 打断上一轮输出 ──
                # 【基础设施】用户新输入到达时，清空 TTS 队列，避免旧消息干扰
                await scheduler.interrupt()

                # ── 6.2 向后台 BrainAgent 投递事件（非阻塞）──
                # 【BrainAgent】 大脑智能体在后台独立运行，不阻塞前台响应
                await bus.publish("user.input", UserInputEvent(
                    msg=msg, round_id=round_num
                ))

                # ── 6.3 前台并行轨道：ChatAgent + EmotionAgent ──
                # 【前台智能体】由 FrontStagePipeline 统一封装并行执行 + 结果组合 + 输出调度
                chat_result, emotion_result = await front_stage.respond(msg)

                # 记录当前轮次的表情，供 midway 汇报复用
                if emotion_result:
                    current_emotion = EmotionAgent.parse_action(
                        emotion_result.get_text_content() or ""
                    )
                else:
                    current_emotion = "smile"
                
                
                # ── 6.4.5 启动 midway_watcher（中间过程监听器）──
                # 【阶段】动态阈值计算 + 独立 Task 启动
                threshold = reflection_agent.compute_dynamic_threshold(chat_result)
                print(f"[Midway] 🕐 动态阈值: {threshold:.1f}s（前台回复 {len(chat_result.get_text_content() or '')} 字符）")
                current_stop_event = asyncio.Event()
                current_midway_task = asyncio.create_task(
                    midway_watcher(
                        brain_bg=brain_bg,
                        chat_agent=chat_agent,
                        scheduler=scheduler,
                        reflection_agent=reflection_agent,
                        emotion=current_emotion,
                        threshold=threshold,
                        stop_event=current_stop_event,
                        user_name=USER_NAME,
                        user_input=user_input,
                    )
                )

                # ── 6.5 等待 BrainAgent 思考结果（带超时，非阻塞前台）──
                BRAIN_TIMEOUT = 360.0  # 秒，深度思考，前台没有响应后最大容许大脑智能体的时间，建议后续根据问题复杂度动态调整
                try:
                    brain_output = await asyncio.wait_for(
                        brain_bg.output_queue.get(),
                        timeout=BRAIN_TIMEOUT,
                    )
                    
                    # brain 完成后，通知 midway_watcher 停止
                    if current_stop_event:
                        current_stop_event.set()
                    if current_midway_task and not current_midway_task.done():
                        try:
                            await asyncio.wait_for(current_midway_task, timeout=3.0)
                        except asyncio.TimeoutError:
                            current_midway_task.cancel()
                            try:
                                await current_midway_task
                            except asyncio.CancelledError:
                                pass
                    current_midway_task = None
                    current_stop_event = None
                    
                    # 只有状态为completed时才触发总结
                    if brain_output["status"] == "completed":
                        thought = brain_output["thought"]
                        summary_thought = thought.raw_data.get("insight", "")

                        await brain_summary(
                            chat_agent=chat_agent,
                            scheduler=scheduler,
                            reflection_agent=reflection_agent,
                            current_emotion=current_emotion,
                            user_input=user_input,
                            summary_thought=summary_thought,
                            source_label="brain_summary",
                            user_name=USER_NAME,
                        )
                                                    

                except asyncio.TimeoutError:
                    print(f"[Reflection] ⏱ BrainAgent 思考超时（>{BRAIN_TIMEOUT}s），触发最后一次总结")

                    # ── 1. 先停止 midway_watcher，确保所有 midway 都已入队后再触发 summary ──
                    if current_stop_event:
                        current_stop_event.set()
                    if current_midway_task and not current_midway_task.done():
                        current_midway_task.cancel()
                        try:
                            await current_midway_task
                        except asyncio.CancelledError:
                            pass
                    current_midway_task = None
                    current_stop_event = None

                    # 先获取 brain 当前已产生的思考内容（取消前快照）
                    snapshot = brain_bg.brain.get_react_snapshot()
                    parts = []
                    for it in snapshot.get("iterations", []):
                        text = it.get("reasoning_text", "")
                        if text:
                            parts.append(text)
                    stream = snapshot.get("stream_buffer", "")
                    if stream:
                        parts.append(stream)
                    summary_thought = "\n\n".join(parts)

                    # 取消 brain（同步接口，不等待 Task 结束）
                    brain_bg.cancel()

                    # 触发最后一次总结（若还有可用内容）
                    if summary_thought.strip():
                        await brain_summary(
                            chat_agent=chat_agent,
                            scheduler=scheduler,
                            reflection_agent=reflection_agent,
                            current_emotion=current_emotion,
                            user_input=user_input,
                            summary_thought=summary_thought,
                            source_label="brain_summary",
                            user_name=USER_NAME,
                        )
                    else:
                        print("[BrainSummary] ⚠️ 超时后无可用思考内容，跳过总结")

                # ── 6.7 本轮统计 ──
                round_elapsed = time.perf_counter() - round_start
                print(f"\n[Rounds] 第 {round_num} 轮结束，总耗时: {round_elapsed:.2f}s")
                latency_tracker.record_agent(
                    agent_name="round_total",
                    node_type="round",
                    start_ts=round_start,
                    end_ts=time.perf_counter(),
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"❌ 第 {round_num} 轮未捕获异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except asyncio.CancelledError:
        pass
    except Exception as e:
        # 【全局异常捕获】初始化阶段或主循环外层异常
        print(f"\n{'='*60}")
        print(f"💥 程序致命异常: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
    finally:
        # ── 优雅关闭 ──
        try:
            print("[Shutdown] 正在关闭输出调度器...")
            await scheduler.stop()
        except Exception as e:
            print(f"[Shutdown] 关闭 scheduler 失败: {e}")

        try:
            print("[Shutdown] 正在关闭 BackgroundBrainAgent...")
            await brain_bg.stop()
            brain_task.cancel()
            try:
                await brain_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            print(f"[Shutdown] 关闭 brain_bg 失败: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Logger] 程序已正常中断")
    except Exception as e:
        # 【入口层兜底】捕获 async.run 抛出的任何未处理异常
        print(f"\n{'='*60}")
        print(f"💥 入口层未捕获异常: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
    finally:
        import os
        from deerberry.logger.logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        os._exit(0)
