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
from fawncortex.agent.chat_agent import ChatAgent
from fawncortex.agent.emotion_agent import EmotionAgent
from fawncortex.agent.brain_agent import BrainAgent
from fawncortex.agent.reflection_agent import ReflectionAgent

# 自定义智能体记忆的实例类
from fawncortex.base.memory import create_long_term_memory

# 外部引用工具
from fawncortex.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
)
from fawncortex.tools.paper_search import (
    search_papers,
    get_paper_details,
    read_paper,
)
from fawncortex.tools.get_current_time import get_current_time
from fawncortex.tools.online_search import online_search

# 外部非智能体执行工具
from fawncortex.components.voice.tts import SiliconFlowCosyVoice
from fawncortex.components.body.vts_controller import VTSController
from fawncortex.pipeline.output_scheduler import OutputScheduler

# 日志打印代码
from fawncortex.logger.latency_tracker import LatencyTracker
from fawncortex.logger.logger import enable_file_logging

# 分布式控制器
from fawncortex.pipeline.event_controller import (
    EventBus,
    BackgroundBrainAgent,
    UserInputEvent
)

# 分布式管线控制
from fawncortex.pipeline.back_stage_midway import midway_watcher, brain_summary
from fawncortex.pipeline.front_stage_pipeline import FrontStagePipeline




# 快速批量封装模型入口
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



# 主函数
async def main() -> None:
    # ── 初始化日志 ──
    enable_file_logging()
    latency_tracker = LatencyTracker()

    # ── 初始化 TTS + VTS -> OutputScheduler（语音输出和表情输出同轨道）──
    tts = SiliconFlowCosyVoice(
        api_key=config.TTS_API_KEY,
        api_url=config.TTS_BASE_URL,
        model=config.TTS_MODEL_NAME,
        voice=config.TTS_VOICE,
    )

    vts = VTSController()
    try:
        await vts.connect_and_auth()
        print("[init] VTS 已连接并认证成功")
    except Exception as e:
        print(f"[init] VTS 连接失败，将以无 VTS 模式运行: {e}")
        vts = None

    scheduler = OutputScheduler(tts=tts, vts=vts, latency_tracker=latency_tracker)
    asyncio.create_task(scheduler.run())
    print(f"[init] TTS 已创建: {config.TTS_MODEL_NAME}, {config.TTS_VOICE}")
    print("[init] OutputScheduler 已完成启动")

    # ── 初始化长期记忆实例（mem0） ──
    memory_cfg = config.LLM_ROLE_CONFIG.get("memory", {})
    long_term_memory = create_long_term_memory(
        agent_name=config.AGENT_NAME,
        user_name=config.USER_NAME,
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


    # ── 按角色创建专用大模型实例 ──
    chat_model = build_model_for_role("chat", stream=config.STREAM)
    emotion_model = build_model_for_role("emotion", stream=config.STREAM)
    brain_model = build_model_for_role("brain", stream=config.STREAM)
    reflection_model = build_model_for_role("reflection", stream=config.STREAM)

    # 仅打印各个智能体的模型配置
    print("[init] 多智能体的LLM配置映射:")
    for role, _ in [
        ("chat", chat_model),
        ("emotion", emotion_model),
        ("brain", brain_model),
        ("reflection", reflection_model),
    ]:
        cfg = config.LLM_ROLE_CONFIG.get(role.replace("(reflection)", ""), {})
        used_model = cfg.get("model_name") or config.LLM_MODEL_NAME
        used_base = cfg.get("base_url") or config.LLM_BASE_URL
        print(f"       {role:25s} model={used_model}, base_url={used_base}")

    # ── 初始化核心智能体 ──
    chat_agent = ChatAgent(model=chat_model, agent_name=config.AGENT_NAME)
    emotion_agent = EmotionAgent(model=emotion_model)

    # 组装大脑智能体的工具
    toolkit = Toolkit()
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
    toolkit.register_tool_function(search_papers)
    toolkit.register_tool_function(read_paper)
    toolkit.register_tool_function(get_paper_details)
    toolkit.register_tool_function(online_search)
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

    # ── 初始化事件总线 + 后台 BrainAgent 挂载 ──
    bus = EventBus()
    # BrainAgent 通过 EventBus 订阅 user.input topic
    bus.subscribe("BrainAgent", ["user.input"])

    brain_bg = BackgroundBrainAgent(brain_agent, bus)
    brain_task = asyncio.create_task(brain_bg.run())
    print("[init] BackgroundBrainAgent 已启动（后台常驻）")

    # ── 初始化前台并行管道 ──
    front_stage = FrontStagePipeline(
        chat_agent=chat_agent,
        emotion_agent=emotion_agent,
        scheduler=scheduler,
    )
    print("[init] FrontStagePipeline 前台并行管道已创建")

    # ── 主循环：事件驱动的聊天管道 ──
    round_num = 0
    # 中间汇报任务管理（每轮独立）
    current_midway_task: Optional[asyncio.Task] = None
    current_stop_event: Optional[asyncio.Event] = None

    try:
        while True:
            try:
                # ── 读取用户输入 ──
                user_input = (
                    await asyncio.get_event_loop().run_in_executor(None, input, "")
                ).strip()
                if not user_input:
                    continue

                # ── 取消上一轮的 midway_watcher（用于中断上轮未完成的中期汇报任务） ──
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
                    
                # 用户新输入到达时，清空 TTS 队列，避免旧消息干扰
                await scheduler.interrupt()

                # 初始计数
                round_num += 1
                msg = Msg(name="user", content=user_input, role="user")
                print(f"\n{'='*60}")
                print(f"🚀 第 {round_num} 轮  |  {user_input}")
                print(f"{'='*60}")

                round_start = time.perf_counter()
                latency_tracker.start_round(round_num, user_input)

                # ── 向后台 BrainAgent 投递事件（非阻塞）──
                # 【BrainAgent】 大脑智能体在后台独立运行，不阻塞前台响应
                await bus.publish("user.input", UserInputEvent(
                    msg=msg, round_id=round_num
                ))

                # ── 6.3 前台并行轨道：ChatAgent + EmotionAgent ──
                # 【前台智能体】由 FrontStagePipeline 统一封装并行执行 + 结果组合 + 输出调度
                chat_result, emotion_result, chat_elapsed, emotion_elapsed = await front_stage.respond(msg)

                # 记录当前轮次的表情，如果有以下pipeline想要复用的话
                current_emotion, current_tone = EmotionAgent.parse_action(emotion_result.get_text_content())
                
                
                # ── 6.4.5 启动 midway_watcher（中间思考过程的监听器）──
                # 根据前台回复计算本轮思考可容忍多少时间  FIXME: 此处是一个可改善的点，可采用字符计算、或反思计算、根据TTS市场动态等待的动态设计
                threshold = reflection_agent.compute_dynamic_threshold(chat_result)
                print(f"[Midway] 🕐 动态阈值: {threshold:.1f}s（前台回复 {len(chat_result.get_text_content() or '')} 字符）")
                current_stop_event = asyncio.Event()
                
                # FIXME: 因为中间汇报和总结目前不考虑再一次推理表情，所以先使用默认占位符表情防止做出多余的动作
                current_emotion = "neural"
                
                # 创建中期汇报任务
                current_midway_task = asyncio.create_task(
                    midway_watcher(
                        brain_bg=brain_bg,
                        chat_agent=chat_agent,
                        scheduler=scheduler,
                        reflection_agent=reflection_agent,
                        emotion=current_emotion,
                        tone=current_tone,
                        threshold=threshold,
                        stop_event=current_stop_event,
                        user_name=config.USER_NAME,
                        user_input=user_input,
                    )
                )

                # ── 等待 BrainAgent 思考结果（带超时，非阻塞前台）──
                try:
                    brain_output = await asyncio.wait_for(
                        brain_bg.output_queue.get(),
                        timeout=config.BRAIN_TIMEOUT,  # 前台没有响应后最大容许大脑智能体的时间，建议后续根据问题复杂度动态调整
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

                        # FIXME: 因为中间汇报和总结目前不考虑再一次推理表情，所以先使用默认占位符表情防止做出多余的动作
                        current_emotion = "neural"
                        
                        # 触发总结
                        await brain_summary(
                            chat_agent=chat_agent,
                            scheduler=scheduler,
                            reflection_agent=reflection_agent,
                            current_emotion=current_emotion,
                            current_tone=current_tone,
                            user_input=user_input,
                            summary_thought=summary_thought,
                            brain_bg=brain_bg,
                            source_label="brain_summary",
                            user_name=config.USER_NAME,
                        )
                                                    

                except asyncio.TimeoutError:
                    print(f"[Reflection] ⏱ BrainAgent 思考超时（>{config.BRAIN_TIMEOUT}s），触发最后一次总结")

                    # ── 先停止 midway_watcher，确保所有 midway 都已入队后再触发 summary ──
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
                            brain_bg=brain_bg,
                            source_label="brain_summary",
                            user_name=config.USER_NAME,
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
        try:
            print("[Shutdown] 正在关闭输出调度器...")
            await scheduler.stop()
        except Exception as e:
            print(f"[Shutdown] 关闭 scheduler 失败: {e}")

        try:
            if vts:
                print("[Shutdown] 正在关闭 VTS...")
                await vts.close()
        except Exception as e:
            print(f"[Shutdown] 关闭 VTS 失败: {e}")

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
        from fawncortex.logger.logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        os._exit(0)
