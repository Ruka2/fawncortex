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
from typing import Any
sys.path.insert(0, str(Path(__file__).parent))

# 模型配置表
import config

# 核心基础依赖
import asyncio
import time
from datetime import datetime

# AgentScope 基础依赖
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit

# 自定义智能体依赖
from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.agent.brain_agent import BrainAgent

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

# 外部非智能体执行工具
from deerberry.components.voice.tts import SiliconFlowCosyVoice
from deerberry.pipeline.output_scheduler import OutputScheduler, Priority

# 日志打印代码
from deerberry.logger.latency_tracker import LatencyTracker
from deerberry.logger.logger import enable_file_logging

# 【基础设施】聊天室控制器
from deerberry.pipeline.chatroom_controller import (
    EventBus,
    BackgroundBrainAgent,
    UserInputEvent,
    InterventionEvent,
)
from deerberry.pipeline.front_stage_pipeline import FrontStagePipeline
from deerberry.agent.reflection_agent import ReflectionAgent


# 基础用户配置
AGENT_NAME = "Ruka"
USER_NAME = "鹿过"


# =============================================================================
# 【基础设施】辅助函数
# =============================================================================

def build_model_for_role(role: str, stream: bool = True) -> OpenAIChatModel:
    """根据 config.LLM_ROLE_CONFIG 中的角色映射创建 OpenAIChatModel。

    与 main5_planqueue.py 保持一致。
    """
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
    reflection_model = build_model_for_role("orchestrator", stream=config.STREAM)

    print("[init] 多角色 LLM 配置映射:")
    for role, model in [
        ("chat", chat_model),
        ("emotion", emotion_model),
        ("brain", brain_model),
        ("orchestrator(reflection)", reflection_model),
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
    schemas = toolkit.get_json_schemas()
    print(f"[init] Brain Agent Toolkit 已组装，共 {len(schemas)} 个工具")

    brain_agent = BrainAgent(
        model=brain_model,
        long_term_memory=long_term_memory,
        toolkit=toolkit,
    )

    # 【策略占位】ReflectionAgent 目前使用规则驱动，model 预留未来升级为 LLM 驱动
    reflection = ReflectionAgent(model=reflection_model)

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

    try:
        while True:
            try:
                # ── 6.0 读取用户输入 ──
                user_input_raw = (
                    await asyncio.get_event_loop().run_in_executor(None, input, "")
                ).strip()

                if not user_input_raw:
                    continue

                round_num += 1
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_input = f"{now_str}\t{USER_NAME}: {user_input_raw}"
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

                # 【ReflectionAgent】积累用户输入到对话历史（observe 方案 C）
                await reflection.observe(msg)

                # ── 6.3 前台并行轨道：ChatAgent + EmotionAgent ──
                # 【前台智能体】由 FrontStagePipeline 统一封装并行执行 + 结果组合 + 输出调度
                chat_result, emotion_result = await front_stage.respond(msg)
                
                # 【BrainAgent】开始记录大脑智能体的响应时间（对话+表情智能体响应完成后）
                front_done_time = time.perf_counter() - round_start

                # ── 6.4 ReflectionAgent 第一次审判：前台完成后 ──
                # 【反思智能体】判断 Brain 是否过度思考 # FIXME: 思考过度截断大脑智能体的思考过程，并总结原因给chat_agent
                intervention = await reflection.judge_after_front(
                    chat_response=chat_result,
                    emotion_response=emotion_result,
                    brain_status=brain_bg.status,
                    elapsed=front_done_time,
                )

                # TODO: 需要优化停止打断大脑智能体的内核算法或机制
                if intervention.action == "stop_brain":
                    print(f"[Reflection] 🛑 {intervention.payload}")
                    brain_bg.cancel()
                    # TODO: 时间太长之后，反思智能体应该给出什么反馈信息给大脑智能体优化下一次任务


                # ── 6.5 等待 BrainAgent 思考结果（带超时，非阻塞前台）──
                # 【策略占位】超时阈值：简单问题 Brain 可能不需要跑完
                BRAIN_TIMEOUT = 60.0  # 秒，建议后续根据问题复杂度动态调整

                try:
                    # FIXME: 深度思考（前台没有响应后最大容许大脑智能体的时间）
                    brain_output = await asyncio.wait_for(
                        brain_bg.output_queue.get(),
                        timeout=BRAIN_TIMEOUT,
                    )
                    
                    if brain_output["status"] == "completed":
                        thought = brain_output["thought"]

                        # 【ReflectionAgent】先积累前台回复到对话历史
                        # 必须在 judge 之前 observe，确保 reflection 的上下文包含
                        # 对话智能体(Chat Agent)的提前响应，避免 history 以 user 结尾
                        # 导致 review prompt 追加后出现连续 user
                        await reflection.observe(chat_result)

                        # ── 6.6 ReflectionAgent 第二次审判：Brain 完成后 ──
                        intervention2 = await reflection.judge_after_brain(
                            thought=thought,
                            chat_response=chat_result,
                            user_question=user_input_raw,
                        )

                    if intervention2.action == "ignore":
                            print(f"[Reflection] ⏭️ Brain 结果被忽略")
                            print(f"   原因: {intervention2.payload}")

                    elif intervention2.action in ("clarify", "summarize"):
                        insight = intervention2.payload
                        action_label = "总结" if intervention2.action == "summarize" else "追问"
                        print(f"[Reflection] 💡 Brain {action_label}同步到 ChatAgent")
                        print(f"   {insight[:100]}{'...' if len(insight) > 100 else ''}")

                        # 将大脑洞察作为 assistant 消息加入对话历史（短期上下文）
                        # insight_msg = Msg(
                        #     name="system",   # FIXME: 小心注意这个位置， 这个是在上下文中间插入一个system作为非用户和非assistant的角色
                        #     content=f"### 智能体已后台思考的内容\n{insight}",
                        #     role="system",
                        # )
                        insight_msg = Msg(
                            name="assistant",   # FIXME: 小心注意这个位置， 这个是在上下文中间插入一个system作为非用户和非assistant的角色
                            content=f"### 智能体已后台思考的内容\n{insight}",
                            role="assistant",
                        )
                        await chat_agent.memory.add(insight_msg)
                        
                        # 触发 ChatAgent 基于大脑洞察再次回复
                        follow_up = await chat_agent.reply(
                            Msg(name="user", content="请你结合上下文和思考内容继续回复", role="user")
                        )
                        
                        follow_text = follow_up.get_text_content() or ""
                        print(f"💬 [{action_label}] {follow_text}")

                        # 将追问回复也加入对话历史，保持上下文连贯
                        await chat_agent.memory.add(follow_up)

                        # 插队播报追问内容（高优先级）
                        await scheduler.schedule(
                            follow_text,
                            EmotionAgent.parse_action(
                                emotion_result.get_text_content()
                            ) if emotion_result else "smile",
                            "brain_clarify",
                            priority=Priority.HIGH,
                        )
                            

                    elif brain_output["status"] == "cancelled":
                        print("🧠 BrainAgent 思考已被取消（被 ReflectionAgent 或新输入打断）")

                except asyncio.TimeoutError:
                    print(f"[Reflection] ⏱ BrainAgent 思考超时（>{BRAIN_TIMEOUT}s），放弃本轮结果")
                    # 【策略占位】超时后可选：强制打断 Brain，或让其继续后台运行但不等待
                    brain_bg.cancel()

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
