"""
Deerberry Web 服务端核心引擎（main8_server.py）
===============================================
基于 main7_reflection.py 改造，适配 Web UI 的事件驱动版本。

设计原则：
- 与前端完全解耦：通过 EventEmitter 事件系统暴露内部状态
- 不修改原有 Agent / Pipeline 代码，仅做包装和适配
- TTS 音频通过事件发送给前端播放，不再本地 sd.play()
- 用户输入通过 asyncio.Queue 接收，替代 input()

事件类型（供 server.py 订阅）：
- user_message      : 用户发送了消息
- chat_message      : ChatAgent 产生回复
- emotion_update    : EmotionAgent 产生表情
- brain_snapshot    : BrainAgent 状态快照（含 ReAct 轮次）
- brain_summary     : BrainAgent 思考完成
- midway_message    : Midway 汇报触发
- reflection_judgment: ReflectionAgent 判决结果
- tts_text          : 即将 TTS 播报的文本
- tts_audio         : TTS 音频数据（base64）
- interrupt         : 用户打断/队列清空
- round_start       : 新一轮开始
- round_end         : 一轮结束
- error             : 错误信息
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
import asyncio
import time
import base64
from typing import Any, Optional, Callable
from collections import defaultdict

from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit

from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.agent.brain_agent import BrainAgent
from deerberry.agent.reflection_agent import ReflectionAgent

from deerberry.base.memory import create_long_term_memory
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
from deerberry.components.voice.tts import SiliconFlowCosyVoice
from deerberry.pipeline.output_scheduler import OutputScheduler, Priority, OutputTask
from deerberry.pipeline.event_controller import (
    EventBus,
    BackgroundBrainAgent,
    UserInputEvent,
)
from deerberry.pipeline.front_stage_pipeline import FrontStagePipeline
from deerberry.pipeline.back_stage_midway import midway_watcher, brain_summary
from deerberry.tools.control_vts import express_emotion

from deerberry.logger.latency_tracker import LatencyTracker
from deerberry.logger.logger import enable_file_logging


AGENT_NAME = "Ruka"
USER_NAME = "鹿过"


# =============================================================================
# 辅助函数
# =============================================================================

def build_model_for_role(role: str, stream: bool = True) -> OpenAIChatModel:
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


# =============================================================================
# 事件发射器
# =============================================================================

class EventEmitter:
    """轻量级异步事件系统。

    将 DeerberryEngine 的内部状态通过事件暴露给外部（server.py）。
    支持同步和异步 handler。
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event_type: str, handler: Callable) -> None:
        """注册事件处理器。"""
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """注销事件处理器。"""
        if handler in self._handlers.get(event_type, []):
            self._handlers[event_type].remove(handler)

    async def emit(self, event_type: str, data: Any) -> None:
        """发射事件到所有注册的处理器。"""
        for handler in self._handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler({"type": event_type, "data": data})
                else:
                    handler({"type": event_type, "data": data})
            except Exception as e:
                print(f"[EventEmitter] ⚠️ handler error for '{event_type}': {e}")


# =============================================================================
# Web UI 适配的输出调度器
# =============================================================================

class WebOutputScheduler:
    """Web UI 适配的输出调度器。

    保留原有 OutputScheduler 的核心逻辑：
    - asyncio.PriorityQueue 优先级队列
    - 打断/清空能力
    - VTS 表情触发（express_emotion）

    关键差异：
    - TTS 不再本地 sd.play() 播放，而是将音频数据通过 EventEmitter 发送给前端
    - 新增 tts_text 事件，让前端可以提前显示即将播报的文字
    """

    def __init__(
        self,
        tts: SiliconFlowCosyVoice,
        emitter: EventEmitter,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> None:
        self.tts = tts
        self.emitter = emitter
        self.latency_tracker = latency_tracker
        self._queue: asyncio.PriorityQueue[tuple[int, int, OutputTask]] = asyncio.PriorityQueue()
        self._seq_counter = 0
        self._speaking_lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task] = None
        self._running = True
        self._tts_total_proc_s: float = 0.0  # 本轮 TTS 合成总耗时

    async def schedule(
        self,
        text: str,
        emotion: str,
        source: str,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """将消息加入播报队列。"""
        if not text or not text.strip():
            return
        self._seq_counter += 1
        task = OutputTask(priority, text, emotion, source, self._seq_counter)
        await self._queue.put((-priority.value, self._seq_counter, task))
        # 通知前端：该消息已进入输出队列
        await self.emitter.emit("output_scheduled", {
            "text": text,
            "source": source,
            "emotion": emotion,
            "priority": priority.name,
        })

    async def interrupt(self) -> None:
        """打断当前播报并清空队列。"""
        self.tts.stop()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            self._current_task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self.emitter.emit("interrupt", {})

    async def run(self) -> None:
        """持续消费队列，按优先级播报（发送音频到前端）。"""
        while self._running:
            try:
                _, _, task = await self._queue.get()
            except asyncio.CancelledError:
                break

            async with self._speaking_lock:
                if self.latency_tracker:
                    self.latency_tracker.mark_first_sound()

                # VTS 表情触发（保留原有逻辑）
                try:
                    express_emotion(action=task.emotion, duration=10.0, intensity=1.0)
                except Exception as e:
                    pass  # VTS 未连接时不影响主流程

                # 通知前端：即将播报这段文本
                await self.emitter.emit("tts_text", {
                    "text": task.text,
                    "source": task.source,
                    "emotion": task.emotion,
                })

                # TTS 合成并发送音频到前端
                self._current_task = asyncio.create_task(self._speak_async(task))
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._current_task = None

    def reset_tts_stats(self) -> None:
        """新一轮开始时重置 TTS 统计。"""
        self._tts_total_proc_s = 0.0

    def get_tts_total_proc_s(self) -> float:
        """获取本轮 TTS 合成总耗时。"""
        return self._tts_total_proc_s

    async def _speak_async(self, task: OutputTask) -> None:
        """在线程池中执行 TTS 合成，将音频通过事件发送给前端。"""
        loop = asyncio.get_event_loop()
        tts_start = time.perf_counter()
        try:
            audio_bytes = await loop.run_in_executor(
                None,
                lambda: self.tts.stream_synthesize(
                    text=task.text.strip(),
                    speed=1.0,
                    gain=0.0,
                    response_format="mp3",
                    play=False,  # 关键：不在本地播放
                ),
            )
            tts_elapsed = time.perf_counter() - tts_start
            self._tts_total_proc_s += tts_elapsed
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await self.emitter.emit("tts_audio", {
                "text": task.text,
                "audio_base64": audio_b64,
                "source": task.source,
                "emotion": task.emotion,
                "mime_type": "audio/mp3",
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  TTS 合成失败: {e}")

    # async def wait_until_idle(self) -> None:
    #     """等待当前正在播放的 TTS 完成（如果有）。

    #     用于 brain_summary 等场景，避免打断正在播放的 midway 语音。
    #     """
    #     if self._current_task and not self._current_task.done():
    #         try:
    #             await self._current_task
    #         except asyncio.CancelledError:
    #             pass

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        await self.interrupt()


# =============================================================================
# BrainAgent 状态监控器
# =============================================================================

async def brain_monitor(
    brain_bg: BackgroundBrainAgent,
    emitter: EventEmitter,
    stop_event: asyncio.Event,
    round_id: int,
) -> None:
    """定期推送 BrainAgent 状态快照到前端。

    每 0.5 秒检查一次 brain 状态，仅在状态有变化时推送，
    避免 WebSocket 消息泛滥。
    """
    last_key = ""
    while not stop_event.is_set():
        snapshot = brain_bg.brain.get_react_snapshot()
        # 用关键字段构造变化检测 key（加入 stream_buffer 长度以实现流式推送）
        key_parts = [
            str(snapshot.get("total_iters", 0)),
            str(snapshot.get("sub_status", "")),
            str(snapshot.get("latest_tool_name", "")),
            str(snapshot.get("has_used_tools", False)),
            str(len(snapshot.get("stream_buffer", ""))),
        ]
        current_key = "|".join(key_parts)

        if current_key != last_key:
            last_key = current_key
            await emitter.emit("brain_snapshot", {
                "round_id": round_id,
                **snapshot,
            })

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass  # 正常超时，继续下一轮检查


# =============================================================================
# 核心引擎
# =============================================================================

class DeerberryEngine:
    """Deerberry Web UI 核心引擎。

    封装完整的 Agent 集群、事件总线、前后台管道。
    通过 EventEmitter 将内部状态暴露给外部，实现前后端解耦。
    """

    def __init__(self) -> None:
        self.emitter = EventEmitter()
        self.user_input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._engine_task: Optional[asyncio.Task] = None

        # 以下在 start() 中初始化
        self.latency_tracker: Optional[LatencyTracker] = None
        self.scheduler: Optional[WebOutputScheduler] = None
        self.chat_agent: Optional[ChatAgent] = None
        self.emotion_agent: Optional[EmotionAgent] = None
        self.brain_agent: Optional[BrainAgent] = None
        self.reflection_agent: Optional[ReflectionAgent] = None
        self.bus: Optional[EventBus] = None
        self.brain_bg: Optional[BackgroundBrainAgent] = None
        self.front_stage: Optional[FrontStagePipeline] = None
        self.long_term_memory = None

        # 运行时状态
        self.round_num = 0
        self.current_emotion = "smile"

        # ChatAgent 面板持久化状态（避免 midway/brain_summary 覆盖 front_stage 信息）
        self._last_chat_elapsed: float = 0.0
        self._last_reflection_action: str = "-"


    # ── 公共接口 ──

    def on(self, event_type: str, handler: Callable) -> None:
        """注册事件处理器（代理到 EventEmitter）。"""
        self.emitter.on(event_type, handler)

    async def send_user_input(self, text: str) -> None:
        """外部调用：发送用户输入到引擎。"""
        await self.user_input_queue.put(text)

    async def start(self) -> None:
        """启动引擎（初始化所有组件 + 启动主循环）。"""
        self._running = True
        self._engine_task = asyncio.create_task(self._run_engine())

    async def stop(self) -> None:
        """停止引擎。"""
        self._running = False
        if self._engine_task and not self._engine_task.done():
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass

    # ── 内部初始化 ──

    async def _init(self) -> None:
        """初始化所有组件（与 main7_reflection.py 保持一致）。"""
        enable_file_logging()
        self.latency_tracker = LatencyTracker()

        # 1. TTS + WebOutputScheduler
        tts = SiliconFlowCosyVoice(
            api_key=config.TTS_API_KEY,
            api_url=config.TTS_BASE_URL,
            model=config.TTS_MODEL_NAME,
            voice=config.TTS_VOICE,
        )
        self.scheduler = WebOutputScheduler(
            tts=tts,
            emitter=self.emitter,
            latency_tracker=self.latency_tracker,
        )
        asyncio.create_task(self.scheduler.run())

        # 2. 长期记忆
        memory_cfg = config.LLM_ROLE_CONFIG.get("memory", {})
        self.long_term_memory = create_long_term_memory(
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
        set_memory_manager(self.long_term_memory)

        # 3. 按角色创建 LLM
        chat_model = build_model_for_role("chat", stream=config.STREAM)
        emotion_model = build_model_for_role("emotion", stream=config.STREAM)
        brain_model = build_model_for_role("brain", stream=config.STREAM)
        reflection_model = build_model_for_role("reflection", stream=config.STREAM)

        # 4. 初始化核心智能体
        self.chat_agent = ChatAgent(model=chat_model, agent_name=AGENT_NAME)
        self.emotion_agent = EmotionAgent(model=emotion_model)

        toolkit = Toolkit()
        toolkit.register_tool_function(retrieve_from_memory)
        toolkit.register_tool_function(record_to_memory)
        toolkit.register_tool_function(search_papers)
        toolkit.register_tool_function(read_paper)
        toolkit.register_tool_function(get_paper_details)
        toolkit.register_tool_function(search_authors)
        toolkit.register_tool_function(get_current_time)

        self.brain_agent = BrainAgent(
            model=brain_model,
            long_term_memory=self.long_term_memory,
            toolkit=toolkit,
        )
        self.reflection_agent = ReflectionAgent(model=reflection_model)

        # 5. 事件总线 + 后台 BrainAgent
        self.bus = EventBus()
        self.bus.subscribe("BrainAgent", ["user.input"])
        self.brain_bg = BackgroundBrainAgent(self.brain_agent, self.bus)
        asyncio.create_task(self.brain_bg.run())

        # 6. 前台并行管道
        self.front_stage = FrontStagePipeline(
            chat_agent=self.chat_agent,
            emotion_agent=self.emotion_agent,
            scheduler=self.scheduler,
        )

    # ── 主循环 ──

    async def _run_engine(self) -> None:
        """引擎主循环（与 main7_reflection.py 对应）。"""
        await self._init()
        await self.emitter.emit("system", {"status": "initialized"})

        round_num = 0
        current_midway_task: Optional[asyncio.Task] = None
        current_stop_event: Optional[asyncio.Event] = None
        current_monitor_task: Optional[asyncio.Task] = None
        current_emotion = "smile"

        try:
            while self._running:
                try:
                    # ── 读取用户输入 ──
                    user_input = await self.user_input_queue.get()
                    user_input = user_input.strip()
                    if not user_input:
                        continue

                    # 取消上一轮的 midway_watcher 和 monitor
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

                    if current_monitor_task and not current_monitor_task.done():
                        current_monitor_task.cancel()
                        try:
                            await current_monitor_task
                        except asyncio.CancelledError:
                            pass
                        current_monitor_task = None

                    round_num += 1
                    msg = Msg(name="user", content=user_input, role="user")

                    # 重置本轮 TTS 统计
                    self.scheduler.reset_tts_stats()

                    await self.emitter.emit("round_start", {
                        "round_id": round_num,
                        "user_input": user_input,
                    })
                    await self.emitter.emit("user_message", {
                        "round_id": round_num,
                        "text": user_input,
                    })

                    round_start = time.perf_counter()
                    self.latency_tracker.start_round(round_num, user_input)

                    # ── 打断上一轮输出 ──
                    await self.scheduler.interrupt()

                    # ── 向后台 BrainAgent 投递事件 ──
                    await self.bus.publish("user.input", UserInputEvent(
                        msg=msg, round_id=round_num
                    ))

                    # ── 前台并行轨道：ChatAgent + EmotionAgent ──
                    chat_result, emotion_result, chat_elapsed, emotion_elapsed = await self.front_stage.respond(msg)

                    # Emit ChatAgent 结果
                    if chat_result:
                        await self.emitter.emit("chat_message", {
                            "round_id": round_num,
                            "text": chat_result.get_text_content() or "",
                            "source": "chat",
                            "role": "assistant",
                        })

                    # 保存 front_stage 信息，避免被后续 midway/brain_summary 覆盖
                    self._last_chat_elapsed = chat_elapsed
                    self._last_reflection_action = "-"

                    # 推送 ChatAgent 当前 memory 到前端上下文面板（携带辅助信息）
                    await self._emit_chat_context(round_num)

                    # Emit EmotionAgent 结果
                    if emotion_result:
                        action = EmotionAgent.parse_action(
                            emotion_result.get_text_content() or ""
                        )
                        current_emotion = action
                        await self.emitter.emit("emotion_update", {
                            "round_id": round_num,
                            "emotion": action,
                            "raw": emotion_result.get_text_content() or "",
                            "elapsed": round(emotion_elapsed, 2),
                        })

                    # ── 启动 midway_watcher ──
                    threshold = self.reflection_agent.compute_dynamic_threshold(chat_result)
                    current_stop_event = asyncio.Event()
                    current_midway_task = asyncio.create_task(
                        self._midway_wrapper(
                            round_id=round_num,
                            user_input=user_input,
                            emotion=current_emotion,
                            threshold=threshold,
                            stop_event=current_stop_event,
                        )
                    )

                    # ── 启动 BrainAgent 状态监控器 ──
                    current_monitor_task = asyncio.create_task(
                        brain_monitor(
                            brain_bg=self.brain_bg,
                            emitter=self.emitter,
                            stop_event=current_stop_event,
                            round_id=round_num,
                        )
                    )

                    # ── 等待 BrainAgent 思考结果 ──
                    BRAIN_TIMEOUT = 360.0
                    try:
                        brain_output = await asyncio.wait_for(
                            self.brain_bg.output_queue.get(),
                            timeout=BRAIN_TIMEOUT,
                        )

                        # Brain 完成后停止 midway 和 monitor
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
                        if current_monitor_task and not current_monitor_task.done():
                            current_monitor_task.cancel()
                            try:
                                await current_monitor_task
                            except asyncio.CancelledError:
                                pass
                        current_monitor_task = None

                        # 【关键修复】推送最终 brain 快照，确保最后一次流式内容被前端接收
                        await self.emitter.emit("brain_snapshot", {
                            "round_id": round_num,
                            **self.brain_bg.brain.get_react_snapshot(),
                        })

                        if brain_output["status"] == "completed":
                            thought = brain_output["thought"]
                            summary_thought = thought.raw_data.get("insight", "")

                            # Emit brain 完成事件（含完整快照）
                            await self.emitter.emit("brain_summary", {
                                "round_id": round_num,
                                "insight": summary_thought,
                                "snapshot": self.brain_bg.brain.get_react_snapshot(),
                            })

                            await self._trigger_brain_summary(
                                round_id=round_num,
                                user_input=user_input,
                                summary_thought=summary_thought,
                                current_emotion=current_emotion,
                                source_label="brain_summary",
                            )

                    except asyncio.TimeoutError:
                        # 超时处理
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
                        if current_monitor_task and not current_monitor_task.done():
                            current_monitor_task.cancel()
                            try:
                                await current_monitor_task
                            except asyncio.CancelledError:
                                pass
                        current_monitor_task = None

                        # 【关键修复】推送最终 brain 快照，确保最后一次流式内容被前端接收
                        await self.emitter.emit("brain_snapshot", {
                            "round_id": round_num,
                            **self.brain_bg.brain.get_react_snapshot(),
                        })

                        snapshot = self.brain_bg.brain.get_react_snapshot()
                        parts = []
                        for it in snapshot.get("iterations", []):
                            text = it.get("reasoning_text", "")
                            if text:
                                parts.append(text)
                        stream = snapshot.get("stream_buffer", "")
                        if stream:
                            parts.append(stream)
                        summary_thought = "\n\n".join(parts)

                        self.brain_bg.cancel()

                        if summary_thought.strip():
                            await self.emitter.emit("brain_summary", {
                                "round_id": round_num,
                                "insight": summary_thought,
                                "snapshot": snapshot,
                                "timeout": True,
                            })
                            await self._trigger_brain_summary(
                                round_id=round_num,
                                user_input=user_input,
                                summary_thought=summary_thought,
                                current_emotion=current_emotion,
                                source_label="brain_summary",
                            )
                        else:
                            await self.emitter.emit("error", {
                                "round_id": round_num,
                                "message": "超时后无可用思考内容，跳过总结",
                            })

                    # ── 本轮统计 ──
                    round_elapsed = time.perf_counter() - round_start
                    tts_total_proc = self.scheduler.get_tts_total_proc_s()
                    round_without_tts = max(0.0, round_elapsed - tts_total_proc)
                    self.latency_tracker.record_agent(
                        agent_name="round_total",
                        node_type="round",
                        start_ts=round_start,
                        end_ts=time.perf_counter(),
                    )
                    report = self.latency_tracker.finish_round()
                    await self.emitter.emit("round_end", {
                        "round_id": round_num,
                        "elapsed_sec": round(round_elapsed, 2),
                        "user_perceived_s": round(report.user_perceived_s, 2) if report else 0.0,
                        "tts_total_proc_s": round(tts_total_proc, 2),
                        "round_without_tts_s": round(round_without_tts, 2),
                    })

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self.emitter.emit("error", {
                        "round_id": round_num,
                        "error_type": type(e).__name__,
                        "message": str(e),
                    })
                    import traceback
                    traceback.print_exc()
                    continue

        except asyncio.CancelledError:
            pass
        finally:
            # 优雅关闭
            try:
                await self.scheduler.stop()
            except Exception as e:
                pass
            try:
                await self.brain_bg.stop()
            except Exception as e:
                pass

    # ── midway 包装器 ──

    async def _midway_wrapper(
        self,
        round_id: int,
        user_input: str,
        emotion: str,
        threshold: float,
        stop_event: asyncio.Event,
    ) -> None:
        """包装 midway_watcher，捕获 midway 消息和 Reflection 判决并 emit 事件。"""
        # Patch chat_agent.reply 来捕获 midway 消息（在聊天框中显示）
        original_reply = self.chat_agent.reply

        async def patched_reply(msg):
            result = await original_reply(msg)
            midway_text = result.get_text_content() or ""
            if midway_text:
                await self.emitter.emit("midway_message", {
                    "round_id": round_id,
                    "text": midway_text,
                })
            return result

        self.chat_agent.reply = patched_reply
        try:
            # Patch reflection_agent.judge_each_chat 以捕获 midway 的判决
            original_judge = self.reflection_agent.judge_each_chat

            async def patched_judge(user_input, agent_response, chat_history):
                result = await original_judge(user_input, agent_response, chat_history)
                self._last_reflection_action = result.action
                # 序列化 chat_history 供前端展示
                history = []
                for msg in chat_history:
                    history.append({
                        "role": getattr(msg, "role", ""),
                        "name": getattr(msg, "name", ""),
                        "content": msg.get_text_content() or "",
                    })
                await self.emitter.emit("reflection_judgment", {
                    "round_id": round_id,
                    "action": result.action,
                    "target": result.target,
                    "agent_response": agent_response,
                    "source": "midway",
                    "chat_history": history,
                })
                return result

            self.reflection_agent.judge_each_chat = patched_judge
            try:
                await midway_watcher(
                    brain_bg=self.brain_bg,
                    chat_agent=self.chat_agent,
                    scheduler=self.scheduler,
                    reflection_agent=self.reflection_agent,
                    emotion=emotion,
                    threshold=threshold,
                    stop_event=stop_event,
                    user_name=USER_NAME,
                    user_input=user_input,
                )
            finally:
                self.reflection_agent.judge_each_chat = original_judge
        finally:
            self.chat_agent.reply = original_reply
            # midway 完成后推送当前 memory 状态（使用实例变量保留 front_stage 信息）
            await self._emit_chat_context(round_id)

    # ── brain_summary 包装器 ──

    async def _trigger_brain_summary(
        self,
        round_id: int,
        user_input: str,
        summary_thought: str,
        current_emotion: str,
        source_label: str,
    ) -> None:
        """包装 brain_summary，捕获总结消息和 Reflection 判决并 emit 事件。"""
        # 【关键修复】等待当前 TTS 播放完成，避免打断正在播放的 midway 语音
        # await self.scheduler.wait_until_idle()

        # Patch chat_agent.reply 来捕获 brain 总结消息（在聊天框中显示）
        original_reply = self.chat_agent.reply

        async def patched_reply(msg):
            result = await original_reply(msg)
            summary_text = result.get_text_content() or ""
            if summary_text:
                await self.emitter.emit("chat_message", {
                    "round_id": round_id,
                    "text": summary_text,
                    "source": source_label,
                    "role": "assistant",
                })
            return result

        self.chat_agent.reply = patched_reply
        try:
            original_judge = self.reflection_agent.judge_each_chat

            async def patched_judge(user_input, agent_response, chat_history):
                result = await original_judge(user_input, agent_response, chat_history)
                self._last_reflection_action = result.action
                # 序列化 chat_history 供前端展示
                history = []
                for msg in chat_history:
                    history.append({
                        "role": getattr(msg, "role", ""),
                        "name": getattr(msg, "name", ""),
                        "content": msg.get_text_content() or "",
                    })
                await self.emitter.emit("reflection_judgment", {
                    "round_id": round_id,
                    "action": result.action,
                    "target": result.target,
                    "agent_response": agent_response,
                    "source": source_label,
                    "chat_history": history,
                })
                return result

            self.reflection_agent.judge_each_chat = patched_judge
            try:
                await brain_summary(
                    chat_agent=self.chat_agent,
                    scheduler=self.scheduler,
                    reflection_agent=self.reflection_agent,
                    current_emotion=current_emotion,
                    user_input=user_input,
                    summary_thought=summary_thought,
                    brain_bg=self.brain_bg,
                    source_label=source_label,
                    user_name=USER_NAME,
                )
            finally:
                self.reflection_agent.judge_each_chat = original_judge
        finally:
            self.chat_agent.reply = original_reply
            # brain_summary 完成后推送当前 memory 状态（使用实例变量保留 front_stage 信息）
            await self._emit_chat_context(round_id)

    async def _emit_chat_context(self, round_id: int) -> None:
        """读取 ChatAgent 当前 memory 并推送到前端。

        右侧 ChatAgent 面板改为实时反映 memory 真实状态，
        使用实例变量保留 front_stage 的耗时/反射判决，
        避免被 midway/brain_summary 的调用覆盖为 0。
        """
        try:
            messages = await self.chat_agent.memory.get_memory()
            context = []
            for msg in messages:
                text = msg.get_text_content() or ""
                context.append({
                    "id": getattr(msg, "id", ""),
                    "role": getattr(msg, "role", ""),
                    "name": getattr(msg, "name", ""),
                    "content": text,
                })
            await self.emitter.emit("chat_context", {
                "round_id": round_id,
                "messages": context,
                "context_length": len(messages),
                "last_response_time": round(self._last_chat_elapsed, 2),
                "reflection_action": self._last_reflection_action,
            })
        except Exception as e:
            print(f"[Engine] ⚠️ _emit_chat_context 失败: {e}")


# =============================================================================
# 便捷入口
# =============================================================================

async def create_engine() -> DeerberryEngine:
    """创建并初始化引擎（但不启动主循环）。"""
    engine = DeerberryEngine()
    return engine


if __name__ == "__main__":
    # 本地测试：模拟输入
    async def test():
        engine = await create_engine()

        # 注册事件处理器，打印到控制台
        def on_event(event):
            print(f"[EVENT] {event['type']}: {str(event['data'])[:200]}")

        for et in [
            "user_message", "chat_message", "emotion_update",
            "brain_snapshot", "brain_summary", "reflection_judgment",
            "tts_text", "interrupt", "round_start", "round_end", "error",
        ]:
            engine.on(et, on_event)

        await engine.start()

        # 模拟用户输入
        await engine.send_user_input("你好，今天天气怎么样？")
        await asyncio.sleep(30)
        await engine.stop()

    asyncio.run(test())
