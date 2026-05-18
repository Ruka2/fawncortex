"""
FawnCortex Web 服务端核心引擎（main8_server.py）
===============================================
基于 chat_cli.py 改造，适配 Web UI 的事件驱动版本。
信息流走向更推荐直接学习 chat_cli.py，因为本代码基于web事件改动特别多信息流，不利于学习参考。

设计原则：
- 与前端解耦：通过 EventEmitter 事件系统暴露内部状态
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
from typing import Any, Optional, Callable
from collections import defaultdict

from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit

from fawncortex.agent.chat_agent import ChatAgent
from fawncortex.agent.emotion_agent import EmotionAgent
from fawncortex.agent.brain_agent import BrainAgent
from fawncortex.agent.reflection_agent import ReflectionAgent

from fawncortex.base.memory import create_long_term_memory
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

from fawncortex.components.voice.tts import SiliconFlowCosyVoice
from fawncortex.components.voice.asr import SiliconFlowASR
from fawncortex.components.body.vts_controller import VTSController
from fawncortex.components.body.emotion_animate import (
    animate_open_mouse,
    animate_smile,
    animate_angry,
    animate_wink_left,
    animate_wink_right,
    animate_close_eyes,
    animate_confused,
    animate_lean_left,
    animate_lean_right,
    animate_look_up,
    animate_look_down,
    animate_surprised,
    animate_smirk,
)
from fawncortex.pipeline.output_scheduler import Priority, OutputTask
from fawncortex.pipeline.event_controller import (
    EventBus,
    BackgroundBrainAgent,
    UserInputEvent,
)
from fawncortex.pipeline.front_stage_pipeline import FrontStagePipeline
from fawncortex.pipeline.back_stage_midway import midway_watcher, brain_summary

from fawncortex.logger.latency_tracker import LatencyTracker
from fawncortex.logger.logger import enable_file_logging




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

    将 FawnCortexEngine 的内部状态通过事件暴露给外部（server.py）。
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

# 表情名称 -> 动画函数映射（neural 为 None，由 idle 循环接管）
_EMOTION_ANIMATION_MAP = {
    "open_mouse": animate_open_mouse,
    "smile": animate_smile,
    "angry": animate_angry,
    "wink_left": animate_wink_left,
    "wink_right": animate_wink_right,
    "close_eyes": animate_close_eyes,
    "confused": animate_confused,
    "lean_left": animate_lean_left,
    "lean_right": animate_lean_right,
    "look_up": animate_look_up,
    "look_down": animate_look_down,
    "surprised": animate_surprised,
    "smirk": animate_smirk,
    "neural": None,
}


class WebOutputScheduler:
    """Web UI 适配的输出调度器。

    保留原有 OutputScheduler 的核心逻辑：
    - asyncio.PriorityQueue 优先级队列
    - 打断/清空能力
    - VTS 表情触发（与 TTS 并行执行）

    关键差异：
    - TTS 不再本地 sd.play() 播放，而是将音频数据通过 EventEmitter 发送给前端
    - 新增 tts_text 事件，让前端可以提前显示即将播报的文字
    """

    def __init__(
        self,
        tts: SiliconFlowCosyVoice,
        emitter: EventEmitter,
        vts: Optional[VTSController] = None,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> None:
        self.tts = tts
        self.emitter = emitter
        self.vts = vts
        self.latency_tracker = latency_tracker
        self._queue: asyncio.PriorityQueue[tuple[int, int, OutputTask]] = asyncio.PriorityQueue()
        self._seq_counter = 0
        self._speaking_lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task] = None
        self._current_emotion_task: Optional[asyncio.Task] = None
        self._running = True
        self._tts_total_proc_s: float = 0.0  # 本轮 TTS 合成总耗时

    async def schedule(
        self,
        text: str,
        emotion: str,
        tone: str,
        source: str,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """将消息加入播报队列。"""
        if not text or not text.strip():
            return
        self._seq_counter += 1
        task = OutputTask(priority, text, emotion, tone, source, self._seq_counter)
        await self._queue.put((-priority.value, self._seq_counter, task))
        # 通知前端：该消息已进入输出队列
        await self.emitter.emit("output_scheduled", {
            "text": text,
            "source": source,
            "emotion": emotion,
            "tone": tone,
            "priority": priority.name,
        })

    async def interrupt(self) -> None:
        """打断当前播报并清空队列。

        【关键修复】TTS 合成在线程池中执行同步调用，cancel() 无法中断线程池任务。
        因此使用 asyncio.wait_for 设置 5 秒超时，避免永远等待。
        """
        self.tts.stop()

        # 取消当前 TTS 和表情任务
        for attr in ("_current_task", "_current_emotion_task"):
            task = getattr(self, attr)
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                setattr(self, attr, None)

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

                # 同时启动 TTS 合成和 VTS 表情动画
                await self.emitter.emit("tts_text", {
                    "text": task.text,
                    "source": task.source,
                    "emotion": task.emotion,
                    "tone": task.tone,
                })

                tts_task = asyncio.create_task(
                    self._speak_async(task.text, task.tone, task.emotion)
                )
                emotion_task = asyncio.create_task(
                    self._express_emotion_async(task.emotion)
                )
                self._current_task = tts_task
                self._current_emotion_task = emotion_task

                try:
                    await asyncio.gather(tts_task, emotion_task)
                except asyncio.CancelledError:
                    # 确保子任务都被清理
                    for t in (tts_task, emotion_task):
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(tts_task, emotion_task, return_exceptions=True)
                finally:
                    self._current_task = None
                    self._current_emotion_task = None

    def reset_tts_stats(self) -> None:
        """新一轮开始时重置 TTS 统计。"""
        self._tts_total_proc_s = 0.0

    def get_tts_total_proc_s(self) -> float:
        """获取本轮 TTS 合成总耗时。"""
        return self._tts_total_proc_s

    async def _express_emotion_async(self, emotion: str, duration: float = 5.0) -> None:
        """异步执行 VTS 表情动画。

        Args:
            emotion: 表情动作名称（需与 _EMOTION_ANIMATION_MAP 的 key 对应）。
            duration: 动画持续时间（秒），默认 5.0 秒。
        """
        if self.vts is None:
            return

        # 设置当前表情的基础嘴型目标值（供 lip sync 叠加）
        if hasattr(self.vts, "_mouth_target"):
            from fawncortex.components.body.emotion_animate import EMOTION_MOUTH_BASE
            self.vts._mouth_target = EMOTION_MOUTH_BASE.get(emotion, 0.05)

        anim_func = _EMOTION_ANIMATION_MAP.get(emotion)
        if anim_func is None:
            return
        try:
            await anim_func(self.vts, duration=duration)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  表情动画执行失败 ({emotion}): {e}")
        finally:
            # 动画结束后重置嘴型目标值为自然状态
            if hasattr(self.vts, "_mouth_target"):
                self.vts._mouth_target = 0.05

    async def _speak_async(self, text: str, tone: str = "", emotion: str = "neural") -> None:
        """后端 PCM 流式播放 TTS，同时实时 lip sync 驱动嘴型。
        通过事件通知前端播放开始/结束，用于 UI 动画同步。
        """
        from fawncortex.components.body.emotion_animate import EMOTION_MOUTH_BASE
        base_mouth_open = EMOTION_MOUTH_BASE.get(emotion, 0.05)

        await self.emitter.emit("tts_started", {
            "text": text,
            "emotion": emotion,
            "tone": tone,
        })

        tts_start = time.perf_counter()
        try:
            await self.tts.stream_synthesize(
                text=text,
                tone=tone,
                speed=1.0,
                gain=0.0,
                response_format="pcm",
                sample_rate=44100,
                play=True,
                vts_controller=self.vts,
                base_mouth_open=base_mouth_open,
            )
            tts_elapsed = time.perf_counter() - tts_start
            self._tts_total_proc_s += tts_elapsed
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  TTS 合成/播放失败: {e}")
        finally:
            await self.emitter.emit("tts_finished", {
                "text": text,
                "emotion": emotion,
                "tone": tone,
                "duration": round(tts_elapsed, 2) if 'tts_elapsed' in locals() else 0.0,
            })

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

class FawnCortexEngine:
    """FawnCortex Web UI 核心引擎。

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
        self.asr: Optional[SiliconFlowASR] = None

        # 运行时状态
        self.round_num = 0
        self.current_emotion = "smile"

        # 名称配置（可由前端动态修改）
        self.agent_name = config.AGENT_NAME
        self.user_name = config.USER_NAME
        self.chat_model = None

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

    async def send_audio_input(self, audio_path: str) -> str:
        """外部调用：发送音频文件到引擎，经 ASR 转录后送入对话流程。

        Args:
            audio_path: 本地音频文件路径

        Returns:
            ASR 转录后的文本
        """
        if self.asr is None:
            raise RuntimeError("ASR 未初始化")
        text = await self.asr.transcribe(audio_path)
        if text:
            await self.user_input_queue.put(text)
        return text

    async def update_names(self, agent_name: str, user_name: str) -> None:
        """更新 Agent 名称和用户名，重新创建 ChatAgent 以应用新名称。

        保留旧 ChatAgent 的 memory，避免对话上下文丢失。
        """
        old_memory = self.chat_agent.memory if self.chat_agent else None
        self.agent_name = agent_name
        self.user_name = user_name

        if self.chat_model is not None:
            self.chat_agent = ChatAgent(
                model=self.chat_model,
                agent_name=agent_name,
                memory=old_memory,
            )
            print(f"[Engine] 📝 名称已更新: Agent={agent_name}, User={user_name}")
        else:
            print(f"[Engine] ⚠️ chat_model 未初始化，无法更新 Agent 名称")

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

    async def reset_memory(self) -> None:
        """清空所有智能体的短期记忆，重置会话状态。

        用于：
        1. 用户手动点击"清空聊天记录"
        2. 自动评测时每个 jsonl 样本作为新场景前重置
        """
        print("[FawnCortexEngine] 🔄 正在重置所有智能体短期记忆...")

        # 1. 中断当前输出调度器（清空 TTS 队列）
        await self.scheduler.interrupt()

        # 2. 取消当前 Brain 思考任务
        await self.brain_bg._cancel_current_think()

        # 3. 清空各 Agent 的短期记忆（Working Memory）
        await self.chat_agent.memory.clear()
        await self.emotion_agent.memory.clear()
        await self.reflection_agent.memory.clear()
        await self.brain_agent.agent.memory.clear()

        # 4. 重置 BrainAgent 的 ReAct 追踪器
        self.brain_agent.reset_react_tracker()

        # 5. 清空 BackgroundBrainAgent 的输出队列（丢弃残留结果）
        while not self.brain_bg.output_queue.empty():
            try:
                self.brain_bg.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 6. 清空 EventBus 中 BrainAgent 的输入队列
        while not self.brain_bg.input_queue.empty():
            try:
                self.brain_bg.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 7. 重置轮次计数器
        self.round_num = 0

        # 8. 重置前端状态相关缓存
        self._last_chat_elapsed = 0.0
        self._last_reflection_action = "-"
        self.current_emotion = "smile"

        # 9. 发射 reset 事件到前端（通知前端清空 UI）
        await self.emitter.emit("system", {
            "status": "reset",
            "message": "所有智能体短期记忆已清空",
        })

        print("[FawnCortexEngine] ✅ 重置完成")

    # ── 内部初始化 ──

    async def _init(self) -> None:
        """初始化所有组件（与 main7_reflection.py 保持一致）。"""
        enable_file_logging()
        self.latency_tracker = LatencyTracker()

        # 1. TTS + VTS + WebOutputScheduler
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
            print(f"[init] ⚠️  VTS 连接失败，将以无 VTS 模式运行: {e}")
            vts = None

        self.scheduler = WebOutputScheduler(
            tts=tts,
            emitter=self.emitter,
            vts=vts,
            latency_tracker=self.latency_tracker,
        )
        asyncio.create_task(self.scheduler.run())

        # 1.5 ASR 语音识别
        if config.ASR_API_KEY:
            try:
                self.asr = SiliconFlowASR()
                print("[init] ASR 已初始化")
            except Exception as e:
                print(f"[init] ⚠️  ASR 初始化失败: {e}")
        else:
            print("[init] ⚠️  ASR_API_KEY 未配置，语音输入功能不可用")

        # 2. 长期记忆
        memory_cfg = config.LLM_ROLE_CONFIG.get("memory", {})
        self.long_term_memory = create_long_term_memory(
            agent_name=self.agent_name,
            user_name=self.user_name,
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

        # 保存 chat_model，供 update_names 重新创建 ChatAgent 时使用
        self.chat_model = chat_model

        # 4. 初始化核心智能体
        self.chat_agent = ChatAgent(model=chat_model, agent_name=self.agent_name)
        self.emotion_agent = EmotionAgent(model=emotion_model)

        toolkit = Toolkit()
        toolkit.register_tool_function(retrieve_from_memory)
        toolkit.register_tool_function(record_to_memory)
        toolkit.register_tool_function(search_papers)
        toolkit.register_tool_function(read_paper)
        toolkit.register_tool_function(get_paper_details)
        toolkit.register_tool_function(online_search)
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
        current_emotion = "neural"

        try:
            while self._running:
                try:
                    # ── 读取用户输入 ──
                    user_input = await self.user_input_queue.get()
                    user_input = user_input.strip()
                    if not user_input:
                        continue

                    # 取消上一轮的 midway_watcher 和 monitor
                    # 【关键修复】midway 可能卡在 LLM 调用中，cancel() 不生效，增加 5 秒超时
                    if current_midway_task and not current_midway_task.done():
                        if current_stop_event:
                            current_stop_event.set()
                        current_midway_task.cancel()
                        try:
                            await asyncio.wait_for(current_midway_task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        current_midway_task = None
                        current_stop_event = None

                    if current_monitor_task and not current_monitor_task.done():
                        current_monitor_task.cancel()
                        try:
                            await asyncio.wait_for(current_monitor_task, timeout=3.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        current_monitor_task = None

                    round_num += 1
                    msg = Msg(name="user", content=user_input, role="user")

                    # 重置本轮 TTS 统计
                    self.scheduler.reset_tts_stats()

                    # 重置各 Agent 的 Token 统计
                    for agent in [self.chat_agent, self.emotion_agent, self.reflection_agent]:
                        if agent and hasattr(agent, 'reset_token_stats'):
                            agent.reset_token_stats()
                    if self.brain_agent and hasattr(self.brain_agent, 'reset_token_stats'):
                        self.brain_agent.reset_token_stats()
                        
                    # ── 打断上一轮输出 ──
                    await self.scheduler.interrupt()

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
                        current_emotion, current_tone = EmotionAgent.parse_action(
                            emotion_result.get_text_content() or ""
                        )
                        await self.emitter.emit("emotion_update", {
                            "round_id": round_num,
                            "emotion": current_emotion + " " + current_tone,
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
                            tone=current_tone,
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
                    # BRAIN_TIMEOUT = 300.0  # 从 360 秒的时间来完成大脑智能体的推理（非常宽松）FIXME: 将全局参数调整到代码顶部
                    try:
                        brain_output = await asyncio.wait_for(
                            self.brain_bg.output_queue.get(),
                            timeout=config.BRAIN_TIMEOUT,
                        )

                        # Brain 完成后停止 midway 和 monitor
                        # 【关键修复】所有 await 都增加超时，防止卡死
                        if current_stop_event:
                            current_stop_event.set()
                        if current_midway_task and not current_midway_task.done():
                            try:
                                await asyncio.wait_for(current_midway_task, timeout=5.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                        current_midway_task = None
                        current_stop_event = None
                        if current_monitor_task and not current_monitor_task.done():
                            try:
                                await asyncio.wait_for(current_monitor_task, timeout=3.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
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
                                current_tone=current_tone,
                                source_label="brain_summary",
                            )

                    except asyncio.TimeoutError:
                        # 超时处理
                        if current_stop_event:
                            current_stop_event.set()
                        if current_midway_task and not current_midway_task.done():
                            current_midway_task.cancel()
                            try:
                                await asyncio.wait_for(current_midway_task, timeout=5.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                        current_midway_task = None
                        current_stop_event = None
                        if current_monitor_task and not current_monitor_task.done():
                            current_monitor_task.cancel()
                            try:
                                await asyncio.wait_for(current_monitor_task, timeout=3.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
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

                    # 收集各 Agent 的 Token 统计
                    token_stats = {}
                    for name, agent in [
                        ("chat", self.chat_agent),
                        ("emotion", self.emotion_agent),
                        ("reflection", self.reflection_agent),
                    ]:
                        if agent and hasattr(agent, 'get_token_stats'):
                            token_stats[name] = agent.get_token_stats()
                    if self.brain_agent and hasattr(self.brain_agent, 'get_token_stats'):
                        token_stats["brain"] = self.brain_agent.get_token_stats()

                    await self.emitter.emit("round_end", {
                        "round_id": round_num,
                        "elapsed_sec": round(round_elapsed, 2),
                        "user_perceived_s": round(report.user_perceived_s, 2) if report else 0.0,
                        "tts_total_proc_s": round(tts_total_proc, 2),
                        "round_without_tts_s": round(round_without_tts, 2),
                        "token_stats": token_stats,
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
                if getattr(self, "vts", None):
                    print("[Shutdown] 正在关闭 VTS...")
                    await self.vts.close()
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
        tone: str,
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
                judge_start = time.perf_counter()
                result = await original_judge(user_input, agent_response, chat_history)
                judge_elapsed = time.perf_counter() - judge_start
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
                    "elapsed": round(judge_elapsed, 3),
                })
                return result

            self.reflection_agent.judge_each_chat = patched_judge
            
            emotion = "neural"  # FIXME: 固定后续不再执行动作，避免混淆
            tone = ""
            
            try:
                await midway_watcher(
                    brain_bg=self.brain_bg,
                    chat_agent=self.chat_agent,
                    scheduler=self.scheduler,
                    reflection_agent=self.reflection_agent,
                    emotion=emotion,
                    tone=tone,
                    threshold=threshold,
                    stop_event=stop_event,
                    user_name=self.user_name,
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
        current_tone: str,
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
                judge_start = time.perf_counter()
                result = await original_judge(user_input, agent_response, chat_history)
                judge_elapsed = time.perf_counter() - judge_start
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
                    "elapsed": round(judge_elapsed, 3),
                })
                return result

            self.reflection_agent.judge_each_chat = patched_judge
            
            current_emotion = "neural"  # FIXME: 固定后续不再执行动作，避免混淆
            current_tone = ""
            
            try:
                await brain_summary(
                    chat_agent=self.chat_agent,
                    scheduler=self.scheduler,
                    reflection_agent=self.reflection_agent,
                    current_emotion=current_emotion,
                    current_tone=current_tone,
                    user_input=user_input,
                    summary_thought=summary_thought,
                    brain_bg=self.brain_bg,
                    source_label=source_label,
                    user_name=self.user_name,
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
async def create_engine() -> FawnCortexEngine:
    """创建并初始化引擎（但不启动主循环）。"""
    engine = FawnCortexEngine()
    return engine


# if __name__ == "__main__":
#     # 本地测试：模拟输入
#     async def test():
#         engine = await create_engine()

#         # 注册事件处理器，打印到控制台
#         def on_event(event):
#             print(f"[EVENT] {event['type']}: {str(event['data'])[:200]}")

#         for et in [
#             "user_message", "chat_message", "emotion_update",
#             "brain_snapshot", "brain_summary", "reflection_judgment",
#             "tts_text", "interrupt", "round_start", "round_end", "error",
#         ]:
#             engine.on(et, on_event)

#         await engine.start()

#         # 模拟用户输入
#         await engine.send_user_input("你好，今天天气怎么样？")
#         await asyncio.sleep(30)
#         await engine.stop()

#     asyncio.run(test())
