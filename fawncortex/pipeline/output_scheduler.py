"""
输出调度器（OutputScheduler）
==============================
统一输出队列：支持普通消息和插队（高优先级）消息。
负责 TTS 播报、VTS 表情触发，以及打断/清空能力。

特性：
- asyncio.PriorityQueue 实现优先级播报
- 支持打断：清空队列 + 停止当前 TTS
- 大脑澄清消息可插队到队列头部
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import random
from typing import Optional

from fawncortex.components.voice.tts import SiliconFlowCosyVoice
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..logger.latency_tracker import LatencyTracker


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


class Priority(Enum):
    """消息优先级。"""
    NORMAL = 1      # 闲聊回复、表情
    HIGH = 2        # 大脑澄清/追答（插队）

@dataclass
class OutputTask:
    """输出任务。"""
    priority: Priority
    text: str
    emotion: str
    tone: str
    source: str
    seq: int = 0

class OutputScheduler:
    """统一输出调度器。

    使用方式：
        scheduler = OutputScheduler(tts)
        asyncio.create_task(scheduler.run())  # 启动消费者
        
        # 普通消息
        await scheduler.schedule("你好", "smile", "chat")
        
        # 插队消息（大脑澄清）
        await scheduler.schedule("等一下...", "surprise", "brain_clarify", Priority.HIGH)
        
        # 打断
        await scheduler.interrupt()
    """

    def __init__(self, tts: SiliconFlowCosyVoice, vts: Optional[VTSController] = None, latency_tracker: Optional["LatencyTracker"] = None):
        self.tts = tts
        self.vts = vts
        self.latency_tracker = latency_tracker
        self._queue: asyncio.PriorityQueue[tuple[int, int, OutputTask]] = asyncio.PriorityQueue()
        self._seq_counter = 0
        self._speaking_lock = asyncio.Lock()
        self._current_tts_task: Optional[asyncio.Task] = None
        self._current_emotion_task: Optional[asyncio.Task] = None
        self._running = True

    async def schedule(
        self,
        text: str,
        emotion: str,
        tone: str,
        source: str,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """将消息加入播报队列。
        犹豫体验性而言，虽然在任务编排下表情智能体和对话智能体是分别进行且在不同时间段获取到信息的，
        schedule调度器仍然决定将对话内容和表情选项统一管理到一起发出是因为发送是因为只有响应那一刻做表情才是用户体验较好的

        Args:
            text: 要播报的文本。
            emotion: VTS 表情名称。
            source: 消息来源标识，哪一个智能体响应的。
            priority: 优先级。
        """
        if not text or not text.strip():
            return
        self._seq_counter += 1
        task = OutputTask(priority, text, emotion, tone, source, self._seq_counter)
        # PriorityQueue 按元组第一个元素排序，负数实现高优先级在前
        await self._queue.put((-priority.value, self._seq_counter, task))

    async def interrupt(self) -> None:
        """打断当前播报并清空队列。

        用户新输入到达时调用，停止所有待播报内容。
        """
        # 1. 立即停止音频播放（同步阻塞的 sd.play 需要通过 sd.stop 中断）
        self.tts.stop()

        # 2. 取消当前正在进行的 TTS 和表情任务
        for attr in ("_current_tts_task", "_current_emotion_task"):   # FIXME: 需不需要加"_current_emotion_task"任务需要debug检查
            task = getattr(self, attr)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, attr, None)

        # 3. 清空待播报队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        print("输出调度器：已打断并清空队列")

    async def run(self) -> None:
        """持续消费队列，按优先级播报。"""
        while self._running:
            try:
                _, _, task = await self._queue.get()
            except asyncio.CancelledError:
                break

            async with self._speaking_lock:
                # 标记本轮首次语音开始播放（用户角度延迟终点）
                if self.latency_tracker:
                    self.latency_tracker.mark_first_sound()

                # 同时启动 TTS 播报和 VTS 表情动画
                print(f"✅ {task.source}: {task.text}")
                tts_task = asyncio.create_task(
                    self._speak_async(task.text, task.tone, task.emotion)
                )
                emotion_task = asyncio.create_task(
                    self._express_emotion_async(task.emotion)
                )
                self._current_tts_task = tts_task
                self._current_emotion_task = emotion_task

                try:
                    await asyncio.gather(tts_task, emotion_task)
                except asyncio.CancelledError:
                    print("🔇 TTS/表情 被打断")
                    # 确保子任务都被清理
                    for t in (tts_task, emotion_task):   # FIXME: 需不需要加 emotion_task 需要DEBUG
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(tts_task, emotion_task, return_exceptions=True)
                finally:
                    self._current_tts_task = None
                    self._current_emotion_task = None

    async def _express_emotion_async(self, emotion: str) -> None:
        """异步执行 VTS 表情动画。

        Args:
            emotion: 表情动作名称（需与 _EMOTION_ANIMATION_MAP 的 key 对应）。
            duration: 动画持续时间（秒），随机 3.0~10.0 秒。
        """
        duration = random.uniform(3.0, 10.0)
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
            # 被打断时正常退出，由动画函数的 finally 恢复 idle
            raise
        except Exception as e:
            print(f"⚠️  表情动画执行失败 ({emotion}): {e}")
        finally:
            # 动画结束后重置嘴型目标值为自然状态
            if hasattr(self.vts, "_mouth_target"):
                self.vts._mouth_target = 0.05

    async def _speak_async(self, text: str, tone: str = "", emotion: str = "neural") -> None:
        """PCM 流式播放 TTS，同时实时 lip sync 驱动嘴型。"""
        from fawncortex.components.body.emotion_animate import EMOTION_MOUTH_BASE
        base_mouth_open = EMOTION_MOUTH_BASE.get(emotion, 0.05)
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
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  TTS 合成/播放失败: {e}")

    async def wait_until_idle(self) -> None:
        """等待当前正在播放的 TTS 完成（如果有）。

        用于 brain_summary 等场景，避免打断正在播放的 midway 语音。
        """
        if self._current_tts_task and not self._current_tts_task.done():
            try:
                await self._current_tts_task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        await self.interrupt()
