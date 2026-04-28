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
from typing import Optional

from agentscope.message import Msg

from components.tts import SiliconFlowCosyVoice
from scripts.tools.control_vts import express_emotion


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
    source: str  # "chat" / "brain_clarify" / "brain_emit"
    # 用于排序：同优先级下按入队时间排序
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

    def __init__(self, tts: SiliconFlowCosyVoice):
        self.tts = tts
        self._queue: asyncio.PriorityQueue[tuple[int, int, OutputTask]] = asyncio.PriorityQueue()
        self._seq_counter = 0
        self._speaking_lock = asyncio.Lock()
        self._current_tts_task: Optional[asyncio.Task] = None
        self._running = True

    async def schedule(
        self,
        text: str,
        emotion: str,
        source: str,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """将消息加入播报队列。

        Args:
            text: 要播报的文本。
            emotion: VTS 表情名称。
            source: 消息来源标识。
            priority: 优先级，大脑澄清用 HIGH。
        """
        if not text or not text.strip():
            return
        self._seq_counter += 1
        task = OutputTask(priority, text.strip(), emotion, source, self._seq_counter)
        # PriorityQueue 按元组第一个元素排序，负数实现高优先级在前
        await self._queue.put((-priority.value, self._seq_counter, task))

    async def interrupt(self) -> None:
        """打断当前播报并清空队列。

        用户新输入到达时调用，停止所有待播报内容。
        """
        # 1. 取消当前正在进行的 TTS 任务
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
            try:
                await self._current_tts_task
            except asyncio.CancelledError:
                pass
            self._current_tts_task = None

        # 2. 清空待播报队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        print("🔇 输出调度器：已打断并清空队列")

    async def run(self) -> None:
        """持续消费队列，按优先级播报。"""
        while self._running:
            try:
                _, _, task = await self._queue.get()
            except asyncio.CancelledError:
                break

            async with self._speaking_lock:
                # 触发表情
                try:
                    express_emotion(action=task.emotion, duration=3.0, intensity=1.0)
                except Exception as e:
                    print(f"⚠️  表情触发失败: {e}")

                # TTS 播报（包装为可取消的任务）
                print(f"🔊 [{task.source}] {task.text}")
                self._current_tts_task = asyncio.create_task(
                    self._speak_async(task.text)
                )
                try:
                    await self._current_tts_task
                except asyncio.CancelledError:
                    print("🔇 TTS 被打断")
                finally:
                    self._current_tts_task = None

    async def _speak_async(self, text: str) -> None:
        """在线程池中执行 TTS，避免阻塞事件循环。"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.tts.stream_synthesize(
                    text=text.strip(),
                    voice="FunAudioLLM/CosyVoice2-0.5B:diana",
                    speed=1.0,
                    gain=0.0,
                    response_format="mp3",
                    play=True,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  TTS 合成/播放失败: {e}")

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        await self.interrupt()
