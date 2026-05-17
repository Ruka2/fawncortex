"""
前台并行管道（FrontStagePipeline）
====================================
System 1 极速响应轨道的统一封装。

职责：
1. 并行执行 ChatAgent + EmotionAgent
2. 保留"谁先完成谁先打印"的用户体验
3. 等两者都完成后，将 (text, emotion) 组合统一调度给 OutputScheduler
4. 避免 Chat 先完成时用了默认表情而导致 TTS/VTS 不匹配的问题

【设计说明】
- 与 BackgroundBrainAgent 对应：Brain 有后台包装器，前台也应该有统一的管道封装
- 主循环不再直接操作 Chat/Emotion 的裸 Task，而是调用 front_stage.respond(msg)
"""

import asyncio
import time
from typing import Any, Optional

from agentscope.message import Msg

from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.pipeline.output_scheduler import OutputScheduler


class FrontStagePipeline:
    """前台并行管道。

    将 ChatAgent + EmotionAgent 的并行执行、结果组合、输出调度内聚为一个整体。
    """

    def __init__(
        self,
        chat_agent: Any,
        emotion_agent: Any,
        scheduler: OutputScheduler,
    ) -> None:
        self.chat = chat_agent
        self.emotion = emotion_agent
        self.scheduler = scheduler

    async def _run_chat(self, msg: Msg) -> tuple[str, Msg, float]:
        """执行 ChatAgent，返回 (name, result, elapsed)。"""
        start_ts = time.perf_counter()
        result = await self.chat.reply(msg)
        elapsed = time.perf_counter() - start_ts
        return "Chat", result, elapsed

    async def _run_emotion(self, msg: Msg) -> tuple[str, Msg, float]:
        """执行 EmotionAgent，返回 (name, result, elapsed)。"""
        start_ts = time.perf_counter()
        result = await self.emotion.reply(msg)
        elapsed = time.perf_counter() - start_ts
        return "Emotion", result, elapsed

    async def respond(self, msg: Msg) -> tuple[Optional[Msg], Optional[Msg], float, float]:
        """前台并行响应。

        流程：
        1. 同时启动 ChatAgent 和 EmotionAgent
        2. 使用 as_completed 实现"谁先完成谁先打印"
        3. 等两者都完成后，统一调度 (text, emotion) 给 OutputScheduler
        4. 返回 (chat_result, emotion_result, chat_elapsed, emotion_elapsed) 供上层继续使用

        Args:
            msg: 用户输入消息。

        Returns:
            (chat_result, emotion_result, chat_elapsed, emotion_elapsed)：
            前两者可能为 None（异常时），后两者为各自的响应耗时（秒）。
        """
        chat_result: Optional[Msg] = None
        emotion_result: Optional[Msg] = None
        chat_elapsed: float = 0.0
        emotion_elapsed: float = 0.0

        # 1. 并行启动两个 Agent
        for coro in asyncio.as_completed([
            self._run_chat(msg),
            self._run_emotion(msg),
        ]):
            name, result, elapsed = await coro

            if name == "Chat":
                chat_result = result
                chat_elapsed = elapsed
                text = result.get_text_content() or ""
                print(f"\n💬 ChatAgent  ({elapsed:.2f}s)\n{text}")

            elif name == "Emotion":
                emotion_result = result
                emotion_elapsed = elapsed
                action = EmotionAgent.parse_action(result.get_text_content() or "")
                print(f"😊 EmotionAgent  ({elapsed:.2f}s)  →  {action}")

        # 2. 两者都完成后，统一调度输出（确保 text 和 emotion 匹配）
        if chat_result is not None and emotion_result is not None:
            text = chat_result.get_text_content() or ""
            emotion, tone = EmotionAgent.parse_action(emotion_result.get_text_content() or "")
            await self.scheduler.schedule(text, emotion, tone, "chat")
        elif chat_result is not None:
            # Emotion 异常失败，只用默认表情兜底
            text = chat_result.get_text_content() or ""
            await self.scheduler.schedule(text, "neural", "", "chat")

        return chat_result, emotion_result, chat_elapsed, emotion_elapsed
