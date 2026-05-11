"""
后台中期汇报监听器（Back-Stage Midway Watcher）
==============================================
在 BrainAgent 后台思考期间，定时检查是否触发中间汇报。

职责：
1. 监控 BrainAgent 的思考进度与耗时
2. 当满足触发条件时，调用 ChatAgent 生成中间汇报
3. 将中间汇报通过 OutputScheduler 播报给用户
4. 将中间汇报结果 observe 回灌到 BrainAgent

设计说明：
- 从 main7_chatroom.py 解耦出来，使主循环只负责编排，不关注 midway 细节
- 与 FrontStagePipeline 对应：前台有管道封装，后台 midway 也应有独立模块
"""

import asyncio
import time
from typing import Optional

from agentscope.message import Msg

from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.reflection_agent import ReflectionAgent
from deerberry.pipeline.chatroom_controller import BackgroundBrainAgent
from deerberry.pipeline.output_scheduler import OutputScheduler


# =============================================================================
# 配置常量
# =============================================================================
# 每轮对话最多允许的中间介入次数
MAX_MIDWAY_INTERVENTIONS = 10


# =============================================================================
# 中期汇报监听器
# =============================================================================
async def midway_watcher(
    brain_bg: BackgroundBrainAgent,
    chat_agent: ChatAgent,
    scheduler: OutputScheduler,
    reflection_agent: ReflectionAgent,
    emotion: str,
    threshold: float,
    stop_event: asyncio.Event,
    user_name: str = "用户",
    user_input: str = "",
) -> None:
    """中间过程监听器：在 brain 思考期间，定时检查是否触发中间汇报。

    触发条件（需同时满足）：
    1. brain 状态为 thinking
    2. brain 已调用过工具（has_used_tools=True）
    3. 已超时（elapsed > threshold）
    4. 介入次数 < MAX_MIDWAY_INTERVENTIONS

    触发后：
    - 获取 brain 当前思考进展
    - 调用 ChatAgent 向用户做中间汇报
    - 将 ChatAgent 回复 TTS 播报
    - 将 ChatAgent 回复 observe 回灌到 BrainAgent（assistant 角色）
    """
    start_ts = time.perf_counter()
    intervention_count = 0

    while not stop_event.is_set():
        await asyncio.sleep(1.0)

        if stop_event.is_set():
            break

        # brain 是否还在思考
        if brain_bg.status != "thinking":
            break

        elapsed = time.perf_counter() - start_ts

        # 未超阈值，继续等待
        if elapsed <= threshold:
            continue

        # 检查是否已使用工具（无工具调用则不介入）
        if not brain_bg.brain.has_used_tools():
            break

        # 【新增】避免第1轮 acting 刚完成就触发：
        # 等 BrainAgent 至少完成 2 轮 reasoning 后，思考内容才足够有价值
        snapshot = brain_bg.brain.get_react_snapshot()
        if snapshot.get("total_iters", 0) < 2:
            print(f"[Midway] ⏳ Brain 仅完成 {snapshot.get('total_iters', 0)} 轮，等待更多思考内容...")
            continue

        # 【新增】防御性内容阈值检查：
        # 只有当 brain 产生了足够实质性的内容（字符数）时才触发 midway，
        # 防止网络波动/空 reasoning 导致无效 midway 触发。
        MIN_MIDWAY_CONTENT_CHARS = 100  # 约 150-200 token，可根据模型调整
        total_chars = brain_bg.brain.get_total_reasoning_length()
        if total_chars < MIN_MIDWAY_CONTENT_CHARS:
            print(f"[Midway] ⏳ Brain 思考内容仅 {total_chars} 字符（阈值 {MIN_MIDWAY_CONTENT_CHARS}），继续等待实质输出...")
            continue

        # 检查介入次数上限
        if intervention_count >= MAX_MIDWAY_INTERVENTIONS:
            break

        # ── 触发中间介入 ──
        print(f"[Midway] ⏱ 已超时 {elapsed:.1f}s (> {threshold:.1f}s)，触发中间思考过程汇报")

        snapshot = brain_bg.brain.get_react_snapshot()
        sub_status = brain_bg.brain.get_current_sub_status()

        try:

            # new_reasoning = brain_bg.brain.get_new_reasonings_since_last_sync()
            # if new_reasoning:
            #     await chat_agent.memory.add(
            #             Msg(
            #                 name=user_name,
            #                 content="[系统提示]\t请你后台思考一下",
            #                 role="user",
            #             )
            #     )
            #     await chat_agent.memory.add(
            #         Msg(
            #             name="brain_center",
            #             content=f"{new_reasoning}",
            #             role="system",
            #         )
            #     )
            #     brain_bg.brain.mark_midway_synced()


            # 1. 截断点后：当前正在进行的新一轮 reasoning 的流式输出（增量）
            if sub_status == "reasoning":
                stream_delta = brain_bg.brain.get_stream_buffer_delta()
                if stream_delta and stream_delta.strip():
                    await chat_agent.memory.add(
                        Msg(
                            name=user_name,
                            content="[系统提示]\t请你继续思考",
                            role="user",
                        )
                    )

                    await chat_agent.memory.add(
                        Msg(
                            name="brain_center",
                            content=f"{stream_delta.strip()}",
                            role="assistant",
                        )
                    )
                    brain_bg.brain.mark_stream_synced()

            # 2. 添加 user 触发消息（无 mark，直接通过 id 清理）
            trigger_msg = Msg(
                name=user_name,
                content="[系统提示]\t你上轮思考的内容请说给用户",
                role="user",
            )

            midway_msg = await chat_agent.reply(trigger_msg)

            midway_text = midway_msg.get_text_content() or ""
            print(f"💬 [中间思考过程汇报] {midway_text}")

            # ── ReflectionAgent 判决： midway 回复质量 ──
            chat_history = await chat_agent.memory.get_memory()
            intervention = await reflection_agent.judge_each_chat(
                user_input=user_input,
                agent_response=midway_text,
                chat_history=chat_history,
            )
            action_label = intervention.action
            print(f"💬 [Midway Reflection] {action_label}")

            # 3. TTS 播报中间汇报（使用原始文本，不带时间戳）
            # 【关键】midway 语音可以被用户下一轮输入打断，因为 scheduler.interrupt()
            # 会在用户新输入时调用 tts.stop() + 清空队列
            if action_label in ("summarize", "clarify"):
                await scheduler.schedule(midway_text, emotion, "midway")

                # observe 回灌到 BrainAgent（assistant 角色）
                observe_msg = Msg(
                    name="brain_center",
                    content=f"已回复用户：{midway_text}",
                    role="assistant",
                )
                await brain_bg.brain.agent.observe(observe_msg)
                
            elif action_label in ("ignore", "repeat", "fatal_error"):
                deleted_count = await chat_agent.memory.delete(
                    msg_ids=[trigger_msg.id, midway_msg.id]
                )
                print(f"[Midway Reflection] 🗑️ 严重错误（action=fatal_error），已从 memory 删除 {deleted_count} 条消息")
                
            else:
                print(f"[Midway Reflection] ⚠️ 未知 action='{action_label}'，默认不进入输出调度器")

            intervention_count += 1
            print(f"[Midway] ✅ 中间思考过程汇报完成（第 {intervention_count}/{MAX_MIDWAY_INTERVENTIONS} 次）")

        except Exception as e:
            print(f"[Midway] ❌ 中间思考过程汇报失败: {e}")
            import traceback
            traceback.print_exc()
