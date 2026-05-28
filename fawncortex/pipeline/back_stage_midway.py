"""
后台中期汇报监听器（Back-Stage Midway Watcher）
==============================================
在 BrainAgent 后台思考期间，定时检查是否触发中间汇报。

职责：
1. 监控 BrainAgent 的思考进度与耗时
2. 当满足触发条件时，调用 ChatAgent 生成中间汇报
3. 将中间汇报通过 OutputScheduler 播报给用户
4. 将中间汇报结果 observe 回灌到 BrainAgent
"""

import asyncio
import time

from agentscope.message import Msg

from fawncortex.agent.chat_agent import ChatAgent
from fawncortex.agent.reflection_agent import ReflectionAgent
from fawncortex.pipeline.event_controller import BackgroundBrainAgent
from fawncortex.pipeline.output_scheduler import OutputScheduler

import config


# =============================================================================
# 中期汇报监听器
# =============================================================================
async def midway_watcher(
    brain_bg: BackgroundBrainAgent,
    chat_agent: ChatAgent,
    scheduler: OutputScheduler,
    reflection_agent: ReflectionAgent,
    emotion: str,
    tone: str,
    threshold: float,
    stop_event: asyncio.Event,
    user_name: str = "用户",
    user_input: str = "",
    longterm_mem = None,
    round_id: int = 0,
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
        if elapsed < threshold:
            continue

        # 【占位逻辑】检查是否已使用工具（无工具调用则不介入）
        # if not brain_bg.brain.has_used_tools():
        #     break

        # 【占位逻辑】避免第1轮 acting 刚完成就触发：
        # 等 BrainAgent 至少完成 2 轮 reasoning 后，思考内容才足够有价值
        # snapshot = brain_bg.brain.get_react_snapshot()
        # if snapshot.get("total_iters", 0) < 1:  # FIXME: 2 为2轮loop才开始记录，此处先测试一下1轮
        #     print(f"[Midway] ⏳ Brain 仅完成 {snapshot.get('total_iters', 0)} 轮，等待更多思考内容...")
        #     continue

        # 防御性内容阈值检查：
        # 只有当 brain 产生了足够实质性的内容（字符数）时才触发 midway，
        # 防止网络波动/空 reasoning 导致无效 midway 触发。
        MIN_MIDWAY_CONTENT_CHARS = 100  # 约 150-200 token，可根据模型调整
        total_chars = brain_bg.brain.get_total_reasoning_length()
        if total_chars < MIN_MIDWAY_CONTENT_CHARS:
            print(f"[Midway] ⏳ Brain 思考内容仅 {total_chars} 字符（阈值 {MIN_MIDWAY_CONTENT_CHARS}），继续等待实质输出...")
            continue

        # 检查介入次数上限
        if intervention_count >= config.MAX_MIDWAY_INTERVENTIONS:
            break

        # ── 触发中间介入 ──
        print(f"[Midway] ⏱ 已超时 {elapsed:.1f}s (> {threshold:.1f}s)，触发中间思考过程汇报")

        sub_status = brain_bg.brain.get_current_sub_status()
        
        try:
            # 1. 增量采集：已完成 ReAct 轮次 + 当前流式缓冲区
            # 双指针采集，与 brain_summary 保持一致，避免多轮 reasoning 间 _stream_buffer
            # 被重置导致的重复累积问题。
            thinking_text = brain_bg.brain.get_new_reasonings_since_last_sync()

            if sub_status == "reasoning":
                stream_delta = brain_bg.brain.get_stream_buffer_delta()
                if stream_delta and stream_delta.strip():
                    if thinking_text:
                        thinking_text += "\n\n" + stream_delta.strip()
                    else:
                        thinking_text = stream_delta.strip()
                    brain_bg.brain.mark_stream_synced()

            if thinking_text and thinking_text.strip():
                # await chat_agent.memory.add(
                #     Msg(
                #         name=user_name,
                #         content="[系统提示]\t请你继续系统思考",
                #         role="user",
                #     )
                # )
                # await chat_agent.memory.add(
                #     Msg(
                #         name="brain_center",
                #         content=f"[系统思考]\t{thinking_text.strip()}",
                #         role="assistant",
                #     )
                # )

                # 标记 reasoning 轮次已同步，供 brain_summary 增量采集
                brain_bg.brain.mark_midway_synced()

                # 2. 添加 user 触发消息（无 mark，直接通过 id 清理）
                trigger_msg = Msg(
                    name=user_name,
                    # content="[系统提示]\t接着呢",
                    content=f"我从你的记忆中找到你的工作记录，请将以下有用的内容回答给我：```{thinking_text}```",
                    role="user",
                )
                
                # 让 ReflectionAgent 看到 BrainAgent 的思考过程，以便基于事实判断
                await reflection_agent.observe(trigger_msg)

                # 【对话智能体】【临时上下文清理】只保留最新最近一轮的系统提示，其余之外全部删除
                # _memory = await chat_agent.memory.get_memory()
                # _sys_msgs = [
                #     _m for _m in _memory
                #     if getattr(_m, "role", "") == "user" and "[系统提示]" in (_m.get_text_content() or "")
                # ]
                # if len(_sys_msgs) > 1:
                #     _to_delete = _sys_msgs[:-1]
                #     _deleted = await chat_agent.memory.delete(msg_ids=[_m.id for _m in _to_delete])
                #     print(f"[Memory] 🗑️ midway 已清理 {_deleted} 条旧系统提示，保留最新 1 条")

                midway_msg = await chat_agent.reply(trigger_msg)

                # 同步到 ReflectionAgent 的短期记忆
                await reflection_agent.observe(midway_msg)

                midway_text = midway_msg.get_text_content() or ""
                print(f"💬 [中间思考过程汇报] {midway_text}")

                # ── ReflectionAgent 判决： midway 回复质量 ──
                intervention = await reflection_agent.judge_each_chat(
                    user_input=user_input,
                    agent_response=midway_text,
                    round_id=round_id,
                )
                action_label = intervention.action
                print(f"💬 [Midway Reflection] {action_label}")

                # 3. TTS 播报中间汇报（使用原始文本）
                # 【关键】midway 语音可以被用户下一轮输入打断，因为 scheduler.interrupt()
                if action_label in ("summarize", "clarify"):
                    # 发送到输出调度器进行播报
                    await scheduler.schedule(midway_text, emotion, tone, "midway")
                    
                    # 同步到 ReflectionAgent 的短期记忆
                    await reflection_agent.observe(midway_msg)
                    
                    # 【关键】在 judge 之前记录 midway 输出，确保后续 brain_summary 的 judge 能看到它
                    reflection_agent.record_output(round_id, "midway", midway_text)
                    
                    # 保存 Agent 中间汇报到长期记忆
                    if longterm_mem is not None:
                        longterm_mem.save("assistant", midway_text)

                    # 【占位逻辑】目前大脑智能体的思考模式还算正常，是否需要将已对话内容回灌到大脑智能体的上下文需要考虑
                    # 因为目前实际上是chat_agent不停的去复制粘贴brain_agent的信息到自己memory中，也就是chat_agent的输出是完成跟着大脑智能体的，所以目前还不太需要考虑需要回灌信息
                    # observe 回灌到 BrainAgent（assistant 角色）
                    # observe_msg = Msg(
                    #     name="brain_center",
                    #     content=f"{midway_text}",
                    #     role="assistant",
                    # )
                    # await brain_bg.brain.agent.observe(observe_msg)
                    
                elif action_label in ("ignore", "repeat", "fatal_error", "done_yet"):
                    deleted_count = await chat_agent.memory.delete(
                        msg_ids=[midway_msg.id]
                    )
                    # 同步删除 ReflectionAgent 短期记忆中的对应消息
                    await reflection_agent.memory.delete(
                        msg_ids=[midway_msg.id]
                    )
                    print(f"[Midway Reflection] 🗑️ 忽略回答（action={action_label}），已从 memory 删除 {deleted_count} 条消息")
    
            intervention_count += 1
            # 重置计时器，控制 midway 触发频率，避免连续触发
            start_ts = time.perf_counter()
            print(f"[Midway] ✅ 中间思考过程汇报完成（第 {intervention_count}/{config.MAX_MIDWAY_INTERVENTIONS} 次）")

        except Exception as e:
            print(f"[Midway] ❌ 中间思考过程汇报失败: {e}")
            import traceback
            traceback.print_exc()


async def brain_summary(
    chat_agent: ChatAgent,
    scheduler: OutputScheduler,
    reflection_agent: ReflectionAgent,
    current_emotion: str,
    current_tone: str,
    user_input: str,
    summary_thought: str,
    brain_bg: BackgroundBrainAgent,
    user_name: str = "用户",
    longterm_mem = None,
    round_id: int = 0,
) -> None:
    """基于 brain 的思考结果触发 ChatAgent 生成总结并调度输出。

    【测试功能】只采集 brain 的中间 thinking 过程（而非最终 answer），
    将其填入 chat_agent.memory，让 ChatAgent 基于 raw thinking 组织回复。

    被 "completed" 状态和 "timeout" 状态共用，避免代码重复。
    """

    ### 【占位逻辑】大脑智能体的总结回答（非思考过程）
    # TODO: 目前此处为留档代码，因为新功能不再需要使用大脑总结，以免出现信息重复，此处留档备份
    # trigger_msg_1 = Msg(
    #     name=user_name,
    #     content="[系统提示]\t请你为用户总结思考下",
    #     role="user",
    # )
    # await chat_agent.memory.add(trigger_msg_1)
    # insight_msg = Msg(
    #     name="assistant",
    #     content=f"{summary_thought.strip()}",
    #     role="assistant",
    # )
    # await chat_agent.memory.add(insight_msg)
    # trigger_msg_2 = Msg(
    #     name=user_name,
    #     content="[系统提示]\t将你的上轮思考组织成通顺句子，承接与用户的对话",
    #     role="user",
    # )

    ### 【核心代码】增量采集：只取 midway 截断端点之后的新 thinking ──
    # (a) 已完成 ReAct 轮次中，_last_midway_sync_iter 之后的增量 reasoning
    thinking_text = brain_bg.brain.get_new_reasonings_since_last_sync()

    # (b) 当前流式缓冲区中尚未被同步的剩余内容
    stream_delta = brain_bg.brain.get_stream_buffer_delta()
    if stream_delta and stream_delta.strip():
        if thinking_text:
            thinking_text += "\n\n" + stream_delta.strip()
        else:
            thinking_text = stream_delta.strip()
        brain_bg.brain.mark_stream_synced()

    if not thinking_text or not thinking_text.strip():
        print(f"[BrainSummary] ⚠️ 思考内容为空，跳过总结触发")
        return

    print(f"[BrainSummary] 🧠 增量采集 thinking 内容 {len(thinking_text)} 字符")
    
    
    # ── 将 thinking 内容注入 chat_agent.memory ──
    # trigger_msg_1 = Msg(
    #     name=user_name,
    #     content="[系统提示]\t请你继续思考",
    #     role="user",
    # )
    # await chat_agent.memory.add(trigger_msg_1)

    # insight_msg = Msg(
    #     name="brain_center",
    #     content=f"[系统思考]\t{thinking_text}",
    #     role="assistant",
    # )
    # await chat_agent.memory.add(insight_msg)

    trigger_msg_2 = Msg(
        name=user_name,
        # content="[系统提示]\t接着呢？",
        content=f"我从你的记忆中找到你的工作记录，请将以下有用的内容回答给我：```{thinking_text}```",
        role="user",
    )
    
    # 让 ReflectionAgent 看到 BrainAgent 的思考过程，以便基于事实判断
    await reflection_agent.observe(trigger_msg_2)

    # 【对话智能体】【临时上下文清理】只保留最新最近一轮的系统提示，其余之外全部删除
    # _memory = await chat_agent.memory.get_memory()
    # _sys_msgs = [
    #     _m for _m in _memory
    #     if getattr(_m, "role", "") == "user" and "[系统提示]" in (_m.get_text_content() or "")
    # ]
    # if len(_sys_msgs) > 1:
    #     _to_delete = _sys_msgs[:-1]
    #     _deleted = await chat_agent.memory.delete(msg_ids=[_m.id for _m in _to_delete])
    #     print(f"[Memory] 🗑️ summary 已清理 {_deleted} 条旧系统提示，保留最新 1 条")

    summary_msg = await chat_agent.reply(trigger_msg_2)

    summary_text = summary_msg.get_text_content() or ""

    # ── ReflectionAgent 判决： brain 总结质量 ──
    intervention = await reflection_agent.judge_each_chat(
        user_input=user_input,
        agent_response=summary_text,
        round_id=round_id,
    )

    action_label = intervention.action
    print(f"💬 [{action_label}] {summary_text}")

    # ── ReflectionAgent 判决后的输出调度 ──
    if action_label in ("summarize", "clarify"):
        # 发送到输出调度器中进行回复
        await scheduler.schedule(summary_text, current_emotion, current_tone, "brain_summary")
        
        # 同步到 ReflectionAgent 的短期记忆
        await reflection_agent.observe(summary_msg)
        
        # 【关键】在 judge 之前记录 brain_summary 输出，确保去重能看到它
        reflection_agent.record_output(round_id, "brain_summary", summary_text)

        # 保存 Agent 总结到长期记忆
        if longterm_mem is not None:
            longterm_mem.save("assistant", summary_text)
    
    elif action_label in ("ignore", "repeat", "fatal_error", "done_yet"):
        deleted_count = await chat_agent.memory.delete(
            msg_ids=[summary_msg.id]
        )
        # 同步删除 ReflectionAgent 短期记忆中的对应消息
        await reflection_agent.memory.delete(
            msg_ids=[summary_msg.id]
        )
        print(f"[Reflection] 🗑️ 忽略回答（action={action_label}），已从 memory 删除 {deleted_count} 条消息")
        
    # 标记本次 brain_summary 已同步，下次 midway/brain_summary 从此处增量
    brain_bg.brain.mark_midway_synced()

