"""
反思智能体（ReflectionAgent）
=============================
元认知审判官 / 聊天室导演。

基于 SimpleAgent（单步 LLM 调用，无 ReAct 循环），快速判断：
- BrainAgent 的思考结果是否需要同步给用户
- 前台 Agent 的响应质量与时机

职责：
1. 监控前台 Agent（Chat/Emotion）的响应质量与时机
2. 监控后台 BrainAgent 的思考状态（是否过度思考、是否有价值）
3. 发布 InterventionEvent，决定：
   - summarize : Brain 有 Chat 未提及的新事实，触发总结插话
   - ignore    : Chat 已正确回答，Brain 结果无需再提
   - clarify   : 发现对话中智能体可能存在信息缺失情况，请求 ChatAgent 追问请求用户补足信息
   - stop_brain: Brain 过度思考，强制打断
   - none      : 不干预
"""

import json
from typing import Optional

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter

from deerberry.base.simple_agent import SimpleAgent
from deerberry.pipeline.chatroom_controller import ThoughtEvent, InterventionEvent


class ReflectionAgent(SimpleAgent):
    """反思智能体（Meta-Cognitive Controller / 审判官）。
    基于 SimpleAgent 实现：单步 LLM 调用，无 ReAct 循环，快速完成判断。
    """

    DEFAULT_SYS_PROMPT = \
"""你是智能体集群的任务编排器，你的职责是判断后台思考智能体(Brain Agent)完成深度思考后，其深度思考的结果是否有需要告知给用户。

输入信息：
1. 用户的对话内容
2. 对话智能体(Chat Agent)已经给出的回复
3. 后台思考智能体(Brain Agent)的深度思考结果（可能包含工具查询到的数据）

判断规则：
- summarize: Brain Agent 提供了 Chat Agent 没有提到的新事实/数据/结论，需要总结告知用户
- clarify: Brain Agent 发现对话中用户可能存在信息缺失或智能体理解模糊，需要请求 Chat Agent 响应向用户追问澄清
- ignore: Chat Agent 已经正确完整地回答了用户，Brain Agent 的结果只是重复或补充说明，则不需要再次响应 Chat Agent，避免打扰用户或出现重复回答

你只能输出一个 token 标记，从以下枚举值进行选取: ["summarize", "ignore", "clarify"]
"""

    def __init__(self, model: Optional[OpenAIChatModel] = None) -> None:
        if model is None:
            raise ValueError("ReflectionAgent 需要传入 model 参数")

        super().__init__(
            name="reflection",
            sys_prompt=self.DEFAULT_SYS_PROMPT,
            model=model,
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
            save_to_memory=False,
        )

        self.chat_history: list[Msg] = []
        self.thought_history: list[ThoughtEvent] = []



    # ── 判断 1：前台完成后（保持规则驱动，轻量快速）──
    # 目前这个判断不知道有什么用，因为后台大脑智能体一直是常驻开着的，前台响应无论是否正常都不影响大脑智能体运行
    async def judge_after_front(
        self,
        chat_response: Optional[Msg],
        emotion_response: Optional[Msg],
        brain_status: str,
        elapsed: float,
    ) -> InterventionEvent:
        """前台 Agent（Chat + Emotion）完成后立即判断。

        当前保持规则驱动：Brain 思考超时则强制打断。
        """
        OVER_THINK_THRESHOLD = 60.0

        if brain_status == "thinking" and elapsed > OVER_THINK_THRESHOLD:
            return InterventionEvent(
                action="stop_brain",
                target="BrainAgent",
                payload="前台已快速解决，Brain 停止过度思考",
            )
        return InterventionEvent(action="none", target="")



    # ── 判断 2：Brain 思考完成后（LLM 驱动）──
    async def judge_after_brain(
        self,
        thought: ThoughtEvent,
        chat_response: Optional[Msg],
        user_question: str = "",
        chat_history: Optional[list[Msg]] = None,
    ) -> InterventionEvent:
        """BrainAgent 思考完成后的 LLM 驱动判断。

        Args:
            thought: BrainAgent 思考完成事件
            chat_response: ChatAgent 的前台回复
            user_question: 用户的原始问题文本
            chat_history: 外部传入的对话历史（方案 A）；若为 None，则从 self.memory 读取（方案 C）

        Returns:
            InterventionEvent，action 可能为 summarize / ignore / clarify
        """
        chat_text = chat_response.get_text_content() if chat_response else ""
        brain_text = thought.raw_data.get("insight", "") if thought else ""

        # ── 方案 A + C：获取对话历史 ──
        # 优先使用外部传入的 chat_history，否则从 self.memory 读取（observe 积累）
        history_msgs = chat_history if chat_history is not None else await self.memory.get_memory()

        # 构造当前审查消息
        review_content = f"""请根据以下对话内容、已回复内容、思考结果，输出任务编排判断：

【用户本轮对话内容】：
{user_question}

【对话Chat Agent已回复】：
{chat_text}

【后台Brain Agent思考结果】：
{brain_text}
"""

        # ── 直接构造完整 prompt（system + 历史 + 当前审查）──
        # 不经过 self.reply()，因为 save_to_memory=False 会导致当前 msg 被丢弃
        messages = [
            Msg("system", self.sys_prompt, "system"),
            *history_msgs,
            Msg(name="user", content=review_content, role="user"),
        ]
        prompt = await self.formatter.format(messages)
        await self.print_llm_prompt(prompt)

        # 直接调用模型
        response = await self.model(prompt)
        result_text = await self._extract_content(response)
        await self.print_llm_response(result_text)

        # ── 解析 token 输出 ──
        text = result_text.strip().lower()

        # 取第一个有效词作为 action（LLM 可能输出换行或额外空格）
        action = text.split()[0] if text else "ignore"

        # payload 固定使用 BrainAgent 的洞察原文
        payload = brain_text

        if action == "summarize":
            return InterventionEvent(
                action="summarize",
                target="ChatAgent",
                payload=payload,
            )
        elif action == "clarify":
            return InterventionEvent(
                action="clarify",
                target="ChatAgent",
                payload=payload,
            )
        else:
            # 包括 "ignore" 或任何无法识别的 token
            return InterventionEvent(
                action="ignore",
                target="",
                payload="",
            )

    # ── 判断 3：Brain 思考超时（规则驱动）──
    async def judge_timeout(
        self,
        brain_status: str,
        timeout_limit: float,
        elapsed: float,
    ) -> InterventionEvent:
        """Brain 思考超时的兜底判断。"""
        if brain_status == "thinking" and elapsed > timeout_limit:
            return InterventionEvent(
                action="stop_brain",
                target="BrainAgent",
                payload=f"思考超时（>{timeout_limit}s），强制终止",
            )
        return InterventionEvent(action="none", target="")
