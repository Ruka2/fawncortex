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



### 中间汇报（Midway Intervention）配置
# 动态阈值基础值（秒）
MIDWAY_BASE_THRESHOLD = float(5.0)
# 动态阈值上限（秒）
MIDWAY_MAX_THRESHOLD = float(30.0)
# 阈值随前台回复长度增长的系数（每字符增加的秒数）
MIDWAY_THRESHOLD_FACTOR = float(0.1)

class ReflectionAgent(SimpleAgent):
    """反思智能体（Meta-Cognitive Controller / 审判官）。
    基于 SimpleAgent 实现：单步 LLM 调用，无 ReAct 循环，快速完成判断。
    """

    DEFAULT_SYS_PROMPT = \
"""你是智能体集群的任务编排器，你的职责是判断后台思考智能体(Brain Agent)完成深度思考后，其深度思考的结果结合最近上下文智能体的回答中，是否还有需要告知给用户。

判断信息：
1. 与用户的历史对话内容
2. 后台思考智能体(Brain Agent)的最新的深度思考结果

判断规则：
- summarize: Brain Agent 在最近上下文的背景下，新的结论提供了没有提到的新事实/数据/结论，需要总结告知用户
- clarify: Brain Agent 发现最近上下文对话中，用户可能存在信息缺失或智能体理解模糊，需要请求响应向用户追问澄清
- ignore: 最近上下文对话历史中，最近一轮对话已经正确解答用户的问题，而 Brain Agent 最新的思考结果只是重复或赘述，则不需要再次响应，避免打扰用户或出现重复回答

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

    # ── 动态阈值计算（供 midway_watcher 调用）──
    @staticmethod
    def compute_dynamic_threshold(chat_result: Optional[Msg]) -> float:
        """根据前台对话长度计算动态阈值。

        逻辑：
        - 前台回复越短 → 用户问题越简单 → 容忍时间越短
        - 前台回复越长 → 用户问题越复杂 → 容忍时间越长

        formula: threshold = BASE + chat_length * FACTOR, capped at MAX
        """
        
        base = MIDWAY_BASE_THRESHOLD
        max_threshold = MIDWAY_MAX_THRESHOLD
        factor = MIDWAY_THRESHOLD_FACTOR

        chat_text = chat_result.get_text_content() if chat_result else ""
        token_count = len(chat_text)  # 简化为字符数，后续可替换为真实 token 数

        threshold = base + token_count * factor
        threshold = min(threshold, max_threshold)
        
        # threshold = float(5.0)

        return threshold

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
        chat_response = chat_response.get_text_content() if chat_response else ""
        brain_text = thought.raw_data.get("insight", "") if thought else ""


        review_content = f"""请根据以下Brain Agent的思考结果，输出本轮任务编排判断：
【用户最开始提问的问题】：
{chat_response}
        
【Brain Agent思考结果】
{brain_text}"""

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
