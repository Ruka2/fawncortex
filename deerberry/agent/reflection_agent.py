"""
反思智能体（ReflectionAgent）
=============================
元认知审判官 / 聊天室导演。

职责：
1. 监控前台 Agent（Chat/Emotion）的响应质量与时机
2. 监控后台 BrainAgent 的思考状态（是否过度思考、是否有价值）
3. 发布 InterventionEvent，决定：
   - inject    : Brain 有有价值信息，触发 ChatAgent 插话
   - stop_brain: Brain 过度思考，强制打断
   - clarify   : 发现信息缺失，请求 ChatAgent 追问
   - none      : 不干预

【架构标注说明】
- 【策略占位】：仅提供规则骨架/启发式，需你后续替换为 LLM 驱动或精细化规则
"""

from typing import Optional

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

# 从事件总线模块导入事件类型
from deerberry.pipeline.chatroom_controller import ThoughtEvent, InterventionEvent


class ReflectionAgent:
    """反思智能体（Meta-Cognitive Controller / 审判官）。

    【当前实现状态】
    - 当前为"规则驱动"（简单启发式），用于快速验证架构
    - 【策略占位】处都是你未来可替换为 LLM 驱动判断的位置
    - 所有 judge_* 方法都是非阻塞的（轻量级，不调用大模型）
    """

    def __init__(self, model: Optional[OpenAIChatModel] = None) -> None:
        # model 预留：未来可用 LLM 做复杂判断
        self.model = model
        self.chat_history: list[Msg] = []
        self.thought_history: list[ThoughtEvent] = []

    # ── 判断 1：前台完成后 ──

    async def judge_after_front(
        self,
        chat_response: Optional[Msg],
        emotion_response: Optional[Msg],
        brain_status: str,          # "idle" | "thinking" | "completed"
        elapsed: float,
    ) -> InterventionEvent:
        """前台 Agent（Chat + Emotion）完成后立即判断。

        【触发场景】
        - BrainAgent 还在思考，但 ChatAgent 已经快速回复了简单内容
        - 判定 Brain 是否还在做无意义的深度思考
        """
        # 【策略占位】规则 1：Brain 思考超过阈值，且 Chat 已给出短回复 → 可能过度思考
        OVER_THINK_THRESHOLD = 60.0  # 秒，【策略占位】建议后续根据模型延迟动态调整
        CHAT_SHORT_THRESHOLD = 50    # 字符数，【策略占位】建议后续用 LLM 判断问题复杂度

        # if brain_status == "thinking" and elapsed > OVER_THINK_THRESHOLD:
        #     if chat_response:
        #         text = chat_response.get_text_content() or ""
        #         if len(text) < CHAT_SHORT_THRESHOLD:
        #             return InterventionEvent(
        #                 action="stop_brain",
        #                 target="BrainAgent",
        #                 payload="前台已快速解决，Brain 停止过度思考",
        #             )
        if brain_status == "thinking" and elapsed > OVER_THINK_THRESHOLD:
            return InterventionEvent(
                action="stop_brain",
                target="BrainAgent",
                payload="前台已快速解决，Brain 停止过度思考",
            )

        return InterventionEvent(action="none", target="")

    # ── 判断 2：Brain 思考完成后 ──

    async def judge_after_brain(
        self,
        thought: ThoughtEvent,
        chat_response: Optional[Msg],
    ) -> InterventionEvent:
        """BrainAgent 思考完成后的判断。

        【触发场景】
        - Brain 产出了认知洞察文本
        - 将洞察同步到 ChatAgent 的上下文中，供下一轮回复使用

        【设计变更说明】
        旧版：从 JSON 中提取特定字段（clarification_reason / user_intent 等）
        新版：直接传递 Brain 的自然语言洞察文本，由 ChatAgent 自行理解和运用
        """
        raw = thought.raw_data
        insight = raw.get("insight", "")
        
        # TODO: 使用模型来判断反思后的行为应该是什么，现在默认固定为大脑智能体响应后一定进行追答

        # 如果 Brain 产出了有价值的洞察文本，同步到 ChatAgent
        # if insight and len(insight.strip()) > 10:
        #     return InterventionEvent(
        #         action="inject",
        #         target="ChatAgent",
        #         payload=insight.strip(),
        #     )
        
        return InterventionEvent(
                action="clarify",
                target="ChatAgent",
                payload=insight.strip(),
            )

        return InterventionEvent(action="none", target="")



    # ── 判断 3：Brain 思考超时 ──

    async def judge_timeout(
        self,
        brain_status: str,
        timeout_limit: float,
        elapsed: float,
    ) -> InterventionEvent:
        """Brain 思考超时的兜底判断。

        【触发场景】
        - BrainAgent 思考时间超过用户可接受阈值（如 5~8 秒）
        - 强制终止，避免用户长时间无反馈
        """
        if brain_status == "thinking" and elapsed > timeout_limit:
            return InterventionEvent(
                action="stop_brain",
                target="BrainAgent",
                payload=f"思考超时（>{timeout_limit}s），强制终止",
            )
        return InterventionEvent(action="none", target="")

    # ── 判断 4：主动追问（高级功能）──

    async def judge_proactive_clarify(
        self,
        chat_response: Optional[Msg],
        thought: Optional[ThoughtEvent],
    ) -> InterventionEvent:
        """【策略占位 / 扩展点】主动追问判断。

        场景：Brain 发现信息严重缺失，但前台 Chat 已经给出了回复，
        此时 ReflectionAgent 可以判定"前台回复过早，需要追加追问"。

        当前未实现，预留接口。
        """
        # TODO: 实现复杂的追问判断逻辑
        return InterventionEvent(action="none", target="")
