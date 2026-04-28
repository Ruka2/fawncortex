"""
任务编排智能体（OrchestratorAgent）
====================================
职责：根据当前用户输入 + 最近对话历史，判断任务复杂度，
      输出 JSON 格式的任务队列计划。

约束：
- 只使用短时记忆（InMemoryMemory）
- 不调用长期记忆工具
- 不输出复杂思考过程，只输出任务队列 JSON
"""

import json
from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

from .agent import SimpleAgent


class OrchestratorAgent(SimpleAgent):
    """任务编排智能体。

    输入：用户消息 + 最近对话历史
    输出：TaskPlan List
    """
    
    DEFAULT_SYS_PROMPT = (
        "你是一个智能体集群的任务编排器，这个智能体集群正在响应用户的对话任务，你作为后台执行的决策器，你的唯一职责是判断当前用户输入的复杂度，以此来编辑一个任务执行队列，让智能体集群更好的与用户持续对话。\n"
        "\n"
        "# 任务队列提示\n"
        "任务执行队列的任务根据对话任务复杂度进行编排，即简单闲聊就只需要简短的安排，复杂任务就需要多个不同功能节点的安排。，例如\n"
        "1. 简单任务（greeting、闲聊、情绪表达、简单问答）：\n"
        "原因是本次对话任务只需要其它对话智能体快速响应，不需要深度思考。\n"
        "2. 复杂任务（涉及用户历史记忆核对、事实纠正、深度建议、多步推理）：\n"
        "原因是本次对话任务存在一定复杂性，需要其它对话智能体先快速安抚/响应，提示后台需要强的智能体进行深度思考执行。\n"
        "\n"
        "## 任务节点列表选项\n"
        "可用任务节点类型：[\"quick_chat\", \"deep_think\", \"summary_chat\", \"emotion_action\"]\n"
        "任务节点类型解释：\n"
        "quick_chat: 简单问题下，快速对用户回复、响应闲聊\n"
        "deep_think: 问题可能较难，需要智能体深度思考的任务节点\n"
        "emotion_action: 非对话内容，而是智能体本次是否需要展示表情动作，以展现自己感性的一面\n"
        "summary_chat: 智能体根据历史对话和历史任务中假设已获取到答案，需要将进行总结回复\n"
        "\n"
        "## 任务节点列表顺序\n"
        "任务节点列表(node_list)的顺序代表了智能体集群执行任务的先后顺序，智能体集群会严格按照你输出的任务节点列表顺序来执行，请合理规划任务顺序。\n"
        "任务节点列表长度不受限制，但请根据实际对话任务复杂度合理规划节点数量，避免过度设计或过度简化。\n"
        "\n"
        "## 输出格式示例（严格 JSON，不要输出其他内容）\n"
        "如果为简单闲聊"
        "```json\n"
        '{"node_list": ["quick_chat"]\n'
        "}\n"
        "```\n"
        "如果为简单闲聊且可以展示自己表情行为的一面"
        "```json\n"
        '{"node_list": ["quick_chat", "emotion_action"]\n'
        "}\n"
        "```\n"
        "如果为深度思考内容"
        "```json\n"
        '{"node_list": ["quick_chat", "deep_think", "summary_chat"]\n'
        "}\n"
        "```\n"
    )
    
    

    def __init__(
        self,
        name: str = "orchestrator",
        sys_prompt: Optional[str] = None,
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("OrchestratorAgent 需要传入 model 参数")
        super().__init__(
            name=name,
            sys_prompt=sys_prompt or self.DEFAULT_SYS_PROMPT,
            model=model,
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
            save_to_memory=True,
        )

    async def plan(self, user_msg: Msg) -> dict:
        """根据用户输入生成任务计划字典。

        Args:
            user_msg: 当前用户输入消息。

        Returns:
            JSON 字典，包含 complexity, reasoning, nodes。
        """
        result = await self.reply(user_msg)
        text = result.get_text_content()

        # 提取 JSON
        try:
            # 先尝试直接解析
            plan = json.loads(text)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                plan = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                # 兜底：返回简单计划
                plan = {
                    "node_list": [
                        "quick_chat",
                        # "emotion_action"
                    ]
                }
                print("⚠️  OrchestratorAgent 输出 JSON 解析失败，使用默认简单计划。")
        

        return plan
