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

from deerberry.base.simple_agent import SimpleAgent


class OrchestratorAgent(SimpleAgent):
    """任务编排智能体。

    输入：用户消息 + 最近对话历史
    输出：TaskPlan List
    """
    
    DEFAULT_SYS_PROMPT = \
"""你是一个智能体集群的任务编排器，智能体集群正在响应用户的对话任务，你作为后台执行的决策器，你的唯一职责是判断当前用户输入的复杂度，从而编辑输出一个任务执行队列，让智能体集群更好的与用户持续对话。

### 任务队列背景简介
任务执行队列的任务根据对话任务复杂度进行编排，即简单闲聊就只需要安排简单、简短安排，复杂任务就需要多个不同功能节点的安排，例如：
1. 简单任务（欢迎语、闲聊、情绪表达、通识问答）
    原因是本次对话任务只需要其它对话智能体快速响应，不需要深度思考。
2. 复杂任务（涉及用户历史记忆核对、事实纠正、深度建议、猜测需要多步骤的推理）
    原因是本次对话问题存在一定复杂性，需要其它对话智能体先快速安抚/响应，提示智能体集群需要强的智能体进行深度思考执行，并且最终将结果进行总结
3. 任务执行顺序
    所有任务节点不管简单与否，都为异步并行执行，执行顺序只有顺序的区别，应当有限以持续对话为优先考虑。

### 任务节点列表选项
可用任务节点类型：["quick_chat", "emotion_action", "deep_think", "summary_chat"]
任务节点类型解释：
    quick_chat: 简单且可快速响应的对话智能体，用于直接回答用户问题，可快速对用户问题进行回复
    emotion_action: 简单且快速响应的表情智能体，用于智能体外部形象进行表情控制和动作控制，当对话内容需要智能体表现出更形象的表情变化时，需要本智能体进行动作控制，以更好展现智能体的情感变化
    deep_think: 推理性能强但响应时间稍长的大脑智能体，用于在智能体集群中后台异步对当前对话历史进行上年度思考，以更好的辅助对话智能体进行对话策略生成
    summary_chat: 假设经过多种智能体任务节点后，猜测任务可被智能体集群顺利完成，即设置节点对任务过程和结果进行总结并对话汇报

### 任务节点列表顺序
任务节点列表(node_list)的顺序代表了智能体集群执行任务的先后顺序，智能体集群会严格按照输出的任务节点列表顺序来执行，请合理规划任务顺序。
任务节点列表长度不受限制，但请根据实际对话任务复杂度合理规划节点数量，避免过度设计或过度简化。

### 输出格式示例
从可用任务节点类型中的各个元素进行编排，仅只输出为一个JSON数据，其中"node_list"字段存放的是编排后的任务顺序：```
例如简单对话为
{"node_list": ["quick_chat", "emotion_chat"]}

例如复杂对话为
{"node_list": ["quick_chat", "emotion_chat", "deep_think", "summary_chat"]}
```
"""
    
    

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
