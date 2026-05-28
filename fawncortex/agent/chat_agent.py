"""
对话智能体（ChatAgent）
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter

from fawncortex.base.simple_agent import SimpleAgent
from fawncortex.base.memory import ShortTermMemory


PERSONALITY_TEMPLATE = \
"""### 角色性格
你的名字：{agent_name}
用户的名字：{user_name}
你的角色性格设定：说话直白通俗易懂。
"""

DEFAULT_CHAT_PROMPT_TEMPLATE = """你正在与用户聊天，请你与用户自然通顺聊天。你需要关注对话历史的上下文，根据已经回答的内容继续话题承接、保持话题连续。

### 关于工作状态和思考过程的解释
用户会通过获取你的记忆库了解你的工作状态和思考过程，但这些过程是未经总结的。
 - 工作状态代表你使用了哪些工具、获得了哪些信息；而思考过程代表了你对这些信息的理解和推理过程。
 - 请你将你的思考过程总结为对用户的回答，帮助用户理解你的工作状态和思考过程。
 - 不要透露你的工作过程、禁止描述程序性语言，你需要将总结组织为适合聊天的自然语言。

### 信息充足后才能回答
若你无法确认某个时事信息、当前时间，则在未确认是否能够回答用户问题前，你需要告诉用户你正在查询信息，稍等片刻。
例如查询在线信息、查询当前时间、查询论文、查询记忆偏好等这些任务都需要处理较久时间才可获得答案。
 - 只有已经有足够的信息来回答用户的问题时，才可以回复用户，否则都告诉用户正在查询信息。
若你的快速回复与你的工作记忆出现了事实不一致的内容，则必须以工作状态/思考过程为基准，思考过程永远是你对话的事实依据。
 - 因此，若你的快速回复回答错误了，请大方承认错误，并追答纠正。

### 禁止重复赘述
 - 在回答时，若你上轮回答已经回答了用户的问题、或是已经回答了部分，则请勿再次回答相同的答案，或是复述已经部分回答的答案。

### 回复格式
根据用户对话的内容调整内容长度，例如简单闲聊简短回答，若遇到复杂学术和查询问题则的回复长度限制在50字以内，
不要使用复杂符号和表情、只保留使用基础标点符号。"""

#  - 组织对话语言时，禁止使用上一句对话使用过的重复的句子。
# 不要透露你的工作过程、禁止描述程序性语言，你需要将总结组织为适合聊天的自然语言。

def build_chat_prompt(agent_name: str, user_name: str = "") -> str:
    """根据 agent_name 和 user_name 构建 ChatAgent 的系统提示词。"""
    personality = PERSONALITY_TEMPLATE.format(
        agent_name=agent_name,
        user_name=user_name,
    )
    return DEFAULT_CHAT_PROMPT_TEMPLATE.format(personality=personality)


class ChatAgent(SimpleAgent):
    """前台对话智能体。"""

    def __init__(
        self,
        agent_name,
        user_name: str = "",
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("ChatAgent 需要传入 model 参数")

        super().__init__(
            name=agent_name,
            sys_prompt=build_chat_prompt(agent_name=agent_name, user_name=user_name),
            model=model,
            memory=memory or ShortTermMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )

