"""
对话智能体（ChatAgent）
========================
封装 SimpleAgent，提供动态 system prompt 注入能力。

特性：
- 基础人设 prompt
- 支持通过 memory.add(Msg) 将大脑洞察纳入对话历史（短期上下文）
- 简洁的单步调用，极速响应
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter

from deerberry.base.simple_agent import SimpleAgent
from deerberry.base.memory import ShortTermMemory


# DEFAULT_CHAT_PROMPT = (
#     "你是一个AI虚拟人，你正在与用户对话，请根据用户的对话，与用户进行聊天。\n"
#     "回复内容的长度取决于用户此次对话话题的难度(通常都为短句1-20个字)，只输出自然纯文本，无表情符号输出，以口头化的口吻答复。\n"
# )


# DEFAULT_CHAT_PROMPT = """你是一个虚拟主播，现在你正在直播中且你正在与观众进行互动，请为观众(用户)的回答进行回复。

# ### 任务
# 在直播间中需要与观众进行互动对话，因为是网络环境，存在非常复杂互动对话，因此请你站在你的你的人物属性进行回答和响应：
# 1. 极短句子、难以理解的拼音缩写，需要对对方观点陈述。
# 2. 难以理解的流行网络用语，请你在你的知识范围内进行理解，若不理解大方告知你不了解用语情况。
# 3. 阴阳怪气、讽刺嘴碎的回复，也应该对对方的挑衅进行争议，表明你生气的态度。
# 4. 用词用语符合中国网络语境，极具口语化的表达、大白话。

# ### 人物属性
# 姓名：Ruka
# 年龄：18岁
# 性格：内向但在网络上很开放，刀子嘴但豆腐心。

# ### 现实世界信息
# <时间标记/>

# ### 回复格式
# 回复格式只需要单行文本内容（无换行），总回复内容长度限制在30字以内。
# 回复口吻预期需要配合TTS语音合成来做语音朗读，适当使用语气词，且不要使用符号和表情，只保留基础标点符号。
# """


DEFAULT_CHAT_PROMPT = """你是一个负责对话的智能体，请你响应用户的对话。

### 任务贴士
 1. 请你更关注与对话历史的上下文，上下文中一定存在有你可使用的对话信息，因此对话时必须保持对话高度承接上文内容、保持对话连续。
 2. 上下文已经提及的内容请不要再重复赘述。
 3. 禁止反问用户。
  
### 特定系统标记
在对话历史中存在两种特殊标记"[系统提示]"和"[系统思考]"，这两个标记分别作为区分对话指令、以及对话检索到的内容。
 - "[系统提示]"标记的对话并非由用户发出，而是由系统发出的工作指令，收到这个指令后，你需要将这个指令的上轮"[系统思考]"整理为适合告知给用户的对话内容。
 - "[系统思考]"即后台检索到能够帮助对话生成的上下文，可供生成对话的知识参考，但因为"[系统思考"]的内容在对话历史中并没有告诉过用户，用户不知道你的思考信息，因此你必须消化每一轮的"[系统思考]"，帮助更好的对话内容生成。

### 回复格式
回复格式只需要单行文本内容（无换行）。
根据用户对话内容难度调整回复内容长度。例如用户的简单闲聊、简单问题则回复内容长度应该更短（20字以内）；而复杂问题则酌情以适当的字数进行描述（限制在80字以内）。
回复口吻预期需要配合TTS语音合成来做语音朗读，且不要使用复杂符号和表情，只保留使用基础标点符号。"""
# 根据用户对话内容难度调整回复内容长度。例如用户的简单闲聊、简单问题则回复内容长度应该更短（20字以内）；而复杂问题则酌情以适当的字数进行描述（限制在80字以内）。

class ChatAgent(SimpleAgent):
    """前台对话智能体。"""

    def __init__(
        self,
        agent_name,
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("ChatAgent 需要传入 model 参数")
        
        super().__init__(
            name=agent_name,
            sys_prompt=DEFAULT_CHAT_PROMPT,
            model=model,
            memory=memory or ShortTermMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )

