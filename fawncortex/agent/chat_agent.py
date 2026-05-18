"""
对话智能体（ChatAgent）
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter

from fawncortex.base.simple_agent import SimpleAgent
from fawncortex.base.memory import ShortTermMemory




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
根据用户对话内容难度调整回复内容长度。例如用户的简单闲聊、简单查询则回复内容长度应该更短（20字以内）；而复杂问题则酌情以适当的字数进行描述（限制在50字以内）。
回复口吻预期需要配合TTS语音合成来做语音朗读，且不要使用复杂符号和表情，只保留使用基础标点符号。"""

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

