"""
对话智能体（ChatAgent）
========================
封装 SimpleAgent，提供动态 system prompt 注入能力。

特性：
- 基础人设 prompt
- 支持从 SharedContext 动态追加洞察上下文
- 简洁的单步调用，极速响应
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

from .agent import SimpleAgent


DEFAULT_CHAT_PROMPT = (
    "你是一个AI虚拟人，你正在与用户对话，请根据用户的对话，与用户进行聊天。\n"
    "回复内容的长度取决于用户此次对话话题的难度(通常1~20字)，只输出自然纯文本，无符号输出，请以口头话的场景口吻答复。\n"
)


class ChatAgent(SimpleAgent):
    """前台对话智能体。"""

    def __init__(
        self,
        name: str = "小花",
        sys_prompt: Optional[str] = None,
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("ChatAgent 需要传入 model 参数")
        super().__init__(
            name=name,
            sys_prompt=sys_prompt or DEFAULT_CHAT_PROMPT,
            model=model,
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )
        self._base_prompt = self.sys_prompt

    def inject_context(self, context: str) -> None:
        """将外部洞察上下文注入到 system prompt 中。

        Args:
            context: 从 SharedContext 获取的 prompt 片段。
        """
        if context:
            self.sys_prompt = f"{self._base_prompt}\n\n{context}\n以上信息仅供参考，请继续自然地与用户对话。"
        else:
            self.sys_prompt = self._base_prompt

    def reset_prompt(self) -> None:
        """恢复为基础 system prompt。"""
        self.sys_prompt = self._base_prompt
