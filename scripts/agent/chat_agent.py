"""
对话智能体（ChatAgent）
========================
封装 SimpleAgent，提供动态 system prompt 注入能力。

特性：
- 基础人设 prompt
- 支持从 SharedContext 动态追加洞察上下文
- 简洁的单步调用，极速响应
"""

from typing import Optional, Any

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter

from .agent import SimpleAgent


DEFAULT_CHAT_PROMPT = (
    "你是一个AI虚拟人，你正在与用户对话，请根据用户的对话，与用户进行聊天。\n"
    "回复内容的长度取决于用户此次对话话题的难度(通常都为短句1-20个字)，只输出自然纯文本，无表情符号输出，以口头化的口吻答复。\n"
)

class ChatAgent(SimpleAgent):
    """前台对话智能体。"""

    def __init__(
        self,
        agent_name,
        sys_prompt: Optional[str] = None,
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("ChatAgent 需要传入 model 参数")
        super().__init__(
            name=agent_name,
            sys_prompt=sys_prompt or DEFAULT_CHAT_PROMPT,
            model=model,
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )
        self._base_prompt = self.sys_prompt

    def inject_context(self, context: dict[str, Any] | None) -> None:
        """将外部洞察上下文（原始字典）注入到 system prompt 中。

        内部自动把结构化字段拼凑成人类可读的 prompt 片段。

        Args:
            context: 从 SharedContext 获取的原始字典。
                     若为 None 或空字典则清空注入，恢复基础 prompt。
        """
        if not context:
            self.sys_prompt = self._base_prompt
            return

        insight_parts = []
        if context.get("user_profile") and context.get("user_emotion"):
            insight_parts.append(f"用户画像：{context['user_profile']} {context['user_emotion']}")
        if context.get("retrieved_memories"):
            insight_parts.append(f"与用户的相关记忆：{context['retrieved_memories']}")
        if context.get("user_intent"):
            insight_parts.append(f"用户意图：{context['user_intent']}")
        if context.get("suggested_dialogue_strategy"):
            insight_parts.append(f"对话策略建议：{context['suggested_dialogue_strategy']}")

        if insight_parts:
            context_text = "\n".join(insight_parts)
            self.sys_prompt = (
                f"{self._base_prompt}\n\n"
                f"## 背景辅助推理信息：\n{context_text}\n"
                f"（提示：结合当前用户的对话，思考用户的意图，自然地与用户对话）"
            )
            context_text = context_text.strip()
        else:
            self.sys_prompt = self._base_prompt

    def reset_prompt(self) -> None:
        """恢复为基础 system prompt。"""
        self.sys_prompt = self._base_prompt
