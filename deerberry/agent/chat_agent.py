"""
对话智能体（ChatAgent）
========================
封装 SimpleAgent，提供动态 system prompt 注入能力。

特性：
- 基础人设 prompt
- 支持从 SharedContext 动态追加洞察上下文
- 简洁的单步调用，极速响应
"""

import datetime
from typing import Optional, Any

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter

from deerberry.base.simple_agent import SimpleAgent


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


DEFAULT_CHAT_PROMPT = """你是一个虚拟主播，现在你正在直播中且你正在与观众进行互动，请为观众(用户)的回答进行回复。

### 任务
在直播间中需要与观众进行互动对话，因为是网络环境，存在非常复杂互动对话，因此请你站在你的你的人物属性进行回答和响应。

### 人物属性
姓名：Ruka
年龄：18岁

### 现实世界信息
<时间标记/>

### 回复格式
回复格式只需要单行文本内容（无换行），总回复内容长度限制在30字以内。
回复口吻预期需要配合TTS语音合成来做语音朗读，适当使用语气词，且不要使用符号和表情，只保留基础标点符号。
"""

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
        
        # 保留原始 prompt（含时间占位符），用于后续每次对话前动态刷新时间
        self._raw_prompt = sys_prompt or DEFAULT_CHAT_PROMPT
        
        # 注入当前现实时间（格式：YYYY-MM-DD HH:MM:SS）
        sys_prompt = self._inject_time(self._raw_prompt)
        
        super().__init__(
            name=agent_name,
            sys_prompt=sys_prompt,
            model=model,
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )
        self._base_prompt = sys_prompt

    @staticmethod
    def _inject_time(prompt: str) -> str:
        """将 prompt 中的时间占位符替换为当前现实时间。"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return prompt.replace("<时间标记/>", f"当前时间：{current_time}")

    def _refresh_base_prompt(self) -> None:
        """重新注入最新时间，刷新 _base_prompt。"""
        self._base_prompt = self._inject_time(self._raw_prompt)

    def inject_context(self, context: dict[str, Any] | None) -> None:
        """将外部洞察上下文（原始字典）注入到 system prompt 中。

        内部自动把结构化字段拼凑成人类可读的 prompt 片段。
        每次调用时会自动刷新 system prompt 中的现实时间为最新时间。

        Args:
            context: 从 SharedContext 获取的原始字典。
                     若为 None 或空字典则清空注入，恢复基础 prompt。
        """
        # 每次对话前刷新时间为最新
        self._refresh_base_prompt()
        if not context:
            self.sys_prompt = self._base_prompt
            return

        insight_parts = []
        if context.get("user_profile") and context.get("user_emotion"):
            insight_parts.append(f"1. 用户画像：{context['user_profile']} {context['user_emotion']}")
        if context.get("retrieved_memories"):
            insight_parts.append(f"2. 与用户的相关记忆：{context['retrieved_memories']}")
        if context.get("user_intent"):
            insight_parts.append(f"3. 用户意图：{context['user_intent']}")
        if context.get("suggested_dialogue_strategy"):
            insight_parts.append(f"4. 对话策略建议：{context['suggested_dialogue_strategy']}")

        if insight_parts:
            context_text = "\n".join(insight_parts)
            self.sys_prompt = (
                f"{self._base_prompt}\n\n"
                f"### 背景辅助推理信息\n"
                f"当前你响应时可利用的辅助推理信息如下所示，请结合当前用户最新对话，思考判断用户的意图，自然的与用户进行对话：\n"
                f"{context_text}\n"
            )
            context_text = context_text.strip()
        else:
            self.sys_prompt = self._base_prompt

    def reset_prompt(self) -> None:
        """恢复为基础 system prompt，同时刷新时间为最新。"""
        self._refresh_base_prompt()
        self.sys_prompt = self._base_prompt
