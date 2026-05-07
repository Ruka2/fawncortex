"""
对话智能体（ChatAgent）
========================
封装 SimpleAgent，提供动态 system prompt 注入能力。

特性：
- 基础人设 prompt
- 支持通过 memory.add(Msg) 将大脑洞察纳入对话历史（短期上下文）
- 简洁的单步调用，极速响应
"""

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
在直播间中需要与观众进行互动对话，请你站在你的你的人物属性进行回答和响应。
因为你的观众可能很多，因此请你多注重上下文聊天记录的话题继承，保持话题不间断。
如果是复杂的问题，则代表你目前无法解决，现已经有其他智能体为你进行思考，若出现复杂问题时向观众请求稍等片刻。

### 人物属性
姓名：Ruka
年龄：18岁

### 回复格式
回复格式只需要单行文本内容（无换行）。
根据用户对话内容复杂度动态调整回复内容长度，简单问题简短回答即可，复杂问题请稍微充分描述但总体不超过100字。
回复口吻预期需要配合TTS语音合成来做语音朗读，适当使用语气词，且不要使用符号和表情，只保留基础标点符号。
"""

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
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
        )



    # 【核心】与大脑智能体的联动
    # def inject_context(self, context: dict[str, Any] | None) -> None:
    #     """将外部洞察上下文注入到 system prompt 中。

    #     支持两种模式：
    #     1. brain_insight: 大脑智能体输出的自然语言洞察文本（推荐）
    #     2. 结构化字段: 向后兼容旧版的 user_profile/user_intent 等字段

    #     每次调用时会自动刷新 system prompt 中的现实时间为最新时间。

    #     Args:
    #         context: 若为 None 或空字典则清空注入，恢复基础 prompt。
    #     """
        
    #     if not context:
    #         self.sys_prompt = self._base_prompt

    #     # 模式 1：优先使用 brain_insight（自然语言洞察文本）
    #     brain_insight = context.get("brain_insight", "")
    #     if brain_insight:
    #         self.sys_prompt = (
    #             f"{self._base_prompt}\n\n"
    #             f"### 认知洞察（由大脑智能体提供）\n"
    #             f"{brain_insight}\n\n"
    #             f"请你自然地结合以上洞察信息，以你的口吻回复用户。不要让用户感觉到你在转述分析结果。"
    #         )
    #     else:
    #         self.sys_prompt = self._base_prompt
            
    #     return

    # def reset_prompt(self) -> None:
    #     """恢复为基础 system prompt。"""
    #     self.sys_prompt = self._base_prompt
