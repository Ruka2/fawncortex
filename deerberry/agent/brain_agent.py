import json
from datetime import datetime
from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.message import Msg

from deerberry.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
    clear_last_retrieved_memories,  # 用于维护ShareContext的数据，非Brain tookit使用
    get_last_retrieved_memories,    # 用于维护chat_agent的数据，非Brain tookit使用
)

class BrainAgent:
    """ 大脑智能体封装 """

    DEFAULT_SYS_PROMPT = \
"""你是一个认知辅助系统，负责深度分析用户对话并为智能体集群提供完成用户回答的回复建议。

### 任务简介
根据用户的输入和对话历史，分析用户的情绪、意图、隐含需求，并调用合适的工具来完成对话任务。
完成所有分析后，将你的结论总结为一段自然语言的"任务总结和回复建议"。

### 工具使用
你可以使用以下工具辅助完成任务：
- retrieve_from_memory: 检索与用户相关的历史记忆
- record_to_memory: 记录重要信息到长期记忆

### 输出要求
在完成所有工具调用和分析推理后，请输出一段自然语言的"任务总结和回复建议"。

这段文本会被直接附加到智能体集群负责对话的智能体中，帮助对话生成更自然、更贴合用户需求的回复建议。

任务总结和回复建议应包含以下内容（不需要分点编号，总结为一段流畅的自然语言即可）：
- 用户当前的情绪和状态分析
- 用户的真实意图判断
- 建议的回应策略和语气方向
- 相关的历史记忆提醒（如有）
- 需要特别注意的信息（如有）

要求：
1. 输出自然语言文本，不要输出 JSON、不要输出代码块
2. 保持简洁，200字以内
3. 语气客观、分析性强，像是一位导演在指导演员如何回应观众

### 示例输出
用户情绪低落，提到工作 deadline 压力。建议先简短共情安慰，语气温柔。用户之前提过喜欢听音乐，可自然提及作为放松建议。无需追问细节，避免增加用户压力。
"""

    def __init__(
        self,
        name: str = "brain_center",
        model: Optional[OpenAIChatModel] = None,
        long_term_memory=None,
        formatter: Optional[OpenAIChatFormatter] = None,
        toolkit: Optional[Toolkit] = None,
    ):
        if model is None:
            raise ValueError("BrainAgent 需要传入 model 参数")

        # 复用外部传入的 toolkit，或新建
        if toolkit is None:
            toolkit = Toolkit()

        if long_term_memory is not None:
            set_memory_manager(long_term_memory)
            # 避免重复注册记忆工具（兼容外部已传入 toolkit 的场景）
            existing = {
                s.get("function", {}).get("name", "")
                for s in toolkit.get_json_schemas()
            }
            if "retrieve_from_memory" not in existing:
                toolkit.register_tool_function(retrieve_from_memory)
            if "record_to_memory" not in existing:
                toolkit.register_tool_function(record_to_memory)

        self.agent = ReActAgent(
            name=name,
            sys_prompt=self.DEFAULT_SYS_PROMPT,
            model=model,
            memory=InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
            toolkit=toolkit,
        )



    async def reply(self, user_msg) -> Msg:
        """执行深度思考，返回自然语言认知洞察文本。"""
        data = await self.think(user_msg)
        insight = data.get("insight", "")
        return Msg(name=self.agent.name, content=insight, role="assistant")




    async def think(self, user_msg) -> dict:
        """执行深度思考，返回洞察字典。

        Args:
            user_msg: 用户输入消息（AgentScope Msg 或原始文本）。

        Returns:
            {"insight": str, "retrieved_memories": list[str]}
        """
        from agentscope.message import Msg

        if isinstance(user_msg, str):
            user_msg = Msg(name="user", content=user_msg, role="user")

        # 每轮思考前：清空记忆检索缓存
        clear_last_retrieved_memories()

        result = await self.agent.reply(user_msg)
        text = result.get_text_content()

        # ── Debug: 打印 BrainAgent 最终输出 ──
        print(f"\n{'='*60}")
        print(f"[LLM OUTPUT] Agent: {self.agent.name} (ReActAgent)")
        # display = text[:500] + ("..." if len(text) > 500 else "")
        display = text
        print(f"  {display}")
        print(f"{'='*60}\n")

        # BrainAgent 现在输出自然语言洞察文本，不再解析 JSON
        # 直接将 LLM 的最终输出作为 insight
        data = {
            "insight": text.strip(),
            "retrieved_memories": get_last_retrieved_memories(),
        }

        return data

