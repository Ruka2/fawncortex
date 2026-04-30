"""
大脑智能体（BrainAgent）
========================
职责：
1. 深度思考用户输入，检索长期记忆
2. 生成策略洞察、用户画像、情绪分析
3. 【新增】任务队列反思：判断当前规划是否合理，是否需要重排
4. 输出 JSON 格式结果，供 TaskExecutor 解析

特性：
- 基于 ReActAgent，支持工具调用
- 挂载长期记忆检索/记录工具
- 支持生成 replan 信号
"""

import json
from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.message import Msg

from scripts.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
    clear_last_retrieved_memories,
    get_last_retrieved_memories,
)



# todo: 封装大脑智能体输出的数据结构，以此来更标准化

class BrainAgent:
    """大脑智能体封装。

    基于 ReActAgent，增加了任务反思和 JSON 输出能力。
    """

    DEFAULT_SYS_PROMPT = (
        "你是一个智能体集群的大脑中枢，负责对用户的输入进行深度思考、任务解决、工具调用等功能，最终为整个智能体集群输出此次与用户对话的策略和关键信息追踪。\n"
        "在这一次任务中，你需要根据用户的输入和历史对话中，分析用户的性格、情绪、意图等，输出对这个智能体集群的策略建议，"
        "同时你需要为智能体已经说给用户的话是否合适进行事实核对、反思，如果发现智能体说给用户的话存在事实错误或遗漏、歧义等问题，你需要输出一个策略来进行追问澄清，帮助其它智能体更好的回答。"
        "若如果对话内容的问题涉及简单、常识问题时，智能体集群个体就已经足够完成此类内容，因此遇到简单问题时可考虑忽略复杂的步骤，只输出简单的对话策略。"
        "\n"
        "## 你可以使用的工具集合：\n"
        "retrieve_from_memory: 检索与用户输入相关的历史记忆\n"
        "record_to_memory: 将有意义有价值的对话内容记录到长期记忆中\n"
        "\n"
        "## 策略标签解释\n"
        "clarification_needed: 发现其它智能体存在严重混淆对话、信息遗漏、错误执行、向用户澄清歧义、纠正错误时，需要向其他智能体反馈进行干预，请求智能体再次响应来解决用户问题，true为需要干预，false即不需要干预，枚举布尔值[true, false]\n"
        "clarification_reason: 需要澄清的内容策略，用于告知其他智能体策略内容，本变量为字符串类型\n"
        # "clarification_option: 根据本次澄清原因和内容，给定本次澄清的选项建议，ignore为智能体集群对话无任何异常，clarify为需要干预澄清，replan为智能体集群本次对话的过度设想、冗余思考，应该减少任务和精准缩短对话策略，枚举字符串[\"ignore\", \"clarify\", \"replan\"]\n"
        "clarification_option: 根据本次澄清原因和内容，给定本次澄清的选项建议。ignore为智能体集群对话无任何异常、或是最近智能体一轮回答已经足以回答用户问题了，所以无须再重复赘述一遍；clarify为智能体最近一轮回答在出现错误或者反事实，则需要干预智能体集群再一次澄清。枚举字符串[\"ignore\", \"clarify\"]\n"
        "user_profile: 用户性格、画像，本变量为字符串类型\n"
        "user_emotion: 用户现在的情绪，本变量为字符串类型\n"
        "user_intent: 用户现在的对话意图，本变量为字符串类型\n"
        "suggested_emotion: 建议其它智能体做出的行为动作、表情，枚举字符串[\"smile\", \"happy\", \"laugh\", \"sad\", \"cry\", \"angry\", \"surprise\", \"shy\", \"sleepy\", \"disgust\", \"neutral\", \"blink\", \"wink\", \"nod\", \"tilt\", \"talk\"]\n"
        "suggested_dialogue_strategy: 建议其它负责对话的智能体本次进行对话的策略建议方向，即对话策略控制，请只输出你推荐对话的策略建议，而非对话内容，本变量为字符串类型\n"
        "\n"
        "## 输出格式（严格 JSON，不要输出其他内容）\n"
        "```json\n"
        "{\n"
        '  "clarification": {'
        '    "clarification_reason": "...",\n'
        '    "clarification_needed": false,\n'
        '    "clarification_option": "ignore"\n'
        '  },\n'
        '  "user_info": {'
        '    "user_profile": "...",\n'
        '    "user_emotion": "...",\n'
        '    "user_intent": "..."\n'
        '  },\n'
        '  "suggested": {'
        '    "suggested_dialogue_strategy": "...",\n'
        '    "suggested_emotion": "neutral"\n'
        '  },\n'
        "}\n"
        "```\n"
    )

    def __init__(
        self,
        name: str = "brain_center",
        model: Optional[OpenAIChatModel] = None,
        long_term_memory=None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("BrainAgent 需要传入 model 参数")

        # 创建工具集
        toolkit = Toolkit()
        if long_term_memory is not None:
            set_memory_manager(long_term_memory)
            toolkit.register_tool_function(retrieve_from_memory)
            toolkit.register_tool_function(record_to_memory)

        self.agent = ReActAgent(
            name=name,
            sys_prompt=self.DEFAULT_SYS_PROMPT,
            model=model,
            memory=InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
            toolkit=toolkit,
        )

    # fixme: 目前brain_agent的think和reply都有上下文不能及时同步的问题，这两个函数暂时不使用
    async def reply(self, user_msg) -> Msg:
        """ 兼容 TaskExecutor 的统一调用接口，外部封装think()函数 """
        data = await self.think(user_msg)   # fixme: 
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return Msg(name=self.agent.name, content=text, role="assistant")

    # fixme: 目前brain_agent的think和reply都有上下文不能及时同步的问题，这两个函数暂时不使用
    async def think(self, user_msg) -> dict:
        """执行深度思考，返回结构化 JSON。

        Args:
            user_msg: 用户输入消息（AgentScope Msg 或原始文本）。

        Returns:
            JSON 字典。
        """
        from agentscope.message import Msg

        if isinstance(user_msg, str):
            user_msg = Msg(name="user", content=user_msg, role="user")

        # 每轮思考前清空记忆检索缓存
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

        # 提取 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                data = {
                    "clarification": {
                        "clarification_needed": False,
                        "clarification_reason": "",
                        "clarification_option": "ignore",
                    },
                    "user_info": {
                        "user_profile": "",
                        "user_emotion": "",
                        "user_intent": "",
                    },
                    "suggested": {
                        "suggested_dialogue_strategy": text[:500],
                        "suggested_emotion": "",
                    },
                }

        # 把 ReActAgent 工具调用过程中检索到的记忆附加到输出中
        data["retrieved_memories"] = get_last_retrieved_memories()  # todo: 可能存在格式问题，待优化

        return data



    async def think_with_context(
        self,
        user_msg,
        assistant_text: str = "",
    ) -> dict:
        """在 memory 已预填充 user + assistant 的情况下执行深度思考。

        用于 chat_agent 比 brain_agent 先响应的场景：
        按正确顺序（user → assistant）将本轮对话预写入 ReActAgent.memory，
        然后调用 reply(None) 直接执行 ReAct 循环（不再重复添加 user_msg）。
        """
        if isinstance(user_msg, str):
            user_msg = Msg(name="user", content=user_msg, role="user")

        # 按正确顺序预填充 memory：user → assistant
        await self.agent.memory.add(user_msg)
        if assistant_text:
            assistant_msg = Msg(
                name=self.agent.name,
                content=assistant_text,
                role="assistant",
            )
            await self.agent.memory.add(assistant_msg)

        # 执行 ReAct 循环（传入 None，避免 reply() 重复添加 user_msg）
        clear_last_retrieved_memories()

        # ── Debug: 打印 BrainAgent 输入 ──
        print(f"{'-'*60}")
        print(f"[LLM INPUT] Agent: {self.agent.name} (ReActAgent)")
        mem = await self.agent.memory.get_memory()
        print(f"大脑智能体的上下文：\n{mem}")

        result = await self.agent.reply(None)
        text = result.get_text_content()

        # ── Debug: 打印 BrainAgent 最终输出 ──
        print(f"[LLM OUTPUT] Agent: {self.agent.name} (ReActAgent)")
        # display = text[:500] + ("..." if len(text) > 500 else "")
        display = text
        print(f"{display}")
        print(f"{'-'*60}")

        # 提取 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                data = {
                    "clarification": {
                        "clarification_needed": False,
                        "clarification_reason": "",
                        "clarification_option": "ignore",
                    },
                    "user_info": {
                        "user_profile": "",
                        "user_emotion": "",
                        "user_intent": "",
                    },
                    "suggested": {
                        "suggested_dialogue_strategy": text[:500],
                        "suggested_emotion": "",
                    },
                }

        # 把 ReActAgent 工具调用过程中检索到的记忆附加到输出中
        data["retrieved_memories"] = get_last_retrieved_memories()

        return data