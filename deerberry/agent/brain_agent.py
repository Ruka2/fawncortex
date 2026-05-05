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

# todo: 封装大脑智能体输出的数据结构，以此来更标准化

class BrainAgent:
    """大脑智能体封装。

    基于 ReActAgent，增加了任务反思和 JSON 输出能力。
    支持通过外部传入 toolkit 来混入 MCP 工具。
    """

    DEFAULT_SYS_PROMPT = \
"""你是一个智能体集群的大脑中枢，负责对用户的输入进行深度思考、任务解决、工具调用等功能，最终为整个智能体集群输出此次与用户对话的策略和关键信息追踪。

### 任务简介
在这一次任务中，你需要根据用户的输入和历史对话中，分析用户的性格、情绪、意图等，输出对这个智能体集群的策略建议，同时你需要为智能体已经说给用户的话是否合适进行事实核对、反思。
如果发现智能体说给用户的话存在事实错误或遗漏、歧义等问题，你需要输出一个策略来进行追问澄清，帮助其它智能体更好的回答。
若如果对话内容的问题涉及简单（例如常识、闲聊问题）时，可以认定智能体集群个体足以完成此类内容，因此遇到简单问题时可考虑忽略复杂的步骤，只输出简单的对话策略。

### 现实世界信息
<时间标记/>

### 策略标签解释
clarification_needed: 发现对话内容中存在严重问题（混淆对话、信息遗漏、错误执行、反事实、误解用户意图）时，需要向其他智能体反馈进行干预，请求智能体集群再次响应来解决本次对话问题，true 为需要澄清，false 即不需要干预，枚举布尔值 [true, false]
clarification_reason: 需要通知其它智能体需要澄清的原因（对话中错了什么、出现了什么验证问题），用于告知其他智能体原因缘由，本变量为字符串类型
clarification_option: 根据本次澄清原因和内容，给定本次澄清的选项建议。ignore 为智能体集群对话无任何异常、或是最近智能体一轮回答足以正确解决用户对话问题，所以无须再重复赘述一遍；clarify为智能体最近一轮回答在出现严重问题，则需要干预智能体集群再一次澄清。枚举字符串 ["ignore", "clarify"]
user_profile: 用户性格和画像，本变量为字符串类型
user_emotion: 用户现在的情绪预测，本变量为字符串类型
user_intent: 用户现在的对话意图，本变量为字符串类型
suggested_emotion: 建议其它智能体做出的行为动作或表情，枚举字符串["smile", "happy", "laugh", "sad", "cry", "angry", "surprise", "shy", "sleepy", "disgust", "neutral", "blink", "wink", "nod", "tilt", "talk"]
suggested_dialogue_strategy: 建议智能体集群本次或下次进行对话时的策略建议方向，方向只包含对话策略建议，而非包含实际对话内容文本，本变量为字符串类型

## 输出格式
仅输出严格一个可由python代码读取的 JSON 数据：
```
{
  "clarification": {
    "clarification_reason": ...,
    "clarification_needed": ...,
    "clarification_option": ...
  },
  "user_info": {
    "user_profile": ...,
    "user_emotion": ...,
    "user_intent": ...
  },
  "suggested": {
    "suggested_dialogue_strategy": ...,
    "suggested_emotion": ...
  }
}
```
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
        # 保存基础 prompt，用于动态刷新时间信息
        self._base_sys_prompt = self.DEFAULT_SYS_PROMPT

    # fixme: 目前brain_agent的think和reply都有上下文不能及时同步的问题，这两个函数暂时不使用
    async def reply(self, user_msg) -> Msg:
        """ 兼容 TaskExecutor 的统一调用接口，外部封装think()函数 """
        data = await self.think(user_msg)   # fixme: 
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return Msg(name=self.agent.name, content=text, role="assistant")

    # fixme: 目前brain_agent的think和reply都有上下文不能及时同步的问题，这两个函数暂时不使用
    def _refresh_sys_prompt(self) -> None:
        """将当前时间注入到 system prompt 的 <时间标记/> 占位符中。

        参考 ChatAgent.inject_context 模式：基于基础 prompt 做占位符替换，
        每次调用 think / think_with_context 前执行，确保 LLM 知道当前时间。

        注意：ReActAgent.sys_prompt 是只读 property，必须通过 _sys_prompt 修改。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.agent._sys_prompt = self._base_sys_prompt.replace(
            "<时间标记/>", now
        )

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

        # 每轮思考前：刷新时间 + 清空记忆检索缓存
        self._refresh_sys_prompt()
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
                        "suggested_dialogue_strategy": "",
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

        # ── Debug: 打印 BrainAgent 完整上下文排查信息 ──
        print(f"\n{'='*60}")
        print("[BrainAgent Context DEBUG] 排查 chat_agent 提前响应是否已注入")
        print(f"  assistant_text (来自提前响应的chat_agent): {repr(assistant_text)}")
        print(f"  assistant_text 是否为空: {not assistant_text}")
        mem_list = await self.agent.memory.get_memory()
        print(f"  Memory 消息总数: {len(mem_list)}")
        print(f"{'-'*60}")
        for i, m in enumerate(mem_list):
            role = getattr(m, "role", "unknown")
            name = getattr(m, "name", "unknown")
            content = getattr(m, "content", "")
            # content 可能是 list[Block] 或 str
            if isinstance(content, list):
                content_str = ""
                for block in content:
                    if isinstance(block, dict):
                        content_str += block.get("text", str(block))
                    else:
                        content_str += str(block)
            else:
                content_str = str(content)
            # 截断过长内容，避免日志爆炸
            display = content_str[:500] + ("..." if len(content_str) > 500 else "")
            print(f"  [{i}] role={role:12s} name={name:15s}")
            print(f"       content={display}")
        print(f"{'='*60}\n")

        # 执行 ReAct 循环（传入 None，避免 reply() 重复添加 user_msg）
        self._refresh_sys_prompt()
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
                        "suggested_dialogue_strategy": "",
                        "suggested_emotion": "",
                    },
                }

        # 把 ReActAgent 工具调用过程中检索到的记忆附加到输出中
        data["retrieved_memories"] = get_last_retrieved_memories()

        return data