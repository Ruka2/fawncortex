
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

import inspect
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter


class SimpleAgent(AgentBase):
    """最简自定义智能体：单步调用大模型，无 ReAct 循环，本身不处理工具调用。
    本身不处理工具调用，如需触发外部动作（如 VTS），在拿到回复后由外部代码处理。
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: OpenAIChatModel,
        memory: MemoryBase,
        formatter: OpenAIChatFormatter,
        save_to_memory: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory = memory
        self.formatter = formatter
        self.save_to_memory = save_to_memory

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        """接收消息 → 调用 LLM → 返回回复。"""
        if msg is not None:
            if self.save_to_memory:  # fixme: 同时额外追加的一层条件
                await self.memory.add(msg)

        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )

        # 测试DEBUG llm prompt
        await self.print_llm_prompt(prompt)

        # 调用模型：兼容 stream=True（异步生成器）和 stream=False（直接对象）
        response = await self.model(prompt)
        content = await self._extract_content(response)

        # 测试DEBUG llm response
        await self.print_llm_response(content)

        reply_msg = Msg(
            name=self.name,
            content=content,
            role="assistant",
        )

        # 只有当设置需要添加记忆后才能添加记忆
        if self.save_to_memory:
            await self.memory.add(reply_msg)
            
        return reply_msg


    async def _extract_content(self, response) -> str:
        """从模型响应中提取文本，支持流式（async_generator）和非流式。"""

        def _extract_text(obj) -> str:
            """从单个响应对象或 chunk 中提取纯文本。"""
            if hasattr(obj, "content") and obj.content is not None:
                # content 通常是 list[TextBlock] 或 list[dict]
                if isinstance(obj.content, list):
                    texts = []
                    for block in obj.content:
                        if isinstance(block, dict):
                            t = block.get("text", "")
                        else:
                            t = getattr(block, "text", "")
                        if t:
                            texts.append(t)
                    return "".join(texts)
                return str(obj.content)
            if hasattr(obj, "text") and obj.text is not None:
                return str(obj.text)
            return str(obj)

        # 情况1：流式输出（stream=True）→ 异步生成器
        if inspect.isasyncgen(response):
            final_text = ""
            async for chunk in response:
                text = _extract_text(chunk)
                if not text:
                    continue
                # AgentScope 的流式 chunk 通常是"完整文本快照"（不是增量 delta），
                # 例如: '好' → '好开心' → '好开心呀'
                # 因此直接覆盖保留最后一个即可；若检测到是增量模式则拼接。
                if text.startswith(final_text) or final_text == "":
                    final_text = text
                else:
                    final_text += text
                    
            return final_text.strip('\n').strip()

        # 情况2：非流式输出（stream=False）
        return _extract_text(response).strip('\n').strip()



    # 目前这个observe只能是在同类SimpleAgent()所使用，fixme: 期望是能够将SimpleAgent的observe()工具使用到大脑智能体接受提前闲聊智能体的输出的时候用，但目前还不需要
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        if msg is not None:
            await self.memory.add(msg)


    async def print_llm_prompt(self, prompt) -> None:
        """调用 LLM 前打印完整的 Prompt Messages（Debug 用）。"""
        print(f"{'-'*60}")
        print(f"[LLM INPUT] Agent: {self.name}")
        for i, m in enumerate(prompt):
            role = m.get("role", "unknown") if isinstance(m, dict) else getattr(m, "role", "unknown")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            content_str = str(content)
            # display = content_str[:1000] + ("..." if len(content_str) > 1000 else "")
            display = content_str
            print(f"[{i}] {role}: {display}")

    async def print_llm_response(self, content: str) -> None:
        """拿到 LLM 响应后打印原始文本（Debug 用）。"""
        print(f"[LLM OUTPUT] Agent: {self.name}")
        content_str = str(content)
        # display = content_str[:1000] + ("..." if len(content_str) > 1000 else "")
        display = content_str
        print(f"{display}")
        print(f"{'-'*60}")

    # async def print(self, msg: Msg) -> None:
    #     text = msg.get_text_content()
    #     if text:
            # print(f"[{self.name}] {text}")
