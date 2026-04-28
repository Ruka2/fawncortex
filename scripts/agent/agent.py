"""
含BaseAgent + TTS + VTS 方式调用智能体
main2.py为使用ReActAgent的版本
main3.py为使用SimpleAgent的版本（本文件），对比两者可见ReAct循环的增删对代码结构的影响。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import inspect
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter

import config
from components.vts_controller import VTSController
from scripts.tools.control_vts import express_emotion, set_vts_controller


class SimpleAgent(AgentBase):
    """最简自定义智能体：单步调用大模型，无 ReAct 循环。
    本身不处理工具调用，如需触发外部动作（如 VTS），在拿到回复后由外部代码处理。
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: OpenAIChatModel,
        memory: MemoryBase,
        formatter: OpenAIChatFormatter,
    ) -> None:
        super().__init__()
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory = memory
        self.formatter = formatter

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        """接收消息 → 调用 LLM → 返回回复。"""
        if msg is not None:
            await self.memory.add(msg)

        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )

        # ── Debug: 打印发送给 LLM 的完整 Messages ──
        print(f"\n{'='*60}")
        print(f"[LLM INPUT] Agent: {self.name}")
        print(f"{'='*60}")
        for i, m in enumerate(prompt):
            role = m.get("role", "unknown") if isinstance(m, dict) else getattr(m, "role", "unknown")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            # 截断过长内容，避免刷屏
            content_str = str(content)
            display = content_str[:400] + ("..." if len(content_str) > 400 else "")
            print(f"  [{i}] {role}: {display}")
        print(f"{'='*60}")

        # 调用模型：兼容 stream=True（异步生成器）和 stream=False（直接对象）
        response = await self.model(prompt)
        content = await self._extract_content(response)

        # ── Debug: 打印 LLM 返回的原始响应 ──
        print(f"\n{'='*60}")
        print(f"[LLM OUTPUT] Agent: {self.name}")
        content_str = str(content)
        display = content_str[:500] + ("..." if len(content_str) > 500 else "")
        print(f"  {display}")
        print(f"{'='*60}\n")

        reply_msg = Msg(
            name=self.name,
            content=content,
            role="assistant",
        )

        await self.memory.add(reply_msg)
        await self.print(reply_msg)
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
            return final_text

        # 情况2：非流式输出（stream=False）
        return _extract_text(response)

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        if msg is not None:
            await self.memory.add(msg)

    async def print(self, msg: Msg) -> None:
        text = msg.get_text_content()
        # if text:
        #     print(f"[{self.name}] {text}")

    async def handle_interrupt(self, *args: Any, **kwargs: Any) -> Msg:
        return Msg(
            name=self.name,
            content="我被打断了，有什么可以帮你的吗？",
            role="assistant",
        )


# =============================================================================
# 已知动作列表（用于 emotion_agent 解析回退）
# =============================================================================

_KNOWN_ACTIONS = {
    "smile", "happy", "laugh", "sad", "cry", "angry",
    "surprise", "shy", "sleepy", "disgust", "neutral",
    "blink", "close_eyes", "wink",
    "lean_left", "lean_right", "nod", "tilt",
    "talk",
}


def _parse_action(text: str) -> str:
    """从 emotion_agent 的输出中提取动作名称，失败则返回 'smile'。"""
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        word = word.strip()
        if word in _KNOWN_ACTIONS:
            return word
    return "smile"


# =============================================================================
# 测试入口：两个 Agent 均使用 SimpleAgent 类
# =============================================================================


async def main() -> None:
    from agentscope.model import OpenAIChatModel
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory

    # ---- 1. 连接 VTS（express_emotion 需要） ----
    try:
        vts = VTSController(host=config.VTS_HOST, port=config.VTS_PORT)
        await vts.connect_and_auth()
        set_vts_controller(vts)
        print("✅ VTS 已连接")
    except Exception as e:
        print(f"⚠️  VTS 连接失败（表情工具将不可用）: {e}")

    # ---- 2. 共用模型配置 ----
    model = OpenAIChatModel(
        model_name=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        stream=config.STREAM,
        client_kwargs={"base_url": config.OPENAI_BASE_URL},
        generate_kwargs={"extra_body": {"enable_thinking": False}},
    )

    # ---- 3. 对话智能体（SimpleAgent） ----
    chat_agent = SimpleAgent(
        name="simple",
        sys_prompt=(
            "你是一个活泼可爱的极简助手，回复内容请简洁简短，"
            "每次对话回复长度约1~15个字，只输出纯文本，不要输出符号。"
        ),
        model=model,
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
    )

    # ---- 4. 表情智能体（同样使用 SimpleAgent 类） ----
    # 让 LLM 只输出一个动作名称，由外部代码手动调用 express_emotion
    emotion_agent = SimpleAgent(
        name="emotion",
        sys_prompt=(
            "你是虚拟形象的表情控制器。\n"
            "分析用户消息的情绪，只输出一个最匹配的动作名称，不要解释。\n"
            "可选名称: smile, happy, laugh, sad, cry, angry, surprise, shy, "
            "sleepy, disgust, neutral, blink, wink, nod, tilt, talk"
        ),
        model=model,
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
    )

    # ---- 5. 发送测试消息 ----
    test_msg = Msg(name="user", content="你好，我今天超级开心！", role="user")
    print(f"\n👤 User: {test_msg.content}\n")

    # 并发调用两个 SimpleAgent
    chat_task = chat_agent(test_msg)
    emotion_task = emotion_agent(test_msg)
    chat_reply, emotion_reply = await asyncio.gather(chat_task, emotion_task)

    # ---- 6. 处理对话回复 ----
    print(f"\n🤖 ChatAgent: {chat_reply.get_text_content()}")

    # ---- 7. 处理表情回复：手动解析并调用 VTS 工具 ----
    raw_emotion = emotion_reply.get_text_content()
    action = _parse_action(raw_emotion)
    print(f"🤖 EmotionAgent 原始输出: {raw_emotion}")
    print(f"🤖 解析为动作: {action}")

    result = express_emotion(action=action, duration=3.0, intensity=1.0)
    # ToolResponse 没有 get_text_content()，直接取 content 列表中的 TextBlock
    result_text = result.content[0]["text"]
    print(f"🤖 VTS 调用结果: {result_text}")
    print("\n✅ 测试完成，两个 SimpleAgent 均正常调用。")


if __name__ == "__main__":
    asyncio.run(main())
