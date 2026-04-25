# 系统组件库
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import asyncio
import time

# AgentScope 基础组件
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.pipeline import FanoutPipeline
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit

# 智能体自定义核心库
from scripts.agent.agent import SimpleAgent
from scripts.agent.memory import create_long_term_memory
from scripts.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
)

# 模型基础配置表
import config

# 外部（非智能体内核）组件
from components.tts import SiliconFlowCosyVoice
from components.vts_controller import VTSController
from scripts.tools.control_vts import set_vts_controller

# 智能体的工具
from scripts.tools.control_vts import express_emotion


# =============================================================================
# 常量：对话智能体的基础系统 Prompt（动态注入时需要保留此基础文本）
# =============================================================================
_BASE_CHAT_PROMPT = (
    "你是一个活泼可爱的人工智能助手，名字叫小花。回复内容请简洁短俑符合口头对话场景。"
    "每次对话回复内容长度约1~15个字，聊天内容请只输出纯文本，不要输出符号。"
)


# =============================================================================
# TTS 异步包装
# =============================================================================
async def speak_async(tts: SiliconFlowCosyVoice, text: str) -> None:
    """在线程池中执行 TTS 合成与播放，避免阻塞 asyncio 事件循环。"""
    if not text or not text.strip():
        return

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: tts.stream_synthesize(
                text=text.strip(),
                voice="FunAudioLLM/CosyVoice2-0.5B:diana",
                speed=1.0,
                gain=0.0,
                response_format="mp3",
                play=True,
            ),
        )
    except Exception as e:
        print(f"⚠️  TTS 合成/播放失败: {e}")


# =============================================================================
# 计时包装器
# =============================================================================
class TimedAgent:
    """对任意 Agent 进行包装，自动统计从调用到返回的耗时。"""
    def __init__(self, agent, name: str):
        self.agent = agent
        self.name = name

    async def __call__(self, msg=None):
        start = time.perf_counter()
        result = await self.agent(msg)
        elapsed = time.perf_counter() - start
        print(f"⏱️  {self.name} 处理耗时: {elapsed:.3f}s")
        return result


# =============================================================================
# 公共信息域：多智能体/多异步任务共享的认知资产
# =============================================================================
class SharedContext:
    """大脑Agent生产的洞察，供前台快速Agent消费。

    特性：
    - asyncio.Lock 保证并发安全
    - 版本号机制避免使用过期洞察
    - 置信度阈值筛掉低质量输出
    """

    def __init__(self):
        self._data = {
            "version": 0,
            "user_profile": "",
            "retrieved_memories": "",
            "strategy_hint": "",
            "user_emotion": "",
            "confidence": 0.0,
            "is_ready": False,
        }
        self._lock = asyncio.Lock()

    async def update(self, version: int, **kwargs) -> None:
        """大脑Agent完成推理后，将结果写入公共域。

        Args:
            version: 对应轮次，前台Agent可根据版本号判断是否为最新洞察。
        """
        async with self._lock:
            for k, v in kwargs.items():
                if k in self._data:
                    self._data[k] = v
            self._data["version"] = version
            self._data["is_ready"] = True

    def get_prompt_context(self) -> str:
        """生成可直接拼接到 system prompt 的文本片段。

        Returns:
            空字符串（未就绪或置信度不足），或格式化后的洞察上下文。
        """
        if not self._data["is_ready"] or self._data["confidence"] < 0.5:
            return ""

        parts = []
        for k, v in self._data.items():
            parts.append(f"{k}: {v}")
        if not parts:
            return ""
        
        return "\n".join(parts)

    def peek(self) -> dict:
        """只读查看当前数据（调试用）。"""
        return dict(self._data)


# =============================================================================
# 大脑Agent 后台异步执行包装
# =============================================================================
async def run_brain_async(
    brain_agent: ReActAgent,
    user_msg: Msg,
    shared_ctx: SharedContext,
    round_num: int,
) -> None:
    """在后台运行大脑Agent，基于用户输入检索记忆并生成洞察摘要。

    完成后自动写入 SharedContext，供下一轮前台Agent使用。
    任何异常都被捕获，不会影响主链路。
    """
    try:
        # 给大脑的输入：明确告知当前用户说了什么
        brain_input = Msg(
            name="user",
            content=(
                f"请分析以下用户输入，检索相关记忆，"
                f"并生成策略洞察摘要。\n\n"
                f"用户输入：{user_msg.content}"
            ),
            role="user",
        )

        result = await brain_agent.reply(brain_input)
        insight_text = result.get_text_content()

        # 简单解析：取前 500 字作为策略提示，可根据需求后续改为 JSON 解析
        await shared_ctx.update(
            version=round_num,
            strategy_hint=insight_text,
            confidence=0.8,
            is_ready=True,
        )
        print(f"🧠  大脑洞察已更新（第{round_num}轮）")

    except Exception as e:
        print(f"⚠️  大脑Agent后台推理失败: {e}")


# =============================================================================
# 主程序
# =============================================================================
async def main() -> None:
    # 会话id
    agent_name = "小花"
    user_name = "fafa"
    
    # 初始化在线 TTS
    tts = SiliconFlowCosyVoice()

    # 初始化长期记忆模块（配置已封装到 scripts.agent.memory）
    long_term_memory = create_long_term_memory(
        agent_name=agent_name,
        user_name=user_name,
    )
    # 将记忆实例注入到工具模块（纯工具函数通过模块级引用调用）
    set_memory_manager(long_term_memory)

    # -------------------------------------------------------------------------
    # 大脑Agent初始化
    # -------------------------------------------------------------------------
    brain_toolkit = Toolkit()
    brain_toolkit.register_tool_function(retrieve_from_memory)
    brain_toolkit.register_tool_function(record_to_memory)

    brain_agent = ReActAgent(
        name="brain_center",
        sys_prompt=(
            "你是大脑中枢，负责分析用户输入并检索长期记忆，为前台对话Agent生成策略洞察。\n"
            "你的工作任务：\n"
            "1. 你拥有可被检索的记忆库，记忆库中你可以找到与用户聊天时相关的历史对话以及记忆，"
            "因此当有需要时你可以使用记忆工具来找到与当前话题/任务/对话相关的内容\n"
            "2. 分析用户画像、用户意图、用户的情感状态，以此来在对话中输出用户潜在需求，引导对话更好的进行\n"
            "3. 记录你认为对用户有价值的信息到数据库中，以便后续记忆检索/回想事情所使用\n"
            "\n"
            "任务输出格式：\n"
            "请输出可由python代码json.loads()加载的列表数据，参考样例为：```\n"
            "[\n"
            '"用户画像": "...(限制1~30字)",\n'
            '"用户意图": "...(限制1~30字)",\n'
            '"用户情感状态": "...(限制1~10个字)",\n'
            '"用户潜在需求": "...(限制1~50个字)",\n'
            '"相关记忆": ["...可由python.loads()加载的列表, 列表内无限长度"],\n'
            "]\n"
        ),
        model=OpenAIChatModel(
            model_name=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            stream=config.STREAM,
            client_kwargs={"base_url": config.OPENAI_BASE_URL},
            generate_kwargs={"extra_body": {"enable_thinking": False}},
        ),
        memory=InMemoryMemory(),
        # 不使用 ReActAgent 内置的 long_term_memory 参数，避免框架自动注册有 bug 的旧版 API 工具，改用上方自定义工具
        formatter=OpenAIChatFormatter(),
        toolkit=brain_toolkit,
    )

    # 对话智能体 初始化
    chat_agent = SimpleAgent(
        name="小花",
        sys_prompt=_BASE_CHAT_PROMPT,
        model=OpenAIChatModel(
            model_name=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            stream=config.STREAM,
            client_kwargs={"base_url": config.OPENAI_BASE_URL},
            generate_kwargs={"extra_body": {"enable_thinking": False}},
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
    )

    # 表情Agent初始化
    emotion_agent = SimpleAgent(
        name="emotion_controller",
        sys_prompt=(
            "你是虚拟形象的表情控制器。\n"
            "分析用户消息的情绪，只输出一个最匹配的动作名称，不要解释。\n"
            "可选名称: smile, happy, laugh, sad, cry, angry, surprise, shy, "
            "sleepy, disgust, neutral, blink, wink, nod, tilt, talk"
        ),
        model=OpenAIChatModel(
            model_name=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            stream=config.STREAM,
            client_kwargs={"base_url": config.OPENAI_BASE_URL},
            generate_kwargs={"extra_body": {"enable_thinking": False}},
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
    )

    # 用 TimedAgent 计算市场和创建管道
    timed_chat = TimedAgent(chat_agent, "对话Agent")
    timed_emotion = TimedAgent(emotion_agent, "表情Agent")

    pipeline = FanoutPipeline(
        agents=[timed_chat, timed_emotion],
        enable_gather=True,
    )

    # 初始化公共信息域
    shared_ctx = SharedContext()

    # 初始问候轮
    round_num = 0
    msg = Msg(name="user", content="你好", role="user")

    # 主循环
    while True:
        round_num += 1

        # 信息域更新
        insight = shared_ctx.get_prompt_context()  # todo：后续需要持久化此文件
        if insight:
            # 关键步骤：每轮动态修改 SimpleAgent 的 sys_prompt  # todo: prompt优化
            timed_chat.agent.sys_prompt = (
                f"{_BASE_CHAT_PROMPT}\n\n"
                f"{insight}\n"
                f"以上信息仅供你参考，请继续自然地与用户对话。"
            )
        else:
            # 大脑Agent 尚未就绪时，恢复为基础 prompt
            timed_chat.agent.sys_prompt = _BASE_CHAT_PROMPT

        # 对话 + 表情Agent 并发执行
        total_start = time.perf_counter()
        replies = await pipeline(msg=msg)
        total_elapsed = time.perf_counter() - total_start
        print(f"⏱️  本次LLM消息端到端总耗时: {total_elapsed:.3f}s")

        chat_reply = replies[0]
        emotion_reply = replies[1]

        # ---- 对话智能体 → 输出文字 + TTS（后台不阻塞） ----
        reply_text = chat_reply.get_text_content()
        print(f"🤖 {agent_name}: {reply_text}")
        # tts_task = asyncio.create_task(speak_async(tts, reply_text))   # 开发版本目前先不播放语音，但任务还是创建着，后续开放
        
        # ── 对话智能体：保存原始对话到长期记忆，防止记忆丢失（fixme: 后续可能存在记忆太多导致检索召回低效问题，优化项） ──
        asyncio.create_task(
            long_term_memory.long_term_working_memory.add(  # 会向量数据库和raw对话历史数据库都写入
                messages=[
                    {"role": "user", "content": msg.content, "name": "user"},
                    {"role": "assistant", "content": reply_text, "name": agent_name},
                ],
                user_id=user_name,
                agent_id=agent_name,
                infer=False,
            )
        )

        # ---- 表情智能体 → VTS 动作触发 ----
        reply_emotion = emotion_reply.get_text_content()
        result = express_emotion(action=reply_emotion, duration=3.0, intensity=1.0)
        result_text = result.content[0]["text"]
        print(f"🎭 Emotion: {reply_emotion} | {result_text}")


        # 后台异步触发大脑Agent（不阻塞），大脑基于本轮用户输入做深度推理，完成后更新 SharedContext，供下一轮使用
        asyncio.create_task(
            run_brain_async(brain_agent, msg, shared_ctx, round_num)
        )

        # 等待用户下一轮输入
        user_input = (
            await asyncio.get_event_loop().run_in_executor(None, input)
        ).strip()


        # await tts_task  # 如需等待 TTS 完成再进入下一轮，可取消注释

        msg = Msg(name="user", content=user_input, role="user")


if __name__ == "__main__":
    asyncio.run(main())
