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

# 智能体自定义核心库
from scripts.agent.agent import SimpleAgent
from scripts.agent.memory import SQLiteMemoryManager


# 模型基础配置表
import config

# 外部（非智能体内核）组件
from components.tts import SiliconFlowCosyVoice
from components.vts_controller import VTSController
from scripts.tools.control_vts import set_vts_controller   # fixme: 暂时不知道这个有什么用，先保留待VTS正式载入

# 智能体的工具
from scripts.tools.control_vts import express_emotion



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
# 主程序
# =============================================================================
async def main() -> None:
    # 初始化在线 TTS
    tts = SiliconFlowCosyVoice()
    
    # 加载智能体记忆库
    manager = SQLiteMemoryManager(db_path="./data/agent_memory.db")
    chat_memory = await manager.create_memory(user_id="user_1", session_id="chat_001")
    

    # ---- 对话智能体（SimpleAgent） ----
    chat_agent = ReActAgent(
        name="fafa",
        sys_prompt=(
            "你是一个活泼可爱的人工智能助手，回复内容请简洁简短符合口头对话场景。"
            "每次对话回复内容长度约1~15个字，聊天内容请只输出纯文本，不要输出符号。"
        ),
        model=OpenAIChatModel(
            model_name=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            stream=config.STREAM,
            client_kwargs={"base_url": config.OPENAI_BASE_URL},
            generate_kwargs={"extra_body": {"enable_thinking": False}},
        ),
        memory=chat_memory,
        formatter=OpenAIChatFormatter(),
    )

    # ---- 动作/表情智能体（同样使用 SimpleAgent） ----
    # 让 LLM 只输出一个动作名称，由外部代码调用 express_emotion
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
        memory=InMemoryMemory(),  # fixme: 表情智能体目前是否添加记忆还需要讨论，所以先以内存记忆占位，后续优化
        formatter=OpenAIChatFormatter(),
    )

    # 关闭 emotion_agent 的终端打印，避免干扰对话界面
    # async def _silent_print(msg: Msg) -> None:
    #     pass
    # emotion_agent.print = _silent_print

    # 用 TimedAgent 包装，统计各自耗时
    timed_chat = TimedAgent(chat_agent, "对话智能体(fafa)")
    timed_emotion = TimedAgent(emotion_agent, "动作智能体(emotion_controller)")

    # ---- 创建扇出管道：同一用户消息并发分发给两个智能体 ----
    pipeline = FanoutPipeline(
        agents=[timed_chat, timed_emotion],
        enable_gather=True,
    )

    # 初始问候轮：用于测试启动
    msg = Msg(name="user", content="你好", role="user")

    try:
        while True:
            # ---- 并发执行：两个 SimpleAgent 同时处理同一条用户消息 ----
            total_start = time.perf_counter()
            replies = await pipeline(msg=msg)
            total_elapsed = time.perf_counter() - total_start
            print(f"⏱️  本次LLM消息端到端总耗时: {total_elapsed:.3f}s")

            chat_reply = replies[0]
            emotion_reply = replies[1]

            # ---- 对话智能体 → 输出文字 + TTS（后台不阻塞） ----
            reply_text = chat_reply.get_text_content()
            print(f"🤖 Agent: {reply_text}")
            tts_task = asyncio.create_task(speak_async(tts, reply_text))

            # ---- 动作智能体 → 解析动作名称并手动触发 VTS ----
            reply_emotion = emotion_reply.get_text_content()
            result = express_emotion(action=reply_emotion, duration=3.0, intensity=1.0)
            result_text = result.content[0]["text"]
            print(f"🤖 Emotion: {reply_emotion} | {result_text}")

            # ---- 异步等待用户输入 ----
            user_input = (
                await asyncio.get_event_loop().run_in_executor(None, input)
            ).strip()

            # await tts_task  如需等待 TTS 完成再进入下一轮，可取消此行注释

            # 用户的输入装载为 Msg 类，然后while遍历循环让智能体响应
            msg = Msg(name="user", content=user_input, role="user")
            
            
    finally:  # 关闭数据库确保资源正确释放
      await chat_memory.close()
      await manager.close()

if __name__ == "__main__":
    asyncio.run(main())
