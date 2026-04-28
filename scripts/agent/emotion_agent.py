"""
表情智能体（EmotionAgent）
==========================
封装 SimpleAgent，分析用户消息情绪并输出 VTS 动作名称。

特性：
- 支持大脑智能体建议的表情覆盖
- 默认基于用户输入分析情绪
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

from .agent import SimpleAgent


DEFAULT_EMOTION_PROMPT = (
    "你是虚拟形象的表情控制器。\n"
    "请根据你当前对话和历史对话记录，输出一个符合当前对话的表情动作。只输出一个枚举值，不要输出任何解释。\n"
    "可选名称(枚举值): smile, happy, laugh, sad, cry, angry, surprise, shy, "
    "sleepy, disgust, neutral, blink, wink, nod, tilt, talk"
)


class EmotionAgent(SimpleAgent):
    """前台表情智能体。"""

    def __init__(
        self,
        name: str = "emotion_controller",
        sys_prompt: Optional[str] = None,
        model: Optional[OpenAIChatModel] = None,
        memory: Optional[MemoryBase] = None,
        formatter: Optional[OpenAIChatFormatter] = None,
    ):
        if model is None:
            raise ValueError("EmotionAgent 需要传入 model 参数")
        super().__init__(
            name=name,
            sys_prompt=sys_prompt or DEFAULT_EMOTION_PROMPT,
            model=model,
            memory=memory or InMemoryMemory(),
            formatter=formatter or OpenAIChatFormatter(),
            save_to_memory=True,
        )

    @staticmethod
    def parse_action(text: str) -> str:
        """从输出文本中提取动作名称，失败则返回 'smile'。"""
        known = {
            "smile", "happy", "laugh", "sad", "cry", "angry",
            "surprise", "shy", "sleepy", "disgust", "neutral",
            "blink", "wink", "nod", "tilt", "talk",
        }
        for word in text.lower().replace(",", " ").replace(".", " ").split():
            word = word.strip()
            if word in known:
                return word
        return "smile"
