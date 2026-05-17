"""
表情智能体（EmotionAgent）
"""

from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import MemoryBase
from agentscope.formatter import OpenAIChatFormatter

from fawncortex.base.simple_agent import SimpleAgent
from fawncortex.base.memory import ShortTermMemory


import re
import json
from fawncortex.components.body.emotion_animate import AVAILABLE_ACTIONS

available_action_content = json.dumps(AVAILABLE_ACTIONS, indent=2, ensure_ascii=False)


DEFAULT_EMOTION_PROMPT = \
f"""你是控制智能体集群中的表情控制器，需要为智能体集群推理出合适对话场景的表情或动作、以及语气口吻。

# 任务简介
智能体集群正在对话，你需要为智能体集群判断选择出一个合适当前对话的表情或动作、和推理合适的语气口吻或方言语种，以此对用户进行响应（表达情感、或展现自我）。

## 任务一：选择表情或动作指令
从以下表情动作集合中进行选择：```
{available_action_content}
```
其中，第一层数据结构为表情动作种类，动作种类用于辅助进行选择，第二层列表内元素是你目标要选择的表情或动作指令。

## 任务二：推理语气口吻
根据对话历史上下文和用户的提问，输出一个简短的语气口吻控制指令，口吻控制指令文本必须在10个字以内。
例子：```
使用开心的语气说
使用怀疑的语气说
使用广东话语种说
使用用标准英语说
...
```

### 输出格式
仅只输出JSON结构的数据：```
{{
"emotion": "...从表情集合选择",
"tone": "...填入语气指令"
}}
```
"""


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
            memory=memory or ShortTermMemory(),
            formatter=formatter or OpenAIChatFormatter(),
            save_to_memory=True,
        )

    @staticmethod
    def parse_action(text: str) -> tuple[str, str]:
        """从模型输出文本中解析 emotion 和 tone。

        Args:
            text: SimpleAgent().reply() 返回的文本内容（期望包含 JSON）。

        Returns:
            (emotion, tone): emotion 若不在合法动作集合中则返回 "neural"；
                             tone 若超过 20 个字符则返回空字符串。
        """

        # 构建合法动作集合
        valid_actions = set()
        for action_list in AVAILABLE_ACTIONS.values():
            valid_actions.update(action_list)

        # 默认值
        emotion = "neural"
        tone = ""

        # 尝试提取并解析 JSON
        try:
            # 优先匹配 markdown 代码块 ```json ... ```
            match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
            )
            if match:
                json_str = match.group(1)
            else:
                # 匹配裸 JSON { ... }
                match = re.search(r"\{.*\}", text, re.DOTALL)
                json_str = match.group(0) if match else text

            data = json.loads(json_str)
            if isinstance(data, dict):
                # 解析 emotion
                raw_emotion = data.get("emotion")
                if isinstance(raw_emotion, str):
                    raw_emotion = raw_emotion.strip()
                    if raw_emotion in valid_actions:
                        emotion = raw_emotion

                # 解析 tone
                raw_tone = data.get("tone")
                if isinstance(raw_tone, str):
                    raw_tone = raw_tone.strip()
                    if len(raw_tone) <= 20:
                        tone = raw_tone
        except Exception:
            pass

        return emotion, tone
