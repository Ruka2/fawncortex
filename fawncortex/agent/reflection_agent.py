"""
反思智能体（ReflectionAgent）
"""

from typing import List, Optional

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter

from fawncortex.base.simple_agent import SimpleAgent
from fawncortex.pipeline.event_controller import ThoughtEvent, InterventionEvent

import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config



### 中间汇报（Midway Intervention）配置 （FIXME: 这些参数设置都是暂时的，随时可以被替换走）
# 动态阈值基础值（秒）
MIDWAY_BASE_THRESHOLD = float(3.0)
# 动态阈值上限（秒）
MIDWAY_MAX_THRESHOLD = float(30.0)
# 阈值随前台回复长度增长的系数（每字符增加的秒数）
MIDWAY_THRESHOLD_FACTOR = float(0.1)


DEFAULT_REFLECTION_SYS_PROMPT = \
"""你一个智能体集群的反思判断器，智能体集群已经在大量的对话历史中回复了用户，你的职责是判断为对话历史反思判断响应回复句子是否能作为合适的回答。

### 任务信息
请你审视整个与用户的对话历史，重点在于你需要审视最近一句的回答是否承接整个话题、整个对话历史是否正常通顺。
你需要判断的信息：
 1. 用户的初始提问
 2. 你最近一轮的回答
 3. 历史对话记录

### 判断标记解释
你的判断将由特定标签进行选择，请根据你的判断枚举选择一个是否能合适作为本次回答的判断：["clarify", "ignore"]
以下是标签的判断解释：
 - clarify: 在历史对话记录中你判断出**你最近一轮的回答**的内容是对事实观点的陈述、新增观点、符合对话语境。
 - ignore: 结合**用户的初始提问**和**你最近一轮的回答**，判断最近一轮回答并不能承接对话历史、对话冗余重复、信息量少，是一个可忽略的对话。
 
### 判断标准
不应该以文本长度内容作为评判指标，必须以语义连贯性、对话话题保持、问题是否回答正确进行评判。
通常闲聊话题都可以直接快速响应，可判为可忽略对话。

### 输出格式
你只能从枚举列表 ["clarify", "ignore"] 选择输出一个标记。
"""


class ReflectionAgent(SimpleAgent):
    """反思智能体（Meta-Cognitive Controller / 审判官）。
    基于 SimpleAgent 实现：单步 LLM 调用，无 ReAct 循环，快速完成判断。
    """
    
    def __init__(self, model: Optional[OpenAIChatModel] = None) -> None:
        if model is None:
            raise ValueError("ReflectionAgent 需要传入 model 参数")

        super().__init__(
            name="reflection",
            sys_prompt=DEFAULT_REFLECTION_SYS_PROMPT,
            model=model,
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
            save_to_memory=False,
        )

        self.chat_history: list[Msg] = []
        self.thought_history: list[ThoughtEvent] = []

    # ── 动态阈值计算（供 midway_watcher 调用）──
    @staticmethod
    def compute_dynamic_threshold(chat_result: Optional[Msg]) -> float:
        """根据前台对话长度计算动态阈值。
        TODO: 后续需要优化这个等待大脑思考的时间阈值，调整为根据对话任务难度估算，而不是现在基于前台回复的字数的长度系数

        逻辑：
        - 前台回复越短 → 用户问题越简单 → 容忍时间越短
        - 前台回复越长 → 用户问题越复杂 → 容忍时间越长

        formula: threshold = BASE + chat_length * FACTOR, capped at MAX
        """
        
        # base = MIDWAY_BASE_THRESHOLD
        # max_threshold = MIDWAY_MAX_THRESHOLD
        # factor = MIDWAY_THRESHOLD_FACTOR

        # chat_text = chat_result.get_text_content() if chat_result else ""
        # token_count = len(chat_text)  # 简化为字符数，后续可替换为真实 token 数

        # threshold = base + token_count * factor
        # threshold = min(threshold, max_threshold)
        
        threshold = config.BRAIN_CUT_TIME_DURATION  # FIXME: 测试功能时会采用这样的时间戳

        return threshold



    # --- 每次返回判断反思
    async def judge_each_chat(self, user_input, agent_response, chat_history: List[Msg]):
        """
        对每一条最终回复都判断是否合理。
        合理：信息正确可解答用户问题、追问内容合适符合主题、正确像用户澄清问题、正确问候用户
        不合理：冗余重复回答、不合理的答案、重复复读答案

        Args:
            chat_history: 外部传入的对话历史
            ...

        Returns:
            标签标记，
        """
        
#         review_content = f"""请使用以下信息完成你本轮对话的判断，**历史对话记录**已存在提供的上下文中。
# **用户的初始提问** :```
# {user_input}
# ```

# **你最近一轮的回答** :```
# {agent_response}
# ```
# """
        review_content = f"""请你判断**你最近一轮的回答**是属于哪一类的回答？
你最近一轮的回答：```{agent_response}```
"""

        # 去除chat_history最后一个已回复的消息（这个消息就是agent_response）
        # chat_history = chat_history[:-2]  # FIXME: 将最近的大脑智能体的最后思考也暂时从智能体剔除掉，只审查智能体对话的内容
        # 拼接prompt（system + 历史 + 当前审查）──
        # 不经过 self.reply()，因为 save_to_memory=False 会导致当前 msg 被丢弃
        messages = [
            Msg("system", DEFAULT_REFLECTION_SYS_PROMPT, "system"),
            *chat_history,
            Msg(name="user", content=review_content, role="user"),
        ]
        prompt = await self.formatter.format(messages)
        await self.print_llm_prompt(prompt)

        # 直接调用模型
        response = await self.model(prompt)
        result_text = await self._extract_content(response)
        await self.print_llm_response(result_text)

        # ── 解析 token 输出 ──
        # 取第一个有效词作为 action（LLM 可能输出换行或额外空格）
        _text = result_text.strip().lower()
        action = _text.split()[0] if _text else ""


        if action == "summarize":
            return InterventionEvent(
                action="summarize",
                target="ChatAgent"
            )
        elif action == "clarify":
            return InterventionEvent(
                action="clarify",
                target="ChatAgent"
            )
        else:
            # 任何无法识别的 token
            return InterventionEvent(
                action="ignore",
                target="",
            )
