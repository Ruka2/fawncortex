"""
反思智能体（ReflectionAgent）
"""

import difflib
import json
from typing import List, Optional

import numpy as np
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter

from fawncortex.base.simple_agent import SimpleAgent
from fawncortex.base.memory import LongTermMemory
from fawncortex.pipeline.event_controller import ThoughtEvent, InterventionEvent

import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config



DEFAULT_REFLECTION_SYS_PROMPT = \
"""你一个智能体集群的反思判断器，你的职责是把控智能体的回复质量。

### 背景信息
在对话历史中，“工作记录”属于非对话内容，而是智能体的事实观点记录，事实观点记录不作为回复质量判断范围内。

### 反思任务
你需要结合对话历史（不包含工作记录），判断这个智能体将要回答的内容是否可以作为本轮回答给用户，判断标准：
 1. 回答的内容是否是重复闲聊？
 2. 若已经问候了对方一次是否还继续重复问候？
 3. 回复内容是否存在知识点重复赘述？

### 输出格式
请根据你的判断，从枚举列表选择 ["yes", "no"] 仅只输出一个标记：
 - no: 代表你认为将要回答的内容不是能直接回复给用户。
 - yes: 代表你认为将要回答的内容是可以正常回复给用户。
"""


# 请你先解释原因20字以内，再根据原因从枚举列表 ["yes", "no"] 选择输出一个标记：

class ReflectionAgent(SimpleAgent):
    """反思智能体（Meta-Cognitive Controller / 审判官）。
    基于 SimpleAgent 实现：单步 LLM 调用，无 ReAct 循环，快速完成判断。
    """
    
    def __init__(
        self,
        model: Optional[OpenAIChatModel] = None,
        longterm_memory: Optional[LongTermMemory] = None,
    ) -> None:
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

        # ========== 去重管道配置 ==========
        # 文本相似度阈值（0~1），>= 此值判定为重复
        self._dedup_similarity_threshold = 0.8

        # 语义去重依赖的 LongTermMemory 实例（可选，未传入时回退到硬去重）
        self._longterm_memory = longterm_memory
        # 语义去重阈值（向量相似度），高于此值判定为语义重复
        self._semantic_dedup_threshold = 0.825

        # ========== 自治反射视图（方案C）==========
        # 轮次输出索引：round_id -> [{type, text}]
        self._round_outputs: dict[int, list[dict]] = {}
        # 清洗后的对话上下文快照（从 self.memory 同步）
        self._reflection_context: list[Msg] = []

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
        # 动态阈值基础值（秒）
        # MIDWAY_BASE_THRESHOLD = float(3.0)
        # # 动态阈值上限（秒）
        # MIDWAY_MAX_THRESHOLD = float(30.0)
        # # 阈值随前台回复长度增长的系数（每字符增加的秒数）
        # MIDWAY_THRESHOLD_FACTOR = float(0.1)
        # base = MIDWAY_BASE_THRESHOLD
        # max_threshold = MIDWAY_MAX_THRESHOLD
        # factor = MIDWAY_THRESHOLD_FACTOR
        # chat_text = chat_result.get_text_content() if chat_result else ""
        # token_count = len(chat_text)  # 简化为字符数，后续可替换为真实 token 数
        # threshold = base + token_count * factor
        # threshold = min(threshold, max_threshold)
        threshold = config.BRAIN_CUT_TIME_DURATION
        return threshold



    # ========== 去重管道方法 ==========
    def record_output(self, round_id: int, output_type: str, text: str) -> None:
        """记录所有历史对话产生的输出，用于下一轮去重比较。
        区别于反思智能体的 observe() 方法，额外引入一个列表做存储是因为observe()虽然也有向量化操作，但是他保存的内容是无须不可索引的，
        较难从本地向量库中索引回来作为去重，并且对于对话存次（从用户输入->midway->summary作为一轮）而言，只去重本轮比较合适。
        因此此处的记录输出只是用作于反思智能体的索引记录（硬编码相似度去重、语义相似度去重会使用），而不是对短期记忆，
        也因此，大模型的质量判断仍然使用的是短期记忆，而此处使用的是额外的存储列表
        # TODO: 目前需要检查硬编码去重和语义去重是不是按轮次去重得

        Args:
            round_id: 当前轮次编号。
            output_type: 输出类型，如 "midway" / "summary" / "chat"。
            text: 输出文本内容。
        """
        if round_id not in self._round_outputs:
            self._round_outputs[round_id] = []
        self._round_outputs[round_id].append({"type": output_type, "text": text})

    def clear_round_outputs(self) -> None:
        """ 清空对话历史的 _round_outputs 输出 """
        print(f"[Reflection] 🗑️ 清空 _round_outputs (原 {len(self._round_outputs)} 轮)")
        self._round_outputs.clear()

    def _get_last_round_outputs(self, current_round: int) -> List[str]:
        """ 获取上一轮的所有 assistant 输出文本列表。
        Args:
            current_round: 当前轮次编号。
        Returns:
            上一轮输出文本列表（空列表表示无上一轮或上一轮无输出）。
        """
        last_round = current_round - 1
        if last_round not in self._round_outputs:
            return []
        return [item["text"] for item in self._round_outputs[last_round]]

    def _get_current_round_outputs(self, current_round: int) -> List[dict]:
        """获取当前轮次的所有 assistant 输出条目（含 type 和 text）。

        给 is_semantic_duplicate() 使用，用于检测本轮 midway/summary 之间的语义重复。
        """
        if current_round not in self._round_outputs:
            print(f"[Reflection] 📭 _get_current_round_outputs({current_round}): 空")
            return []
        outputs = self._round_outputs[current_round]
        print(f"[Reflection] 📬 _get_current_round_outputs({current_round}): {len(outputs)} 条")
        for i, o in enumerate(outputs):
            print(f"  [{i}] type={o.get('type')}, text='{o.get('text', '')[:50]}...'")
        return outputs

    # == 核心重复判断方法 ==
    ### 文本硬去重
    def is_hard_duplicate(self, agent_response: str, history_texts: List[str]) -> bool:
        """文本硬去重：基于 difflib 相似度判断是否与历史输出重复。

        Args:
            agent_response: 当前待判断的回复
            history_texts: 历史 assistant 回复列表

        Returns:
            True 如果判定为重复
        """
        if not agent_response or not history_texts:
            return False

        agent_response = agent_response.strip()

        for existing in history_texts:
            existing = existing.strip()
            # 完全相同的短文本，直接判定重复
            if agent_response == existing:
                print(f"[Reflection] 🚫 硬去重命中（完全重复）")
                return True

            # 长文本用 difflib 计算相似度
            similarity = difflib.SequenceMatcher(None, agent_response, existing).ratio()
            if similarity >= self._dedup_similarity_threshold:
                print(
                    f"[Reflection] 🚫 硬去重命中（相似度 {similarity:.2f}），"
                    f"当前: '{agent_response[:40]}...' | 历史: '{existing[:40]}...'"
                )
                return True

        return False

    ### 向量相似度去重
    async def is_semantic_duplicate(
        self,
        agent_response: str,
        round_id: int = 0,
    ) -> bool:
        """语义去重：基于本轮历史输出进行向量相似度比较。
        ### FIXME： 待优化，目前虽然是按轮次计算是否重复，但是向量内存缓存是没有办法清空的，即后续如果主动清空了上下文但是回复了相似的内容，就会被判断重复导致被误ignore

        与 is_hard_duplicate() 对齐比较范围：只查当前 round_id 内的
        _round_outputs，不再检索全量 LongTermMemory。embedding 优先从
        LongTermMemory 的内存缓存 / ChromaDB 反查获取，避免重复调用 API。

        向量会从这2处地方获取embedding：
        1. LongTermMemory 本地向量库（在对话主循环时已经异步添加，基于doc_id追踪）
        2. （兜底）如果主流程有时候异步失败了，会根据轮次对话内容现场计算嵌入兜底

        Args:
            agent_response: 当前待判断的回复
            round_id: 当前轮次编号

        Returns:
            True 是否与已有输出语义重复。
        """
        # 将前台回复 / 上轮回复也纳入比较范围
        last_round_outputs = self._get_current_round_outputs(round_id)
        # 处理上下文
        context_outputs = [
            {"text": msg.get_text_content()}
            for msg in self._reflection_context
            if getattr(msg, "role", "") == "assistant" and msg.get_text_content()
        ]
        all_outputs = last_round_outputs + context_outputs

        # print(
        #     f"[Reflection] 🔍 语义去重检查: round={round_id}, "
        #     f"candidates={len(all_outputs)} (outputs={len(last_round_outputs)}, context={len(context_outputs)})"
        # )

        # 获取待检测文本的embedding嵌入
        query_emb = await self._get_embedding_with_fallback(agent_response.strip())
        if query_emb is None:
            return False
        
        # 遍历所有候选，逐条比较语义相似度
        for item in all_outputs:
            cached_text = item.get("text", "")
            if not cached_text or not cached_text.strip():
                continue
            cached_emb = await self._get_embedding_with_fallback(cached_text.strip())
            if cached_emb is None:
                continue

            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity >= self._semantic_dedup_threshold:
                current = agent_response.strip()[:200]
                historical = cached_text.strip()[:200]
                print(
                    f"[Reflection] 🚫 语义去重命中（相似度 {similarity:.3f}），"
                    f"\n当前: '{current}'\n历史: '{historical}'"
                )
                return True

        return False

    async def _get_embedding_with_fallback(self, text: str) -> np.ndarray | None:
        """从本地chromadb获取embedding；或兜底现场计算 """
        # 1. 本地数据
        if self._longterm_memory is not None:
            emb = self._longterm_memory.get_cached_embedding(text)
            if emb is not None:
                # print(f"[Reflection] 💾 embedding 内存缓存命中")
                return emb

            # FIXME: 目前这一块未通过测试 2. ChromaDB 反查
            emb = self._longterm_memory.get_embedding_by_content(text)
            if emb is not None:
                # print(f"[Reflection] 💾 embedding ChromaDB 命中")
                return emb

        # 3. 现场计算（兜底）
        # print(f"[Reflection] 💾 embedding 未命中，现场计算...")
        return await self._compute_embedding(text)

    async def _compute_embedding(self, text: str) -> np.ndarray | None:
        """调用 Embedding Model 现场计算向量。

        计算成功后同步落入 LongTermMemory 内存缓存，供本轮后续 candidate 复用。
        """
        if self._longterm_memory is None:
            print(f"[Reflection] ⚠️ embedding 计算失败: longterm_memory 未配置")
            return None
        try:
            model = self._longterm_memory._embedding_model
            response = await model([text])
            emb = np.array(response.embeddings[0], dtype=np.float32)
            
            # 现场计算的结果也落入缓存，避免同一轮内重复计算
            self._longterm_memory._put_embedding_cache(text, emb)
            print(f"[Reflection] 💾 embedding 现场计算成功并落入缓存")
            return emb
        except Exception as e:
            print(f"[Reflection] ⚠️ embedding 计算失败: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度，范围 [-1, 1]。"""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)
    
    
    async def is_llm_judge(
        self,
        agent_response: str,
    ) -> InterventionEvent:
        """LLM 质量判断：调用大模型评估当前回复是否合理。

        使用 self._reflection_context（已从 self.memory 同步并清洗的上下文）。

        Args:
            agent_response: 当前待判断的智能体回复。

        Returns:
            InterventionEvent，action 为 clarify / ignore / summarize 等。
        """
        
        review_content = \
f"""智能体将要回答的内容：```
{agent_response}
```
"""

        # 拼接 prompt（system + 历史 + 当前审查）──
        messages = [
            Msg("system", DEFAULT_REFLECTION_SYS_PROMPT, "system"),
            *self._reflection_context,
            Msg(name="user", content=review_content, role="user"),
        ]
        prompt = await self.formatter.format(messages)
        await self.print_llm_prompt(prompt)

        # 直接调用模型
        response = await self.model(prompt)
        result_text = await self._extract_content(response)
        await self.print_llm_response(result_text)

        # ── 解析 token 输出 ──
        _text = result_text.strip()
        # action = _text.split()[0] if _text else ""
        if "no" in _text.lower():
            action = "ignore"
        elif "yes" in _text.lower():
            action = "clarify"
    
        return action
        



    # 【核心反思流程】
    async def judge_each_chat(
        self,
        user_input: str,
        agent_response: str,
        round_id: int = 0,
    ) -> InterventionEvent:
        """【管道流程】对每一条最终回复都判断是否合理。

        流程：
        1. 文本硬去重（查上一轮输出索引）
        2. 语义去重（LongTermMemory 向量检索）
        3. LLM 质量判断（基于自治反射上下文）

        Args:
            user_input: 用户的初始提问。
            agent_response: 当前待判断的智能体回复。
            round_id: 当前轮次编号（用于轮次去重索引）。

        Returns:
            InterventionEvent，action 为 clarify / ignore / summarize 等。
        """
        # 获取反思上下文（从自己智能体 memory 获取并清洗）
        raw = await self.memory.get_memory()
        self._reflection_context = [m for m in raw]
        
        # 默认反思是clasify, 即所有回答都先被认定合理的
        action = "clarify"

        # ============================================================
        # 管道步骤 1：文本硬去重（查上一轮 + 本轮已有输出 + 近期对话上下文）
        # ============================================================
        last_round_texts = self._get_last_round_outputs(round_id)
        current_round_items = self._get_current_round_outputs(round_id)
        current_round_texts = [item["text"] for item in current_round_items]
        
        # 把 reflection_context 中的 assistant 回复也纳入去重范围，
        # 捕获 chat 第一轮回复与 brain_summary 之间的重复
        context_texts = [
            msg.get_text_content() for msg in self._reflection_context
            if getattr(msg, "role", "") == "assistant"
        ]
        all_history_texts = last_round_texts + current_round_texts + context_texts
        
        # 去重
        seen = set()
        deduped = []
        for t in all_history_texts:
            t_stripped = t.strip()
            if t_stripped and t_stripped not in seen:
                seen.add(t_stripped)
                deduped.append(t_stripped)
                
        print(f"[Reflection] 🔍 硬去重检查: round={round_id}, histories={len(deduped)}, current='{agent_response[:40]}...'")
        if self.is_hard_duplicate(agent_response, deduped):
            return InterventionEvent(action="ignore", target="")

        # ============================================================
        # 管道步骤 2：语义去重（利用 LongTermMemory 向量检索）
        # ============================================================
        if await self.is_semantic_duplicate(agent_response, round_id=round_id):
            return InterventionEvent(action="ignore", target="")

        # ============================================================
        # 管道步骤 3：LLM 质量判断
        # ============================================================
        action = await self.is_llm_judge(agent_response)
        
        if action == "clarify":
            return InterventionEvent(action="clarify", target="ChatAgent")
        else:
            return InterventionEvent(action="ignore", target="")