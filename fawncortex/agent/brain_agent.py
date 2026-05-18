import asyncio
import time
from datetime import datetime
from typing import Optional

from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OpenAIChatFormatter
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.message import Msg

from fawncortex.base.simple_agent import estimate_tokens

from fawncortex.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
    clear_last_retrieved_memories,  # 用于维护ShareContext的数据，非Brain tookit使用
    get_last_retrieved_memories,    # 用于维护chat_agent的数据，非Brain tookit使用
)

class BrainAgent:
    """ 大脑智能体封装 """

    DEFAULT_SYS_PROMPT = \
"""你是一个智能体集群的大脑核心，负责深度分析用户对话并为子智能体集群提供对话上的推理辅助。

### 任务简介
根据用户的输入和对话历史，分析用户的情绪、意图、隐含需求，并调用合适的工具来完成对话任务，因此你需要：
 1. 你的所有thinking推理过程都必须以第一人称“我”角度进行思考。
 2. 用户可以观察到你的思考过程，因此你应该在思考过程中将部分思考的观点提前告知出来，以此在思考过程中用户可以看到思考中的想法与观点。
   2.1 你的思考过程可以作为一个对话过程，因此你的整个思考过程必须是一个流畅通顺的思考文本
   2.2 你的思考过程结束后，不需要总结信息，你需要一步一步从中间思考过程提前回复用户你的答案
   2.3 你的推理过程需要注意：用户意图、相关历史记忆、执行任务的结果
   2.4 你需要输出的内容含有：每一步的思考过程、每一步思考后指导下一步聊天的指示内容
 3. 根据用户问题难度酌情使用工具，简单问题无须调用复杂工具，请分析对话与任务情况使用工具。

### 输出要求
输出自然文本对话，要求：
仅只输出自然语言文本，不要输出JSON及代码块，不要输出表情符号。
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

        # ── ReAct 轮次追踪 ──
        self._current_iter = 0
        self._iter_results = []
        self._react_start_ts = 0.0
        # midway 已同步到的 reasoning 轮次指针（用于增量同步）
        self._last_midway_sync_iter = 0

        # ── 子状态机（阶段1：状态同步）──
        # 子状态：idle | reasoning | acting
        self._sub_status: str = "idle"
        # 最新 reasoning 文本快照（供 midway_watcher 读取）
        self._latest_reasoning_text: str = ""
        # 最新调用的工具名
        self._latest_tool_name: str = ""
        # 本轮是否调用过任何工具
        self._has_used_tools: bool = False
        # think() 开始时间戳
        self._think_start_ts: float = 0.0

        # ── 流式截取缓冲区（阶段1：流式截取）──
        self._stream_buffer: str = ""
        self._is_streaming_reasoning: bool = False
        # midway 已同步到的流式内容长度指针（用于增量同步，方案B）
        self._last_stream_sync_len: int = 0

        # ── Hook 注册 ──
        # 闭包 wrapper：避免 bound-method 的 double-self 问题
        async def _hook_pre_reasoning(react_self, kwargs):
            """在 _reasoning() 前标记 stream 开始，更新子状态。"""
            try:
                self._is_streaming_reasoning = True
                self._stream_buffer = ""
                self._last_stream_sync_len = 0  # 新一轮 reasoning，重置增量指针
                self._sub_status = "reasoning"
            except Exception as e:
                print(f"[BrainAgent] ⚠️ pre_reasoning hook 异常（已吞）: {e}")
            return kwargs  # pre_hook 必须返回 kwargs（或 None）

        async def _hook_post_reasoning(react_self, kwargs, output):
            """在 _reasoning() 后记录本轮 reasoning 信息，更新子状态。"""
            try:
                self._current_iter += 1
                text = ""
                tool_uses = []
                if hasattr(output, "get_content_blocks"):
                    text = BrainAgent._extract_text_and_thinking(output)
                if hasattr(output, "get_content_blocks"):
                    for block in output.get_content_blocks("tool_use"):
                        if isinstance(block, dict):
                            tool_uses.append({
                                # "name": block.get("name", "unknown"),
                                "name": block.get("name", ""),  # FIXME: 临时将不调用工具的异常兜底改为空
                                "input": block.get("input", {}),
                            })

                # 更新最新 reasoning 文本
                self._latest_reasoning_text = text

                # 如果有 tool_use，子状态转为 acting；否则本轮结束，回到 idle
                if tool_uses:
                    self._sub_status = "acting"
                    self._has_used_tools = True
                else:
                    self._sub_status = "idle"

                self._iter_results.append({
                    "iter": self._current_iter,
                    "reasoning_text": text,
                    "tool_calls": tool_uses,
                    "timestamp": datetime.now().isoformat(),
                    "acting": None,
                })
                print(
                    f"[BrainAgent] 🔄 第 {self._current_iter} 轮 reasoning 完成"
                    f"（tool_calls={len(tool_uses)}）"
                )
            except Exception as e:
                # 追踪 hook 内部异常不得外抛，避免破坏 ReAct 流程
                print(f"[BrainAgent] ⚠️ post_reasoning hook 异常（已吞）: {e}")
            return output  # 必须原样返回，否则 ReActAgent 会丢失后续流程

        async def _hook_post_acting(react_self, kwargs, output):
            """在 _acting() 后记录本轮 acting 信息，更新子状态。"""
            try:
                tool_call = kwargs.get("tool_call", {})
                tool_name = ""
                tool_input = {}
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    tool_input = tool_call.get("input", {})
                elif hasattr(tool_call, "name"):
                    tool_name = getattr(tool_call, "name", "")
                    if hasattr(tool_call, "input"):
                        tool_input = getattr(tool_call, "input", {})

                self._latest_tool_name = tool_name

                if self._iter_results:
                    self._iter_results[-1]["acting"] = {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "timestamp": datetime.now().isoformat(),
                    }
                    print(
                        f"[BrainAgent] 🔧 第 {self._current_iter} 轮 acting 完成"
                        f"（tool={tool_name}）"
                    )

                # acting 完成后，子状态回到 idle（等待下一轮 reasoning）
                self._sub_status = "idle"
            except Exception as e:
                print(f"[BrainAgent] ⚠️ post_acting hook 异常（已吞）: {e}")
            return output  # 必须原样返回

        # 【关键修复】AgentScope 的 _ReActAgentMeta metaclass 只会在当前类的 attrs
        # 中查找 _reasoning/_acting 进行 hook 包装。但 ReActAgent 的这两个方法实际
        # 定义在父类 ReActAgentBase 中，导致 hook 永远不会触发。
        # 因此这里采用 monkey-patch 直接包装，绕过 metaclass 缺陷。
        agent = self.agent

        # 包装 _reasoning：注入 pre + post 逻辑
        _original_reasoning = agent._reasoning

        async def _wrapped_reasoning(*args, **kwargs):
            # pre-reasoning
            self._is_streaming_reasoning = True
            self._stream_buffer = ""
            self._last_stream_sync_len = 0
            self._sub_status = "reasoning"
            reasoning_start = time.perf_counter()
            # 调用原始方法
            output = await _original_reasoning(*args, **kwargs)
            reasoning_end = time.perf_counter()
            # post-reasoning（内联原 _hook_post_reasoning 逻辑）
            try:
                self._current_iter += 1
                text = ""
                tool_uses = []
                if hasattr(output, "get_content_blocks"):
                    text = BrainAgent._extract_text_and_thinking(output)
                if hasattr(output, "get_content_blocks"):
                    for block in output.get_content_blocks("tool_use"):
                        if isinstance(block, dict):
                            _name = block.get("name", "")
                            # 过滤掉 name 为空的无效工具调用块
                            if _name and _name.strip():
                                tool_uses.append({
                                    "name": _name,
                                    "input": block.get("input", {}),
                                })
                self._latest_reasoning_text = text
                if tool_uses:
                    self._sub_status = "acting"
                    self._has_used_tools = True
                else:
                    self._sub_status = "idle"
                self._iter_results.append({
                    "iter": self._current_iter,
                    "reasoning_text": text,
                    "tool_calls": tool_uses,
                    "timestamp": datetime.now().isoformat(),
                    "reasoning_start_ts": reasoning_start,
                    "reasoning_end_ts": reasoning_end,
                    "reasoning_sec": round(reasoning_end - reasoning_start, 3),
                    "acting": None,
                })
                print(
                    f"[BrainAgent] 🔄 第 {self._current_iter} 轮 reasoning 完成"
                    f"（tool_calls={len(tool_uses)}）"
                )
            except Exception as e:
                print(f"[BrainAgent] ⚠️ post_reasoning 包装异常（已吞）: {e}")
            return output

        agent._reasoning = _wrapped_reasoning

        # 包装 _acting：注入 post 逻辑
        _original_acting = agent._acting

        async def _wrapped_acting(*args, **kwargs):
            acting_start = time.perf_counter()
            output = await _original_acting(*args, **kwargs)
            acting_end = time.perf_counter()
            # post-acting（内联原 _hook_post_acting 逻辑）
            try:
                # AgentScope 以位置参数调用 _acting(tool_call)，所以从 args[0] 获取
                tool_call = args[0] if args else kwargs.get("tool_call", {})
                tool_name = ""
                tool_input = {}
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    tool_input = tool_call.get("input", {})
                elif hasattr(tool_call, "name"):
                    tool_name = getattr(tool_call, "name", "")
                    if hasattr(tool_call, "input"):
                        tool_input = getattr(tool_call, "input", {})
                self._latest_tool_name = tool_name
                # 只记录有效的工具调用（name 非空）
                if tool_name and tool_name.strip() and self._iter_results:
                    self._iter_results[-1]["acting"] = {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "timestamp": datetime.now().isoformat(),
                        "acting_start_ts": acting_start,
                        "acting_end_ts": acting_end,
                        "acting_sec": round(acting_end - acting_start, 3),
                    }
                    print(
                        f"[BrainAgent] 🔧 第 {self._current_iter} 轮 acting 完成"
                        f"（tool={tool_name}）"
                    )
                self._sub_status = "idle"
            except Exception as e:
                print(f"[BrainAgent] ⚠️ post_acting 包装异常（已吞）: {e}")
            return output

        agent._acting = _wrapped_acting

        # 保留 register_instance_hook 调用（ harmless，但已不生效）
        self.agent.register_instance_hook(
            "pre_reasoning", "brain_stream_start", _hook_pre_reasoning
        )
        self.agent.register_instance_hook(
            "post_reasoning", "brain_track_reasoning", _hook_post_reasoning
        )
        self.agent.register_instance_hook(
            "post_acting", "brain_track_acting", _hook_post_acting
        )

        # ── Patch print() 方法实现流式截取 ──
        # _reasoning() 的 stream 循环每次 token 都会调用 self.print()
        # print() 内部有 await asyncio.sleep(0)，会 yield 控制权给事件循环
        # 我们通过 patch print() 在 stream 过程中实时捕获 _stream_buffer
        original_print = self.agent.print

        async def patched_print(msg, last=True, speech=None):
            """Patch print 方法，在 reasoning stream 过程中实时捕获文本。"""
            try:
                if self._is_streaming_reasoning:
                    text = BrainAgent._extract_text_and_thinking(msg)
                    self._stream_buffer = text
            except Exception as e:
                print(f"[BrainAgent] ⚠️ patched_print 异常（已吞）: {e}")
            # 调用原始 print（保持原有行为：msg_queue + console 输出）
            await original_print(msg, last, speech)

        self.agent.print = patched_print

    # ── 状态机公共接口 ──
    def get_current_sub_status(self) -> str:
        """获取当前子状态（idle | reasoning | acting）。"""
        return self._sub_status

    def has_used_tools(self) -> bool:
        """本轮是否调用过任何工具。"""
        return self._has_used_tools

    def get_stream_buffer(self) -> str:
        """获取当前流式生成中的最新文本（供 midway_watcher 定时轮询）。"""
        return self._stream_buffer

    def get_stream_buffer_delta(self) -> str:
        """获取自上次 midway 同步以来新增的流式内容（增量，方案B）。

        配合 mark_stream_synced() 使用，确保每次 midway 只追加新增部分，
        避免完整快照导致的重复累积。
        """
        current = self._stream_buffer
        last = self._last_stream_sync_len
        if len(current) > last:
            return current[last:]
        return ""

    def mark_stream_synced(self) -> None:
        """标记当前流式内容已同步到 chat_agent，更新增量指针。"""
        self._last_stream_sync_len = len(self._stream_buffer)

    def get_latest_reasoning(self) -> str:
        """获取最新一轮 reasoning 的完整文本。"""
        return self._latest_reasoning_text

    def get_new_reasonings_since_last_sync(self) -> str:
        """获取上次 midway 同步后新增的 reasoning 内容（增量）。

        每次 midway 触发时，只返回 _last_midway_sync_iter 之后完成的 reasoning。
        返回空字符串表示没有新增内容。
        """
        parts = []
        for it in self._iter_results:
            if it["iter"] > self._last_midway_sync_iter:
                text = it.get("reasoning_text", "")
                if text and text.strip():
                    # parts.append(f"【第 {it['iter']} 轮思考】\n{text.strip()}")
                    parts.append(f"{text.strip()}")
        return "\n\n".join(parts)

    def mark_midway_synced(self) -> None:
        """标记当前所有 reasoning 已同步到 chat_agent。"""
        self._last_midway_sync_iter = self._current_iter

    def get_latest_tool_name(self) -> str:
        """获取最新调用的工具名。"""
        return self._latest_tool_name

    def get_total_reasoning_length(self) -> int:
        """获取当前已产生的 reasoning 内容总长度（字符数）。

        用于 midway_watcher 判断 brain 是否产生了足够实质性的内容，
        防止网络波动或空 reasoning 导致的无效 midway 触发。

        计算范围包括：
        1. 所有已完成 reasoning 轮次的文本
        2. 当前正在进行中的流式输出文本
        """
        total = 0
        # 1. 已完成的 reasoning 轮次
        for it in self._iter_results:
            text = it.get("reasoning_text", "")
            if text:
                total += len(text)
        # 2. 当前流式输出（如果有）
        total += len(self._stream_buffer)
        return total

    @staticmethod
    def _extract_text_and_thinking(msg) -> str:
        """从 Msg 中同时提取 text 块和 thinking 块，合并为完整字符串。

        AgentScope 的流式输出将模型外显文本放在 `type="text"` 块，
        将内部思考过程放在 `type="thinking"` 块（如 Qwen3、DeepSeek-R1
        等模型通过 OpenAI 兼容 API 输出的 reasoning_content）。
        `get_text_content()` 仅提取 text 块，会丢失完整的思考过程。
        """
        if msg is None:
            return ""
        if hasattr(msg, "get_content_blocks"):
            parts = []
            for block in msg.get_content_blocks():
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        parts.append(block["text"])
                    elif btype == "thinking" and block.get("thinking"):
                        parts.append(block["thinking"])
            if parts:
                return "\n\n".join(parts)
        if hasattr(msg, "get_text_content"):
            return msg.get_text_content() or ""
        return str(msg) if msg else ""

    def _build_status_suffix(self) -> str:
        """构建状态摘要字符串，可追加到中间汇报内容尾部。

        Returns:
            如 "正在调用 search_papers 工具中，请稍候..."
        """
        if self._sub_status == "reasoning":
            return f"\n不过后续我还正在思考中..."
        elif self._sub_status == "acting":
            return f"\n不过后续我还在调用工具，正在等待工具给我返回结果..."
        return ""

    # ── 追踪器公共接口 ──
    def reset_react_tracker(self):
        """重置 ReAct 轮次追踪器。"""
        self._current_iter = 0
        self._iter_results = []
        self._react_start_ts = 0.0
        # 重置 midway 同步指针
        self._last_midway_sync_iter = 0
        # 重置子状态机
        self._sub_status = "idle"
        self._latest_reasoning_text = ""
        self._latest_tool_name = ""
        self._has_used_tools = False
        self._think_start_ts = 0.0
        # 重置流式截取缓冲区
        self._stream_buffer = ""
        self._is_streaming_reasoning = False
        self._last_stream_sync_len = 0
        # 重置 Token 计数
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_llm_call_count = 0

    def reset_token_stats(self) -> None:
        """重置本轮 Token 和调用计数。"""
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_llm_call_count = 0

    def get_token_stats(self) -> dict:
        """获取最近一次 think() 的 Token 统计。

        LLM 调用次数 = ReAct 循环轮数（每次 reasoning 调用一次 LLM）。
        """
        return {
            "input_tokens": getattr(self, "_last_input_tokens", 0),
            "output_tokens": getattr(self, "_last_output_tokens", 0),
            "llm_call_count": self._current_iter,
        }

    def get_react_snapshot(self) -> dict:
        """获取当前 ReAct 循环的快照。

        Returns:
            {
                "total_iters": int,
                "elapsed_sec": float,
                "sub_status": str,
                "has_used_tools": bool,
                "latest_reasoning": str,
                "latest_tool_name": str,
                "stream_buffer": str,
                "iterations": list[dict],
            }
        """
        elapsed = time.time() - self._react_start_ts if self._react_start_ts else 0.0
        return {
            "total_iters": self._current_iter,
            "elapsed_sec": round(elapsed, 2),
            "sub_status": self._sub_status,
            "has_used_tools": self._has_used_tools,
            "latest_reasoning": self._latest_reasoning_text,
            "latest_tool_name": self._latest_tool_name,
            "stream_buffer": self._stream_buffer,
            "iterations": self._iter_results.copy(),
        }

    async def reply(self, user_msg) -> Msg:
        """执行深度思考，返回自然语言认知洞察文本。"""
        data = await self.think(user_msg)
        insight = data.get("insight", "")
        return Msg(name=self.agent.name, content=insight, role="assistant")

    async def think(self, user_msg) -> dict:
        """执行深度思考，返回洞察字典。

        Args:
            user_msg: 用户输入消息（AgentScope Msg 或原始文本）。

        Returns:
            {"insight": str, "retrieved_memories": list[str]}
        """
        from agentscope.message import Msg

        if isinstance(user_msg, str):
            user_msg = Msg(name="user", content=user_msg, role="user")

        # 每轮思考前：清空记忆检索缓存
        clear_last_retrieved_memories()

        # ── Token 与调用计数准备 ──
        self.reset_react_tracker()
        self._last_input_tokens = 0
        self._last_output_tokens = 0

        # 估算输入 Token（system prompt + memory + user_msg）
        # 使用 run_in_executor 避免 tiktoken 阻塞事件循环
        try:
            memory_msgs = await self.agent.memory.get_memory()
            memory_text = ""
            for m in memory_msgs:
                memory_text += BrainAgent._extract_text_and_thinking(m)
            input_text = (
                self.DEFAULT_SYS_PROMPT + "\n" + memory_text + "\n"
                + BrainAgent._extract_text_and_thinking(user_msg)
            )
            loop = asyncio.get_event_loop()
            self._last_input_tokens = await loop.run_in_executor(
                None, estimate_tokens, input_text
            )
        except Exception:
            pass

        self._react_start_ts = time.time()
        self._think_start_ts = time.time()

        result = await self.agent.reply(user_msg)

        text = BrainAgent._extract_text_and_thinking(result)

        # ── Debug: 打印 BrainAgent 最终输出 ──
        print(f"\n{'='*60}")
        print(f"[LLM OUTPUT] Agent: {self.agent.name} (ReActAgent)")
        print(f"  {text}")
        print(f"{'='*60}\n")

        # 打印 ReAct 轮次追踪摘要
        snapshot = self.get_react_snapshot()
        if snapshot["total_iters"] > 0:
            print(f"[BrainAgent] 📊 ReAct 循环摘要: {snapshot['total_iters']} 轮, "
                  f"耗时 {snapshot['elapsed_sec']}s")
            for it in snapshot["iterations"]:
                acting = it.get("acting")
                if acting:
                    print(f"  - 轮次 {it['iter']}: reasoning → acting({acting['tool_name']})")
                else:
                    print(f"  - 轮次 {it['iter']}: reasoning（无工具调用）")

        # 估算输出 Token（所有 reasoning + 最终 insight）
        try:
            output_text = text
            for it in self._iter_results:
                output_text += it.get("reasoning_text", "")
            loop = asyncio.get_event_loop()
            self._last_output_tokens = await loop.run_in_executor(
                None, estimate_tokens, output_text
            )
        except Exception:
            pass

        data = {
            "insight": text.strip(),
            "retrieved_memories": get_last_retrieved_memories(),    # FIXME: 主流程并没有使用相关记忆
        }

        return data
