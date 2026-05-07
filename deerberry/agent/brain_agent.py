import json
import time
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

class BrainAgent:
    """ 大脑智能体封装 """

    DEFAULT_SYS_PROMPT = \
"""你是一个智能体集群的大脑核心系统，负责深度分析用户对话并为子智能体集群提供完成核心困难的推理任务。

### 任务简介
根据用户的输入和对话历史，分析用户的情绪、意图、隐含需求，并调用合适的工具来完成对话任务。

### 工具使用
你可以使用以下工具辅助完成任务：
- retrieve_from_memory: 检索与用户相关的历史记忆
- record_to_memory: 记录重要信息到长期记忆

### 输出要求
在完成所有任务后，将任务结果总结为一段自然语言的文本的答复，使对话生成更自然、更贴合用户需求的回复响应。

回复建议应包含以下内容：
- 用户当前的情绪和状态分析
- 用户的真实意图判断
- 相关的历史记忆提醒（如有）
- 需要特别注意的信息（如有）
- 所执行的任务结果
- 建议的回应策略和语气方向

要求：
1. 仅只输出自然语言文本，不要输出 JSON、不要输出代码块
2. 语气客观、分析性强，指出你所思考的内容
3. 第一人称为“我”，内容倾向于“我”智能体的思考过程
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

        # 闭包 wrapper：避免 bound-method 的 double-self 问题
        async def _hook_post_reasoning(react_self, kwargs, output):
            """在 _reasoning() 后记录本轮 reasoning 信息。"""
            try:
                self._current_iter += 1
                text = ""
                tool_uses = []
                if hasattr(output, "get_text_content"):
                    text = output.get_text_content() or ""
                if hasattr(output, "get_content_blocks"):
                    for block in output.get_content_blocks("tool_use"):
                        if isinstance(block, dict):
                            tool_uses.append({
                                "name": block.get("name", "unknown"),
                                "input": block.get("input", {}),
                            })

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
            """在 _acting() 后记录本轮 acting 信息。"""
            try:
                tool_call = kwargs.get("tool_call", {})
                tool_name = "unknown"
                tool_input = {}
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_input = tool_call.get("input", {})
                elif hasattr(tool_call, "name"):
                    tool_name = getattr(tool_call, "name", "unknown")
                    if hasattr(tool_call, "input"):
                        tool_input = getattr(tool_call, "input", {})

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
            except Exception as e:
                print(f"[BrainAgent] ⚠️ post_acting hook 异常（已吞）: {e}")
            return output  # 必须原样返回

        self.agent.register_instance_hook(
            "post_reasoning", "brain_track_reasoning", _hook_post_reasoning
        )
        self.agent.register_instance_hook(
            "post_acting", "brain_track_acting", _hook_post_acting
        )

    # ── 追踪器公共接口 ──
    def reset_react_tracker(self):
        """重置 ReAct 轮次追踪器。"""
        self._current_iter = 0
        self._iter_results = []
        self._react_start_ts = 0.0

    def get_react_snapshot(self) -> dict:
        """获取当前 ReAct 循环的快照。

        Returns:
            {
                "total_iters": int,
                "elapsed_sec": float,
                "iterations": list[dict],
            }
        """
        elapsed = time.time() - self._react_start_ts if self._react_start_ts else 0.0
        return {
            "total_iters": self._current_iter,
            "elapsed_sec": round(elapsed, 2),
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

        self.reset_react_tracker()
        self._react_start_ts = time.time()

        result = await self.agent.reply(user_msg)
        text = result.get_text_content()

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

        data = {
            "insight": text.strip(),
            "retrieved_memories": get_last_retrieved_memories(),
        }

        return data
