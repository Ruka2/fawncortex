"""
长期记忆工具集（纯工具函数）
============================
通过模块级注入模式接收 Mem0LongTermMemory 实例，
与 scripts.agent.memory 解耦，只保留可被 Toolkit 注册的工具函数。

使用方式：
    from scripts.tools.search_memory import (
        set_memory_manager,
        retrieve_from_memory,
        record_to_memory,
    )
    set_memory_manager(long_term_memory)
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
"""

import json
from typing import Any, Optional

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

# 模块级记忆管理器引用，由主程序通过 set_memory_manager() 注入
_memory_manager: Optional[Any] = None


def set_memory_manager(manager: Any) -> None:
    """注入已初始化的 Mem0LongTermMemory 实例。

    Args:
        manager: Mem0LongTermMemory 实例。
    """
    global _memory_manager
    _memory_manager = manager


async def retrieve_from_memory(keywords: list[str], limit: int = 5) -> ToolResponse:
    """根据关键词检索长期记忆（兼容新版 mem0 filters API）。

    Args:
        keywords: 检索关键词列表，每个词会独立执行一次语义搜索。
        limit: 每个关键词返回的最相关记忆条数上限。

    Returns:
        ToolResponse，content 中为 JSON 格式的记忆文本列表。
    """
    global _memory_manager
    if _memory_manager is None:
        return ToolResponse(
            content=[TextBlock(type="text", text="错误：记忆管理器尚未初始化")]
        )

    try:
        results = []
        for keyword in keywords:
            search_res = await _memory_manager.long_term_working_memory.search(
                query=keyword,
                filters={
                    "user_id": _memory_manager.user_id,
                    "agent_id": _memory_manager.agent_id,
                },
                top_k=limit,
            )
            for item in search_res.get("results", []):
                mem_text = item.get("memory", "")
                if mem_text and mem_text not in results:
                    results.append(mem_text)
        text = json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        text = f"记忆检索失败: {e}"

    return ToolResponse(content=[TextBlock(type="text", text=text)])


async def record_to_memory(thinking: str, content: list[str]) -> ToolResponse:
    """记录重要信息到长期记忆（兼容新版 mem0 API）。

    Args:
        thinking: 对记录内容的思考/推理说明。
        content: 要记录的具体内容列表。

    Returns:
        ToolResponse，content 中为操作结果文本。
    """
    global _memory_manager
    if _memory_manager is None:
        return ToolResponse(
            content=[TextBlock(type="text", text="错误：记忆管理器尚未初始化")]
        )

    try:
        messages = [
            {"role": "user", "content": thinking + "\n" + "\n".join(content)}
        ]
        res = await _memory_manager.long_term_working_memory.add(
            messages=messages,
            user_id=_memory_manager.user_id,
            agent_id=_memory_manager.agent_id,
            infer=False,  # 直接保存原始文本，禁止 mem0 调用 LLM 改写
        )
        count = len(res.get("results", []))
        text = f"成功记录 {count} 条记忆到长期记忆库"
    except Exception as e:
        text = f"记忆记录失败: {e}"

    return ToolResponse(content=[TextBlock(type="text", text=text)])
