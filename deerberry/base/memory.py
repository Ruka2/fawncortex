"""
记忆管理模块
============
封装短期记忆（ShortTermMemory）与长期记忆（Mem0LongTermMemory）的创建，
供主程序和各个 Agent 模块调用。
"""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from agentscope.message import Msg
from agentscope.memory import MemoryBase, Mem0LongTermMemory
from agentscope.model import OpenAIChatModel
from agentscope.embedding import OpenAITextEmbedding


# 创建长期记忆实例
def create_long_term_memory(
    agent_name: str,
    user_name: str,
    vector_store_path: str,
    history_db_path: str,
    llm_model_name: str,
    llm_api_key: str,
    llm_base_url: str,
    embedding_model_name: str,
    embedding_api_key: str,
    embedding_base_url: str,
    llm_generate_kwargs: dict | None = None,
) -> Mem0LongTermMemory:
    """创建并返回 Mem0LongTermMemory 实例。

    Args:
        agent_name: Agent 标识名，用于记忆隔离。
        user_name: 用户标识名，用于记忆隔离。
        vector_store_path: ChromaDB 向量存储路径。
        history_db_path: Mem0 历史数据库路径。
        llm_model_name: LLM 模型名称。
        llm_api_key: LLM API 密钥。
        llm_base_url: LLM API 基础地址。
        embedding_model_name: 嵌入模型名称。
        embedding_api_key: 嵌入模型 API 密钥。
        embedding_base_url: 嵌入模型 API 基础地址。
        llm_generate_kwargs: 传给 LLM 的额外生成参数（如 enable_thinking）。

    Returns:
        配置好的 Mem0LongTermMemory 实例。
    """
    vector_store_config = VectorStoreConfig(
        provider="chroma",
        config={"path": vector_store_path},
    )
    mem0_config = MemoryConfig(
        vector_store=vector_store_config,
        history_db_path=history_db_path,
    )

    model_kwargs = {
        "model_name": llm_model_name,
        "api_key": llm_api_key,
        "stream": False,
        "client_kwargs": {"base_url": llm_base_url},
    }
    if llm_generate_kwargs:
        model_kwargs["generate_kwargs"] = llm_generate_kwargs

    return Mem0LongTermMemory(
        agent_name=agent_name,
        user_name=user_name,
        model=OpenAIChatModel(**model_kwargs),
        embedding_model=OpenAITextEmbedding(
            model_name=embedding_model_name,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        ),
        mem0_config=mem0_config,
    )


# 短期记忆窗口大小
SHORTTERMMEMORY_WINDOW_SIZE = 30

# 短期记忆
class ShortTermMemory(MemoryBase):
    """带容量上限的短期内存记忆实现。

    特性：
    - 固定容量上限（默认 30 条），超过时自动移除最旧的消息
    - 保持压缩摘要支持（与 InMemoryMemory 兼容）
    - 支持消息标记（mark）过滤
    - 支持去重（默认不允许重复消息）
    """

    def __init__(self, max_size: int = SHORTTERMMEMORY_WINDOW_SIZE) -> None:
        """初始化滑动窗口记忆。

        Args:
            max_size: 最大消息条数，超过时淘汰最旧的消息。
        """
        super().__init__()
        self.max_size = max_size
        # 与 InMemoryMemory 兼容的内部存储格式：list[tuple[Msg, list[str]]]
        self.content: list[tuple[Msg, list[str]]] = []
        self.register_state("content")

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
        prepend_summary: bool = True,
        **kwargs: Any,
    ) -> list[Msg]:
        """获取消息，支持按 mark 过滤和压缩摘要前置。

        Args:
            mark: 只返回带此标记的消息。
            exclude_mark: 排除带此标记的消息。
            prepend_summary: 是否在结果最前面加上压缩摘要。

        Returns:
            过滤后的消息列表。
        """
        if not (mark is None or isinstance(mark, str)):
            raise TypeError(
                f"The mark should be a string or None, but got {type(mark)}."
            )
        if not (exclude_mark is None or isinstance(exclude_mark, str)):
            raise TypeError(
                f"The exclude_mark should be a string or None, but got "
                f"{type(exclude_mark)}."
            )

        # 先按 mark 过滤
        filtered = [
            (msg, marks)
            for msg, marks in self.content
            if mark is None or mark in marks
        ]

        # 再按 exclude_mark 排除
        if exclude_mark is not None:
            filtered = [
                (msg, marks)
                for msg, marks in filtered
                if exclude_mark not in marks
            ]

        # 前置压缩摘要
        if prepend_summary and self._compressed_summary:
            return [
                Msg(
                    "user",
                    self._compressed_summary,
                    "user",
                ),
                *[msg for msg, _ in filtered],
            ]

        return [msg for msg, _ in filtered]

    async def add(
        self,
        memories: Msg | list[Msg] | None,
        marks: str | list[str] | None = None,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> None:
        """添加消息，超过容量上限时自动淘汰最旧的消息。

        Args:
            memories: 要添加的消息或消息列表。
            marks: 关联的标记。
            allow_duplicates: 是否允许重复消息（按 msg.id 判断）。
        """
        if memories is None:
            return

        if isinstance(memories, Msg):
            memories = [memories]

        if marks is None:
            marks = []
        elif isinstance(marks, str):
            marks = [marks]
        elif not isinstance(marks, list) or not all(
            isinstance(m, str) for m in marks
        ):
            raise TypeError(
                f"The mark should be a string, a list of strings, or None, "
                f"but got {type(marks)}."
            )

        if not allow_duplicates:
            existing_ids = {msg.id for msg, _ in self.content}
            memories = [msg for msg in memories if msg.id not in existing_ids]

        for msg in memories:
            self.content.append((deepcopy(msg), deepcopy(marks)))

        # ── 滑动窗口核心逻辑：超过上限时移除最旧的消息 ──
        while len(self.content) > self.max_size:
            removed_msg, _ = self.content.pop(0)
            print(
                f"[ShortTermMemory] 记忆达到上限 ({self.max_size})，"
                f"移除最旧消息: {removed_msg.name}"
            )

    async def delete(
        self,
        msg_ids: list[str],
        **kwargs: Any,
    ) -> int:
        """按消息 ID 删除消息。

        Returns:
            实际删除的消息数量。
        """
        initial_size = len(self.content)
        self.content = [
            (msg, marks)
            for msg, marks in self.content
            if msg.id not in msg_ids
        ]
        return initial_size - len(self.content)

    async def delete_by_mark(
        self,
        mark: str | list[str],
        **kwargs: Any,
    ) -> int:
        """按标记删除消息。

        Returns:
            实际删除的消息数量。
        """
        if isinstance(mark, str):
            mark = [mark]

        if isinstance(mark, list) and not all(
            isinstance(m, str) for m in mark
        ):
            raise TypeError(
                f"The mark should be a string or a list of strings, "
                f"but got {type(mark)} with elements of types "
                f"{[type(m) for m in mark]}."
            )

        initial_size = len(self.content)
        for m in mark:
            self.content = [
                (msg, marks)
                for msg, marks in self.content
                if m not in marks
            ]

        return initial_size - len(self.content)

    async def clear(self) -> None:
        """清空所有消息。"""
        self.content.clear()

    async def size(self) -> int:
        """获取当前消息数量。"""
        return len(self.content)

    async def update_messages_mark(
        self,
        new_mark: str | None,
        old_mark: str | None = None,
        msg_ids: list[str] | None = None,
    ) -> int:
        """更新消息的标记。

        Returns:
            更新的消息数量。
        """
        updated_count = 0

        for idx, (msg, marks) in enumerate(self.content):
            if msg_ids is not None and msg.id not in msg_ids:
                continue

            if old_mark is not None and old_mark not in marks:
                continue

            if new_mark is None:
                if old_mark in marks:
                    marks.remove(old_mark)
                    updated_count += 1
            else:
                if old_mark is not None and old_mark in marks:
                    marks.remove(old_mark)
                if new_mark not in marks:
                    marks.append(new_mark)
                    updated_count += 1

            self.content[idx] = (msg, marks)

        return updated_count

    def state_dict(self) -> dict:
        """获取序列化状态字典。"""
        return {
            **super().state_dict(),
            "content": [[msg.to_dict(), marks] for msg, marks in self.content],
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """从状态字典恢复。"""
        if strict and "content" not in state_dict:
            raise KeyError(
                "The state_dict does not contain 'content' "
                "keys required for ShortTermMemory."
            )

        self._compressed_summary = state_dict.get("_compressed_summary", "")

        self.content = []
        for item in state_dict.get("content", []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                msg_dict, marks = item
                msg = Msg.from_dict(msg_dict)
                self.content.append((msg, marks))
            elif isinstance(item, dict):
                # 兼容旧版本
                msg = Msg.from_dict(item)
                self.content.append((msg, []))
            else:
                raise ValueError(
                    "Invalid item format in state_dict for ShortTermMemory."
                )
