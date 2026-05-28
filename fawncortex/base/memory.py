"""
记忆管理模块
============
封装三类记忆的创建与管理：
1. 工具挑选记忆（summarize_mem）—— Mem0LongTermMemory，供 BrainAgent 调用工具时存储高价值提炼记忆。
2. 短期记忆（ShortTermMemory）—— 滑动窗口式内存，供各 Agent 维护当前会话上下文。
3. 全量对话归档记忆（LongTermMemory）—— 独立的向量+文本双写系统，按轮次自动归档原始对话。

供主程序和各个 Agent 模块调用。
"""

import asyncio
import hashlib
import json
import sqlite3
import sys
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from agentscope.embedding import EmbeddingModelBase, OpenAITextEmbedding
from agentscope.message import Msg
from agentscope.memory import MemoryBase, Mem0LongTermMemory
from agentscope.model import OpenAIChatModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig


# =============================================================================
# 1. 工具挑选记忆（summarize_mem）
# =============================================================================

def create_summarize_memory(
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
    """创建并返回 Mem0LongTermMemory 实例（summarize_mem）。

    Args:
        agent_name: Agent 标识名，用于记忆隔离。
        user_name: 用户标识名，用于记忆隔离。
        vector_store_path: ChromaDB 向量存储目录路径。
        history_db_path: Mem0 历史数据库文件路径。
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


# =============================================================================
# 2. 短期记忆（ShortTermMemory）
# =============================================================================

SHORTTERMMEMORY_WINDOW_SIZE = 30


class ShortTermMemory(MemoryBase):
    """带容量上限的短期内存记忆实现。

    特性：
    - 固定容量上限（默认 30 条），超过时自动移除最旧的消息。
    - 保持压缩摘要支持（与 InMemoryMemory 兼容）。
    - 支持消息标记（mark）过滤。
    - 支持去重（默认不允许重复消息）。
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


# =============================================================================
# 3. 全量对话归档记忆（LongTermMemory）
# =============================================================================

class LongTermMemory:
    """全量对话长期记忆归档器（与 summarize_mem / mem0 完全隔离）。

    设计目标：
    - 将**每一轮**用户输入与 Agent 响应自动归档到独立的向量库 + 文本库。
    - 与 BrainAgent 的 `record_to_memory` 工具（summarize_mem）用途不同：
      summarize_mem 存储的是 BrainAgent "挑选提炼" 后的高价值记忆；
      LongTermMemory 存储的是**原始对话原文**的完整归档。

    核心特性：
    1. save() —— 异步非阻塞（fire-and-forget）。调用后立即返回，后台完成
       向量化 + 双写（ChromaDB + SQLite），不阻塞主对话流程。
    2. retrieve() —— 阻塞式（必须 await）。输入查询文本，返回 top-k 条最相关的
       历史对话，默认 top_k=10。

    存储结构：
    - 向量库：data/db/longterm_mem/  （ChromaDB 自动管理内部文件）
    - 文本库：data/db/longterm_mem/rawtext.db  （SQLite）

    使用示例：
        # 初始化（通常在 chat_cli.py / web_scheduler.py 的 main() 中）
        embedding_model = OpenAITextEmbedding(...)
        longterm_mem = LongTermMemory(
            agent_name=config.AGENT_NAME,
            user_name=config.USER_NAME,
            vector_store_path=config.LONGTERM_VECTOR_STORE_PATH,
            rawtext_db_path=config.LONGTERM_RAWTEXT_DB_PATH,
            embedding_model=embedding_model,
        )

        # 保存（非阻塞，主流程不等待）
        longterm_mem.save("user", "你好，今天天气怎么样？")
        longterm_mem.save("assistant", "今天北京晴天，气温 25°C 左右。")

        # 检索（阻塞，必须 await）
        results = await longterm_mem.retrieve("用户之前问过天气吗？", top_k=10)
        # results -> [{"id": "...", "role": "user", "content": "...",
        #              "similarity": 0.92, "metadata": {...}, "created_at": "..."}, ...]
    """

    def __init__(
        self,
        agent_name: str,
        user_name: str,
        vector_store_path: str,
        rawtext_db_path: str,
        embedding_model: EmbeddingModelBase,
    ) -> None:
        self.agent_id = agent_name
        self.user_id = user_name
        self._vector_store_path = vector_store_path
        self._rawtext_db_path = rawtext_db_path
        self._embedding_model = embedding_model

        # ChromaDB 持久化客户端（同步 API，后续在线程池中调用）
        self._chroma_client = chromadb.PersistentClient(path=vector_store_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name="longterm_mem",
            metadata={"agent_id": agent_name, "user_id": user_name},
        )

        # 初始化 SQLite 原始文本库
        self._init_rawtext_db()

        # 用于防止后台 asyncio.Task 被 GC 回收
        self._pending_tasks: set[asyncio.Task] = set()

        # embedding 内存缓存（LRU，上限 1000 条）
        # 供 ReflectionAgent 语义去重时直接读取已计算的向量，避免重复调用 API
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._embedding_cache_maxsize = 1000

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _init_rawtext_db(self) -> None:
        """创建 SQLite 原始文本表（如果不存在）。"""
        conn = sqlite3.connect(self._rawtext_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_id TEXT,
                    user_id TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_created_at
                ON conversation_turns(created_at)
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # 保存（非阻塞，fire-and-forget）
    # ------------------------------------------------------------------ #

    def save(self, role: str, content: str, metadata: dict | None = None) -> None:
        """将单条对话保存到长期记忆，后台异步完成向量化。

        调用方**无需 await**，立即返回，不阻塞主流程：
            longterm_mem.save("user", "你好")
            # 主流程继续执行...

        后台自动完成：
        1. 写入 SQLite 原始文本库（conversation_turns 表）。
        2. 调用 Embedding Model 获取向量。
        3. 写入 ChromaDB 向量库（longterm_mem collection）。

        若后台保存失败，异常会被捕获并打印到控制台，**不会抛到主流程**。

        Args:
            role: 发言角色，如 "user" / "assistant"。
            content: 对话内容。
            metadata: 可选的附加元数据字典（会 JSON 序列化后存入 SQLite）。
        """
        task = asyncio.create_task(
            self._do_save(role, content, metadata),
            name=f"ltm_save_{role}_{datetime.now().isoformat()}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        task.add_done_callback(self._on_save_done)

    async def _do_save(
        self,
        role: str,
        content: str,
        metadata: dict | None,
    ) -> None:
        """实际保存逻辑（在后台 Task 中执行）。"""
        # 缓存 key / doc_id 统一基于 strip 后的内容，
        # 避免 ReflectionAgent 查询时因 .strip() 导致 key 不匹配
        content_stripped = content.strip()
        doc_id = hashlib.md5(content_stripped.encode('utf-8')).hexdigest()
        created_at = datetime.now().isoformat()
        meta_dict = {
            "role": role,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "created_at": created_at,
            **(metadata or {}),
        }

        # 1. 写入 SQLite（同步 IO，在线程池中执行避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_insert_sqlite,
            doc_id,
            role,
            content_stripped,
            json.dumps(meta_dict, ensure_ascii=False),
            created_at,
        )

        # 2. 获取 embedding（AgentScope EmbeddingModel 为异步接口，直接 await）
        response = await self._embedding_model([content_stripped])
        embedding = response.embeddings[0]

        # 同步落入内存缓存（LRU），供 ReflectionAgent 语义去重直接读取
        self._put_embedding_cache(content_stripped, np.array(embedding, dtype=np.float32))

        # 3. 写入 ChromaDB（同步 API，在线程池中执行）
        await loop.run_in_executor(
            None,
            self._sync_insert_chroma,
            doc_id,
            content_stripped,
            embedding,
            meta_dict,
        )

    def _sync_insert_sqlite(
        self,
        doc_id: str,
        role: str,
        content: str,
        metadata_json: str,
        created_at: str,
    ) -> None:
        conn = sqlite3.connect(self._rawtext_db_path)
        try:
            # 使用 INSERT OR IGNORE 避免相同 content-hash ID 重复插入时报错
            # doc_id 为 content 的 MD5 hash，同一内容多次保存时保留首次记录
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_turns
                    (id, role, content, agent_id, user_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, role, content, self.agent_id, self.user_id,
                 metadata_json, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _sync_insert_chroma(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        self._collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def _put_embedding_cache(self, content: str, embedding: np.ndarray) -> None:
        """将 embedding 放入内存缓存（LRU 淘汰）。"""
        if content in self._embedding_cache:
            self._embedding_cache.move_to_end(content)
        self._embedding_cache[content] = embedding
        while len(self._embedding_cache) > self._embedding_cache_maxsize:
            self._embedding_cache.popitem(last=False)

    def get_cached_embedding(self, content: str) -> np.ndarray | None:
        """从内存缓存获取已计算的 embedding（零延迟，不查库）。

        供 ReflectionAgent._is_semantic_duplicate() 使用，避免重复调用 API。
        """
        # 统一用 strip 后的文本做 key，与 _do_save 保持一致
        key = content.strip()
        emb = self._embedding_cache.get(key)
        if emb is not None:
            self._embedding_cache.move_to_end(key)
        return emb

    def get_embedding_by_content(self, content: str) -> np.ndarray | None:
        """通过内容反查 ChromaDB 获取 embedding（第二级缓存）。

        利用 doc_id 为 content-hash 的特性，直接精确命中，无需重新计算。
        若记录尚未被后台 task 写入 ChromaDB，则返回 None。
        """
        # 统一用 strip 后的文本算 doc_id，与 _do_save 保持一致
        doc_id = hashlib.md5(content.strip().encode('utf-8')).hexdigest()
        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["embeddings"],
            )
            # 避免直接对 numpy array 做布尔判断（"truth value of array is ambiguous"）
            if result is None:
                return None
            embeddings = result.get("embeddings")
            if embeddings is None:
                return None
            # embeddings 可能是 list 或 numpy array，统一用 len() 判断
            if len(embeddings) > 0:
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            print(f"[LongTermMemory] ⚠️ ChromaDB 反查失败: {e}")
        return None

    def _on_save_done(self, task: asyncio.Task) -> None:
        """后台任务完成回调：吞掉异常，不抛到主流程。"""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[LongTermMemory] ⚠️ 后台保存失败: {e}")

    # ------------------------------------------------------------------ #
    # 检索（阻塞，必须 await）
    # ------------------------------------------------------------------ #

    async def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """根据对话上下文向量检索相关记忆。

        此方法**必须 await**，是阻塞式的：
            results = await longterm_mem.retrieve("用户提到了什么？")

        Args:
            query: 查询文本（通常是当前对话上下文或用户问题）。
            top_k: 返回最相关的记忆条数，默认 10 条。

        Returns:
            相关记忆列表，每项为字典：
            {
                "id": str,          # 唯一标识
                "role": str,        # "user" / "assistant"
                "content": str,     # 原始对话内容
                "similarity": float,# 相似度分数（0~1，越接近 1 越相关）
                "metadata": dict,   # 完整元数据
                "created_at": str,  # ISO 格式时间戳
            }
        """
        # 1. query 向量化
        response = await self._embedding_model([query])
        query_embedding = response.embeddings[0]

        # 2. ChromaDB 相似度搜索（同步 API，在线程池中执行）
        loop = asyncio.get_event_loop()
        chroma_results = await loop.run_in_executor(
            None,
            self._sync_query_chroma,
            query_embedding,
            top_k,
        )

        # 3. 组装结果
        results: list[dict] = []
        ids = chroma_results.get("ids", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            # Chroma 默认返回 L2 距离，转换为近似相似度（1 - 归一化距离）
            distance = distances[i] if i < len(distances) else 0.0
            similarity = max(0.0, 1.0 - distance)

            results.append({
                "id": doc_id,
                "role": meta.get("role", "unknown"),
                "content": documents[i] if i < len(documents) else "",
                "similarity": round(similarity, 4),
                "metadata": meta,
                "created_at": meta.get("created_at", ""),
            })

        return results

    def _sync_query_chroma(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> dict:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
