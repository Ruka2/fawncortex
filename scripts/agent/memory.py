

"""
AgentScope + mem0 记忆管理器
=============================
基于 mem0 框架的记忆模块，使用 ChromaDB 作为本地向量存储，
SQLite 作为历史记录数据库。

与 AgentScope MemoryBase 的区别：
- 支持语义搜索召回（search）
- 支持向量相似度检索
- 返回 dict 列表而非 Msg 列表，由调用方自行决定如何使用

使用方法：
    from scripts.agent.memory_mem0 import Mem0MemoryManager

    manager = Mem0MemoryManager(user_id="user_1", session_id="chat_001")
    manager.add("你好", name="fafa", role="user")
    results = manager.search("问候", top_k=5)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from datetime import datetime
from typing import Any, Optional


from mem0 import Memory




class Mem0MemoryManager:
    """基于 mem0 的记忆管理器。

    内部使用 ChromaDB 做向量存储，SQLite 做历史记录，
    通过 SiliconFlow 的 OpenAI 兼容接口调用 Embedding 和 LLM。

    Attributes:
        user_id: 用户 ID，用于多用户隔离。
        session_id: 会话 ID，记录在 metadata 中。
        memory: mem0 的 Memory 实例。
    """

    def __init__(
        self,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id

        # mem0 配置：ChromaDB + OpenAI 兼容 API
        mem0_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": config.MEM0_VECTOR_STORE_PATH,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.MODEL_NAME,
                    "api_key": config.OPENAI_API_KEY,
                    "openai_base_url": config.OPENAI_BASE_URL,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": config.EMBEDDING_MODEL_NAME,
                    "api_key": config.OPENAI_API_KEY,
                    "openai_base_url": config.OPENAI_BASE_URL,
                },
            },
            "history_db_path": config.MEM0_HISTORY_DB_PATH,
        }

        self.memory = Memory.from_config(mem0_config)



    # =====================================================================
    # 写入
    # =====================================================================

    def add(
        self,
        content: str,
        name: str,
        role: str,
        msg_type: str = "",
        timestamp: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """添加一条记忆。

        Args:
            content: 记忆文本内容（原始对话文本）。
            name: Agent 名称或用户名称。
            role: "assistant" 或 "user"。
            msg_type: 记忆来源类型。可选：
                ""（默认）, "from_user", "from_agent_summary", "from_daily_summary"。
            timestamp: 时间戳，格式 YYYY-MM-DD HH:MM:SS。
                       为 None 时自动使用当前时间。
            metadata: 额外的自定义元数据（会合并到内部 metadata 中）。
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        meta: dict[str, Any] = {
            "name": name,
            "role": role,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "type": msg_type,
        }
        if metadata:
            meta.update(metadata)

        # infer=False：禁止 mem0 调用 LLM 对内容进行 fact extraction，
        # 直接存储原始文本，确保 content 不被改写。
        self.memory.add(
            content,
            user_id=self.user_id,
            metadata=meta,
            infer=False,
        )

    # =====================================================================
    # 读取 / 召回
    # =====================================================================

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义搜索记忆，返回与 query 最相关的 top_k 条记忆。

        Args:
            query: 查询文本。
            top_k: 返回结果数量上限。

        Returns:
            dict 列表，每条包含以下字段：
                - id: 记忆 UUID
                - name: 名称
                - role: "assistant" | "user"
                - timestamp: 时间戳
                - content: 原始记忆文本
                - session_id: 会话 ID
                - type: 记忆来源类型
                - score: 相似度分数（0~1，越高越相关）
        """
        raw = self.memory.search(
            query,
            filters={"user_id": self.user_id},
            top_k=top_k,
        )
        return self._format_results(raw)

    def get_all(self, top_k: int = 100) -> list[dict[str, Any]]:
        """获取该用户的全部记忆。

        Args:
            top_k: 返回数量上限。

        Returns:
            dict 列表，字段同 search()。
        """
        raw = self.memory.get_all(
            filters={"user_id": self.user_id},
            top_k=top_k,
        )
        return self._format_results(raw)

    # =====================================================================
    # 删除
    # =====================================================================

    def delete(self, memory_id: str) -> None:
        """删除单条记忆。

        Args:
            memory_id: 记忆的 UUID（即返回 dict 中的 "id" 字段）。
        """
        self.memory.delete(memory_id)

    def clear(self) -> int:
        """清空该用户的全部记忆。

        Returns:
            被删除的记忆数量。
        """
        all_memories = self.memory.get_all(filters={"user_id": self.user_id})
        count = 0
        for item in all_memories.get("results", []):
            self.memory.delete(item["id"])
            count += 1
        return count

    # =====================================================================
    # 资源释放
    # =====================================================================

    def close(self) -> None:
        """关闭 mem0 资源。

        注：mem0 的 Memory 类没有显式 close 方法，此处为接口预留。
        """
        pass

    # =====================================================================
    # 内部辅助
    # =====================================================================

    def _format_results(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """将 mem0 的原始返回格式转换为用户要求的标准 dict 列表。"""
        formatted: list[dict[str, Any]] = []
        for item in raw.get("results", []):
            meta = item.get("metadata", {}) or {}
            formatted.append(
                {
                    "id": item.get("id", ""),
                    "name": meta.get("name", ""),
                    "role": meta.get("role", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "content": item.get("memory", ""),
                    "session_id": meta.get("session_id", self.session_id),
                    "type": meta.get("type", ""),
                    "score": item.get("score", 0.0),
                }
            )
        return formatted


# =============================================================================
# 快速调用示例（直接运行本文件即可测试）
# =============================================================================


def demo() -> None:
    """非异步演示：展示 Mem0MemoryManager 的基本用法。"""
    print("=== Mem0MemoryManager 快速演示 ===\n")

    manager = Mem0MemoryManager(
        user_id="demo_user",
        session_id="demo_session",
    )

    # 1. 添加记忆
    print("1. 添加记忆...")
    manager.add("你好，我是fafa，我喜欢喝咖啡", name="fafa", role="user", msg_type="from_user")
    manager.add("你好fafa，很高兴认识你", name="assistant", role="assistant")
    manager.add("今天天气真不错", name="fafa", role="user", msg_type="from_user")
    print("   ✅ 已添加 3 条记忆\n")

    # 2. 语义搜索
    print("2. 语义搜索 '喜欢喝什么' (top_k=3)...")
    results = manager.search("我喜欢喝什么", top_k=3)
    for r in results:
        print(f"   [{r['role']}] {r['name']}: {r['content']} (score={r['score']:.3f})")
    print()

    # 3. 获取全部
    print("3. 获取全部记忆...")
    all_memories = manager.get_all(top_k=10)
    for r in all_memories:
        print(f"   [{r['role']}] {r['name']}: {r['content']}")
    print()

    # 4. 删除测试
    print("4. 清空记忆...")
    deleted = manager.clear()
    print(f"   ✅ 已删除 {deleted} 条记忆\n")

    print("=== 演示完成 ===")


if __name__ == "__main__":
    demo()
