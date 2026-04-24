"""
AgentScope SQLite 记忆快速上手
本项目即将作废，英文AgentScope的记忆没有记忆召回功能
================================
基于 AsyncSQLAlchemyMemory，提供持久化的对话记忆存储。

功能：
- 创建 SQLite 异步记忆库
- 按 user_id / session_id 隔离记忆
- 添加、读取、删除、按标记筛选记忆

运行：
    python memory_sqlite.py
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from agentscope.memory import AsyncSQLAlchemyMemory
from agentscope.message import Msg


# =============================================================================
# 封装好的 SQLite 记忆管理器
# =============================================================================


class SQLiteMemoryManager:
    """基于 SQLite 的 AgentScope 记忆管理器。"""

    def __init__(self, db_path: str = "./data/db/agent_memory.db") -> None:
        """初始化并创建异步引擎。

        Args:
            db_path: SQLite 数据库文件路径，默认放在 ./data/db/ 目录下。
        """
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async def create_memory(
        self,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> AsyncSQLAlchemyMemory:
        """创建一个新的 AsyncSQLAlchemyMemory 实例。

        Args:
            user_id: 用户 ID，用于多用户隔离。
            session_id: 会话 ID，用于同用户下的多会话隔离。

        Returns:
            AsyncSQLAlchemyMemory 实例。
        """
        return AsyncSQLAlchemyMemory(
            engine_or_session=self.engine,
            user_id=user_id,
            session_id=session_id,
        )

    async def save(
        self,
        memory: AsyncSQLAlchemyMemory,
        msg: Msg,
        mark: str | None = None,
    ) -> None:
        """向记忆中添加一条消息。

        Args:
            memory: AsyncSQLAlchemyMemory 实例。
            msg: AgentScope 的 Msg 消息对象。
            mark: 可选的标记字符串，用于后续筛选（如 "hint", "important"）。
        """
        if mark:
            await memory.add(msg, marks=mark)
        else:
            await memory.add(msg)

    async def load_all(self, memory: AsyncSQLAlchemyMemory) -> list[Msg]:
        """读取当前记忆中的所有消息。

        Returns:
            Msg 对象列表。
        """
        return await memory.get_memory()

    async def load_by_mark(
        self,
        memory: AsyncSQLAlchemyMemory,
        mark: str,
    ) -> list[Msg]:
        """按标记读取消息。

        Args:
            memory: AsyncSQLAlchemyMemory 实例。
            mark: 标记字符串。

        Returns:
            带有该标记的 Msg 列表。
        """
        return await memory.get_memory(mark=mark)

    async def delete_by_mark(
        self,
        memory: AsyncSQLAlchemyMemory,
        mark: str,
    ) -> int:
        """按标记删除消息。

        Returns:
            被删除的消息数量。
        """
        return await memory.delete_by_mark(mark)

    async def clear(self, memory: AsyncSQLAlchemyMemory) -> None:
        """清空当前记忆。"""
        await memory.clear()

    async def close(self) -> None:
        """关闭数据库引擎，释放连接。"""
        await self.engine.dispose()


# =============================================================================
# 演示用法
# =============================================================================


async def main() -> None:
    # 1. 初始化管理器（数据库文件会自动创建）
    manager = SQLiteMemoryManager(db_path="./data/db/agent_memory.db")

    # 2. 为 "user_1" 的 "chat_001" 会话创建记忆
    memory = await manager.create_memory(
        user_id="user_1",
        session_id="chat_001",
    )

    # 3. 保存普通对话消息
    await manager.save(memory, Msg("user", "你好", "user"))
    await manager.save(memory, Msg("assistant", "你好呀！", "assistant"))
    await manager.save(memory, Msg("user", "今天天气怎么样？", "user"))

    # 4. 保存带标记的提示消息（如系统提示、重要上下文）
    await manager.save(
        memory,
        Msg("system", "<system-hint>回复请保持简洁。</system-hint>", "system"),
        mark="hint",
    )

    # 5. 读取所有记忆
    print("📦 所有记忆：")
    all_msgs = await manager.load_all(memory)
    for msg in all_msgs:
        print(f"   [{msg.role}] {msg.name}: {msg.content}")

    # 6. 按标记读取
    print("\n🏷️  带 'hint' 标记的记忆：")
    hint_msgs = await manager.load_by_mark(memory, mark="hint")
    for msg in hint_msgs:
        print(f"   [{msg.role}] {msg.name}: {msg.content}")

    # 7. 删除标记消息
    deleted = await manager.delete_by_mark(memory, mark="hint")
    print(f"\n🗑️  已删除 {deleted} 条带 'hint' 标记的消息")

    print("\n📦 删除后的记忆：")
    remaining = await manager.load_all(memory)
    for msg in remaining:
        print(f"   [{msg.role}] {msg.name}: {msg.content}")

    # 8. 关闭资源
    await memory.close()
    await manager.close()
    print("\n✅ SQLite 记忆演示完成")


if __name__ == "__main__":
    asyncio.run(main())
