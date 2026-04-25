"""
长期记忆管理器封装
==================
负责 Mem0LongTermMemory 的初始化配置与实例创建，
将向量存储、嵌入模型、LLM 配置集中管理，供主程序和工具模块调用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from agentscope.memory import Mem0LongTermMemory
from agentscope.model import OpenAIChatModel
from agentscope.embedding import OpenAITextEmbedding


# =============================================================================
# 全局：ChromaDB 向量存储配置 + mem0 主配置
# =============================================================================
_VECTOR_STORE_CONFIG = VectorStoreConfig(
    provider="chroma",
    config={
        "path": config.MEM0_VECTOR_STORE_PATH,
    },
)

_MEM0_CONFIG = MemoryConfig(
    vector_store=_VECTOR_STORE_CONFIG,
    history_db_path=config.MEM0_HISTORY_DB_PATH,
)


# =============================================================================
# 工厂函数：创建长期记忆实例
# =============================================================================
def create_long_term_memory(
    agent_name: str = "default_agent",
    user_name: str = "default_user",
) -> Mem0LongTermMemory:
    """创建并返回 Mem0LongTermMemory 实例。

    Args:
        agent_name: Agent 标识名，用于记忆隔离。
        user_name: 用户标识名，用于记忆隔离。

    Returns:
        配置好的 Mem0LongTermMemory 实例。
    """
    return Mem0LongTermMemory(
        agent_name=agent_name,
        user_name=user_name,
        model=OpenAIChatModel(
            model_name=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            stream=False,
            client_kwargs={"base_url": config.OPENAI_BASE_URL},
        ),
        embedding_model=OpenAITextEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        ),
        mem0_config=_MEM0_CONFIG,
    )
