"""
长期记忆管理器封装
==================
负责 Mem0LongTermMemory 的初始化配置与实例创建，
将向量存储、嵌入模型、LLM 配置集中管理，供主程序和工具模块调用。
本代码文件目前只有初始化数据库使用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from agentscope.memory import Mem0LongTermMemory
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

    return Mem0LongTermMemory(
        agent_name=agent_name,
        user_name=user_name,
        model=OpenAIChatModel(
            model_name=llm_model_name,
            api_key=llm_api_key,
            stream=False,
            client_kwargs={"base_url": llm_base_url},
        ),
        embedding_model=OpenAITextEmbedding(
            model_name=embedding_model_name,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        ),
        mem0_config=mem0_config,
    )
