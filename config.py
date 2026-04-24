import os

from dotenv import load_dotenv
load_dotenv()


# LLM配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-max")

# LLM是否流式输出
STREAM = os.getenv("STREAM", "true").lower() in ("1", "true", "yes")


# Vtube Studio 端口配置
VTS_HOST = os.getenv("VTS_HOST", "localhost")
VTS_PORT = int(os.getenv("VTS_PORT", "25565"))


# 向量模型配置
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

# 缓存数据存放位置
MEM0_VECTOR_STORE_PATH = os.getenv("MEM0_VECTOR_STORE_PATH", "./data/db/mem0_embedd_chroma")
MEM0_HISTORY_DB_PATH = os.getenv("MEM0_HISTORY_DB_PATH", "./data/db/mem0_raw_chroma.db")
