import os

from dotenv import load_dotenv
load_dotenv()

### LLM模型配置
# LLM配置
LLM_API_KEY = os.getenv("LLM_API_KEY", "")      # e.g. OPENAI_API_KEY "sk-xxx"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # e.g. OPENAI_BASE_URL "https://dashscope.aliyuncs.com/compatible-mode/v1" 
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")    # e.g. qwen-max

### LLM推理细节设置
# 智能体所使用到的LLM是否流式输出
STREAM = True  # e.g. enum[True, False]

### 向量嵌入配置
# 向量模型配置
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")  # e.g. Qwen/Qwen3-Embedding-0.6B

### TTS语音合成配置
# TTS模型
TTS_API_KEY = os.getenv("TTS_API_KEY", "")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "")
TTS_VOICE = os.getenv("TTS_VOICE", "")

### 项目文件存放位置
# 缓存数据存放位置
MEM0_VECTOR_STORE_PATH = os.getenv("MEM0_VECTOR_STORE_PATH", "./data/db/mem0_embedd_chroma")
MEM0_HISTORY_DB_PATH = os.getenv("MEM0_HISTORY_DB_PATH", "./data/db/raw_history.db")

# 日志存放位置
LOG_DIR = os.getenv("LOG_DIR", "./logs")

### 外部接口设置
# Vtube Studio 端口配置
VTS_HOST = os.getenv("VTS_HOST", "localhost")
VTS_PORT = int(os.getenv("VTS_PORT", "25565"))

