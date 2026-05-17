import os

from dotenv import load_dotenv
load_dotenv()

### 项目启动设置

# 用于项目管道数值控制的参数
# 大脑智能体最大思考时间
BRAIN_TIMEOUT = 300
# 每次容忍大脑智能体中间思考的时间间隔，超过这个时间之后自动截断正在生成的思维过程，并发送到对话中进行提前回复
BRAIN_CUT_TIME_DURATION = 5

# 默认智能体名称
AGENT_NAME = "Ruka"
USER_NAME = "鹿过"



### LLM模型配置
# 默认全局LLM配置（用于本项目非智能体相关的数据清洗、快速调试
LLM_API_KEY = os.getenv("LLM_API_KEY", "")      # e.g. OPENAI_API_KEY "sk-xxx"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # e.g. OPENAI_BASE_URL "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")    # e.g. qwen-max

### 多智能体LLM配置映射表
# 项目中的智能体/模块可以根据推理任务难度配置不同的云端LLM API
# 每个角色需独立设置 api_key / base_url / model_name / generate_kwargs
# 若某项未设置（空字符串或None），则自动回退到上述LLM的默认全局配置
# 支持的角色 key：chat(对话), emotion(表情), brain(大脑), reflection(反思), memory(记忆系统)
def _role_cfg(role: str):
    """读取单个角色的LLM配置，未设置时回退到默认全局配置。"""
    prefix = role.upper()
    return {
        "api_key": os.getenv(f"{prefix}_LLM_API_KEY", LLM_API_KEY),
        "base_url": os.getenv(f"{prefix}_LLM_BASE_URL", LLM_BASE_URL),
        "model_name": os.getenv(f"{prefix}_LLM_MODEL_NAME", LLM_MODEL_NAME),
    }

# 完全配置（每个都填）
# 拷贝配置，防止每次调整浪费时间
LLM_ROLE_CONFIG = {
    "chat": _role_cfg("chat"),
    "emotion": _role_cfg("chat"),      # 沿用chat_agent
    "brain": _role_cfg("brain"),
    "reflection": _role_cfg("brain"),  # 沿用brain_agent
    "memory": _role_cfg("brain"),      # 沿用brain_agent
}

# 角色专属 generate_kwargs（按模型特性定制，例如是否开启 thinking）
# 如需为某角色开启 thinking，可修改为 {"extra_body": {"enable_thinking": True}} ，必须是所使用的模型基座有这样的选项才有用
# 部分LLM加了禁用think模式的入参之后会报错，需要注意
LLM_ROLE_GENERATE_KWARGS = {
    "chat": {"extra_body": {"enable_thinking": False}},
    "emotion": {"extra_body": {"enable_thinking": False}},
    "reflection": {"extra_body": {"enable_thinking": False}},
    "brain": {"extra_body": {"enable_thinking": True}},
    "memory": {"extra_body": {"enable_thinking": True}},
}

### LLM推理细节设置
# 智能体所使用到的LLM是否流式输出（若设置为False，则大脑智能体将不再流式生成，会导致pipeline变为一问一答的情况，响应延迟会变得特别大）
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


### Semantic Scholar 论文搜索配置
S2_API_KEY = os.getenv("S2_API_KEY", "")

