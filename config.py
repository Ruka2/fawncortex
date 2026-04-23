import os

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-max")
STREAM = os.getenv("STREAM", "true").lower() in ("1", "true", "yes")
VTS_HOST = os.getenv("VTS_HOST", "localhost")
VTS_PORT = int(os.getenv("VTS_PORT", "25565"))
