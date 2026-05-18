"""
SiliconFlow ASR 语音识别客户端
支持在线音频转文字，基于 SiliconFlow /v1/audio/transcriptions API

API: https://docs.siliconflow.cn/cn/api-reference/audio/create-audio-transcriptions
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import config


import httpx


class SiliconFlowASR:
    """SiliconFlow 在线 ASR 语音识别客户端"""

    def __init__(self,):
        self.api_key = config.ASR_API_KEY
        self.api_url = config.ASR_BASE_URL
        self.model = config.ASR_MODEL_NAME

    async def transcribe(self, audio_path: str) -> str:
        """将音频文件转录为文本。

        Args:
            audio_path: 本地音频文件路径（支持 wav、mp3、m4a 等格式）

        Returns:
            转录后的文本字符串

        Raises:
            FileNotFoundError: 音频文件不存在
            httpx.HTTPStatusError: API 请求失败
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # multipart/form-data 上传音频文件
        with open(audio_file, "rb") as f:
            files = {
                "file": (audio_file.name, f, "audio/wav"),
                "model": (None, self.model),
            }

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                )
                response.raise_for_status()
                result = response.json()

        text = result.get("text", "").strip()
        print(f"[ASR] 转录完成 | 文件: {audio_file.name} | 文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        return text





async def main():
    audio_path = "data/tmp/sounds/hello_zh.wav"
    asr = SiliconFlowASR()

    audio_file = Path(audio_path)
    text = await asr.transcribe(audio_file)
    print(f"转录结果: {text}")


if __name__ == "__main__":
    asyncio.run(main())