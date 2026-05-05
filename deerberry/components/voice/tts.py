"""
SiliconFlow CosyVoice2-0.5B 流式 TTS 客户端
API: https://api.siliconflow.cn/v1/audio/speech
"""

import io
import time
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# 可选依赖：安装后可自动播放
# pip install sounddevice soundfile
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class SiliconFlowCosyVoice:
    """SiliconFlow CosyVoice2-0.5B 流式 TTS 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.siliconflow.cn/v1/audio/speech",
        model: str = "FunAudioLLM/CosyVoice2-0.5B",
        voice: str = "FunAudioLLM/CosyVoice2-0.5B:diana",
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("请提供 api_key 参数")
        self.api_url = api_url
        self.model = model
        self.voice = voice

    def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        gain: float = 0.0,
        response_format: str = "mp3",
        save_path: Optional[str] = None,
        play: bool = False,
    ) -> bytes:
        """
        流式合成语音。

        Args:
            text: 要合成的文本
            voice: 音色 ID，格式为 "模型名:voice_id"。
                CosyVoice2 预设女声: diana(开朗)、claire(温柔)、bella(热情)、anna(稳重)
                CosyVoice2 预设男声: david(阳光)、charles(磁性)、benjamin(低沉)、alex(稳重)
                示例: "FunAudioLLM/CosyVoice2-0.5B:diana"
            speed: 语速倍率，0.25 ~ 4.0，默认 1.0
            gain: 音量增益(dB)，-10 ~ 10，默认 0.0
            response_format: 音频格式 mp3 / wav / opus / pcm
            save_path: 保存路径，如 "output.mp3"
            play: 合成完成后是否自动播放

        Returns:
            完整音频二进制数据
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        use_voice = voice if voice is not None else self.voice
        payload = {
            "model": self.model,
            "input": text,
            # "input": f"使用小男孩的声音<|endofprompt|>{text}",
            "voice": use_voice,
            "speed": speed,
            "gain": gain,
            "response_format": response_format,
        }

        # print(f"🎙️ 流式合成: {text[:50]}{'...' if len(text) > 50 else ''}")
        t0 = time.perf_counter()

        audio_buffer = io.BytesIO()
        first_chunk_time = None
        chunk_count = 0

        # ========== 核心：流式请求 ==========
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            stream=True,  # 启用流式接收
            timeout=20,
        )
        response.raise_for_status()

        # 边接收边写入 buffer
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - t0
                    # print(f"语音合成(TTS): 首音频块延迟 {first_chunk_time:.4f}ms | 音频大小 {len(audio_bytes)} bytes, {chunk_count} chunks | 总生成耗时 {total_time:.2f}s")

                audio_buffer.write(chunk)
                chunk_count += 1

        total_time = time.perf_counter() - t0
        audio_bytes = audio_buffer.getvalue()

        # print(
        #     f"🔊 完成 | {len(audio_bytes)} bytes | "
        #     f"{chunk_count} chunks | 总耗时 {total_time:.2f}s"
        # )
        
        print(f"[TTS] 首音频块延迟 {first_chunk_time:.3f}s | 总TTS处理耗时 {total_time:.3f}s | 音频大小 {len(audio_bytes)} bytes, {chunk_count} chunks")

        # 保存到文件
        if save_path:
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            print(f"💾 已保存: {save_path}")

        # 播放
        if play and HAS_AUDIO:
            self._play(audio_bytes)
        elif play and not HAS_AUDIO:
            print("⚠️ pip install sounddevice soundfile 以支持播放")

        return audio_bytes

    def _play(self, audio_bytes: bytes):
        """播放音频"""
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32")
            sd.play(audio, sr)
            sd.wait()
            # print("🔊 播放完毕")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import config

    # 流式合成 + 播放 + 保存
    tts = SiliconFlowCosyVoice(
        api_key=config.TTS_API_KEY,
        api_url=config.TTS_BASE_URL,
        model=config.TTS_MODEL_NAME,
        voice=config.TTS_VOICE,
    )
    tts.stream_synthesize(
        text="你好，我正在测试流式语音合成效果，一二三四五六七八，testing testing",
        speed=1.2,
        gain=0.0,
        response_format="mp3",
        save_path="output.mp3",
        play=True,
    )