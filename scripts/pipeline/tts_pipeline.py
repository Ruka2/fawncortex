
import asyncio

# 外部（非智能体内核）组件
from components.tts import SiliconFlowCosyVoice
from components.vts_controller import VTSController
from scripts.tools.control_vts import set_vts_controller   # fixme: 暂时不知道这个有什么用，先保留待VTS正式载入

# =============================================================================
# TTS 异步包装
# =============================================================================
async def speak_async(tts: SiliconFlowCosyVoice, text: str) -> None:
    """在线程池中执行 TTS 合成与播放，避免阻塞 asyncio 事件循环。"""
    if not text or not text.strip():
        return

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: tts.stream_synthesize(
                text=text.strip(),
                voice="FunAudioLLM/CosyVoice2-0.5B:diana",
                speed=1.0,
                gain=0.0,
                response_format="mp3",
                play=True,
            ),
        )
    except Exception as e:
        print(f"⚠️  TTS 合成/播放失败: {e}")