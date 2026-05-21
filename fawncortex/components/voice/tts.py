"""
SiliconFlow CosyVoice2-0.5B 流式 TTS 客户端
支持 PCM 流式播放 + 实时 lip sync（RMS 音量分析驱动 MouthOpen）

API: https://api.siliconflow.cn/v1/audio/speech
"""

import asyncio
import io
import math
import time
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np

# 可选依赖：安装后可自动播放
# pip install sounddevice
try:
    import sounddevice as sd
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
        self.api_url = api_url
        self.model = model
        self.voice = voice
        self._current_stream: Optional[sd.OutputStream] = None
        self._stop_playback = False

    async def stream_synthesize(
        self,
        text: str,
        tone: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        gain: float = 0.0,
        response_format: str = "pcm",
        sample_rate: int = 44100,
        save_path: Optional[str] = None,
        play: bool = False,
        vts_controller=None,
        base_mouth_open: float = 0.05,
    ) -> bytes:
        """
        流式合成语音。

        Args:
            text: 要合成的文本
            tone: 语气口吻，会拼接到 prompt 中
            voice: 音色 ID
            speed: 语速倍率，0.25 ~ 4.0
            gain: 音量增益(dB)，-10 ~ 10
            response_format: 音频格式 mp3 / wav / opus / pcm，默认 pcm（支持流式 lip sync）
            sample_rate: 采样率，pcm 默认 44100
            save_path: 保存路径
            play: 是否边合成边播放（pcm 模式下支持流式播放 + lip sync）
            vts_controller: VTSController 实例，用于实时 lip sync
            base_mouth_open: 嘴型基础值（由当前表情决定）

        Returns:
            完整音频二进制数据
        """
        if not HAS_AUDIO and play:
            print("⚠️ pip install sounddevice 以支持播放")
            play = False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        use_voice = voice if voice is not None else self.voice

        # 拼接 tone + text
        text = text.strip()
        # if tone:
        #     input_text = tone + "。" + "<|endofprompt|>" + text
        # else:
        #     input_text = text
        input_text = text

        payload = {
            "model": self.model,
            "input": input_text,
            "voice": use_voice,
            "speed": speed,
            "gain": gain,
            "response_format": response_format,
        }
        # 只有 pcm/wav 才传 sample_rate，mp3/opus 由服务端决定
        if response_format in ("pcm", "wav"):
            payload["sample_rate"] = sample_rate

        t0 = time.perf_counter()
        audio_buffer = io.BytesIO()
        first_chunk_time = None
        chunk_count = 0

        # ========== 核心：流式请求（使用 httpx 异步，避免阻塞事件循环） ==========
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            # ========== PCM 流式播放 + lip sync ==========
            if play and response_format == "pcm":
                audio_bytes = await self._stream_play_pcm(
                    response_iter=response.aiter_bytes(chunk_size=4096),
                    sample_rate=sample_rate,
                    vts_controller=vts_controller,
                    base_mouth_open=base_mouth_open,
                    audio_buffer=audio_buffer,
                )
                total_time = time.perf_counter() - t0
                print(
                    f"[TTS] PCM 流式播放完成 | 总耗时 {total_time:.3f}s | "
                    f"音频大小 {len(audio_bytes)} bytes"
                )
            else:
                # 传统模式：先全部下载，再播放（mp3/wav 等）
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - t0
                        audio_buffer.write(chunk)
                        chunk_count += 1

                total_time = time.perf_counter() - t0
                audio_bytes = audio_buffer.getvalue()

                print(
                    f"[TTS] 首音频块延迟 {first_chunk_time:.3f}s | "
                    f"总TTS处理耗时 {total_time:.3f}s | "
                    f"音频大小 {len(audio_bytes)} bytes, {chunk_count} chunks"
                )

            # if play:
            #     self._play(audio_bytes)

        # 保存到文件
        if save_path:
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            print(f"💾 已保存: {save_path}")

        return audio_bytes

    async def _stream_play_pcm(
        self,
        response_iter,
        sample_rate: int,
        vts_controller,
        base_mouth_open: float,
        audio_buffer: io.BytesIO,
    ) -> bytes:
        """PCM 流式播放：边接收边播放边分析 RMS 驱动 lip sync。

        Returns:
            完整音频二进制数据（所有 chunk 拼接后的结果）
        """
        self._stop_playback = False

        # ── 设备自适应：尝试打开输出流，支持采样率/设备回退 ──
        stream = None
        sample_rates_to_try = [sample_rate, 48000, 44100, 22050]
        # 去重并保持顺序
        seen = set()
        unique_rates = []
        for sr in sample_rates_to_try:
            if sr not in seen:
                seen.add(sr)
                unique_rates.append(sr)

        # 尝试策略：(默认设备+高延迟) → (自动选择设备+高延迟) → 仅保存
        strategies = [
            {"device": None, "latency": "high"},
            {"device": None, "latency": "low"},
        ]

        last_error = None
        for strategy in strategies:
            for sr in unique_rates:
                try:
                    stream = sd.OutputStream(
                        samplerate=sr,
                        channels=1,
                        dtype="float32",
                        blocksize=1024,
                        device=strategy["device"],
                        latency=strategy["latency"],
                    )
                    stream.start()
                    dev_name = strategy["device"]
                    if dev_name is None:
                        try:
                            dev_name = sd.query_devices(kind="output")["name"]
                        except Exception:
                            dev_name = "默认输出"
                    if sr != sample_rate:
                        print(
                            f"[TTS] 使用回退采样率 {sr}Hz"
                            f"（原请求 {sample_rate}Hz 不被当前设备支持）"
                        )
                    print(f"[TTS] 音频输出设备: {dev_name}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            if stream is not None:
                break

        if stream is None:
            print(f"⚠️  TTS 合成/播放失败: {last_error}")
            try:
                default_dev = sd.query_devices(kind="output")
                # print(
                #     f"    当前默认输出: {default_dev.get('name', 'unknown')}"
                # )
            except Exception:
                pass
            print("    提示: 蓝牙耳机被其他应用占用时容易出现此问题，"
                  "可尝试断开重连或关闭其他占用音频的应用")
            # 无法打开音频设备，仅将数据写入 buffer 后返回
            async for chunk in response_iter:
                if chunk:
                    audio_buffer.write(chunk)
            return audio_buffer.getvalue()

        self._current_stream = stream

        pcm_buffer = b""
        chunk_count = 0
        lip_sync_counter = 0
        loop = asyncio.get_event_loop()

        # lip sync 平滑状态
        _rms_history = []
        _envelope = 0.0

        try:
            async for chunk in response_iter:
                if self._stop_playback:
                    print("[TTS] ⏹️ 播放被用户打断")
                    break

                if not chunk:
                    continue

                pcm_buffer += chunk
                audio_buffer.write(chunk)

                # PCM int16 mono = 2 bytes per sample，对齐到完整采样
                valid_bytes = (len(pcm_buffer) // 2) * 2
                if valid_bytes == 0:
                    continue

                audio_bytes = pcm_buffer[:valid_bytes]
                pcm_buffer = pcm_buffer[valid_bytes:]

                # int16 -> float32 (-1.0 ~ 1.0)
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # ── 1. 播放（放到线程池，避免阻塞 asyncio 事件循环）──
                await loop.run_in_executor(None, stream.write, audio_array)

                # ── 2. 实时 lip sync（每 3 个 chunk 注入一次，约 70ms）──
                lip_sync_counter += 1
                if vts_controller is not None and lip_sync_counter % 3 == 0:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    # 叠加模式：表情基础嘴型 + 音量动态
                    mouth_target = getattr(vts_controller, "_mouth_target", 0.0)
                    mouth_open = mouth_target + rms * 3.0
                    mouth_open = max(-1.0, min(1.0, mouth_open))

                    try:
                        await vts_controller.inject_now(
                            {"MouthOpen": float(mouth_open)}
                        )
                    except Exception:
                        # lip sync 失败静默处理，不阻断播放，避免刷屏
                        pass

                chunk_count += 1

        finally:
            stream.stop()
            stream.close()
            self._current_stream = None

            # 播放结束后重置包络并嘴闭上
            if vts_controller is not None:
                try:
                    vts_controller._lip_sync_envelope = 0.0
                    await vts_controller.inject_now({"MouthOpen": 0.0})
                except Exception:
                    pass

        # 返回已写入 audio_buffer 的数据
        return audio_buffer.getvalue()

    def _play(self, audio_bytes: bytes):
        """传统整体播放（mp3/wav 等格式回退）"""
        try:
            import soundfile as sf
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32")
                sd.play(audio, sr)
                sd.wait()
        except Exception as e:
            print(f"⚠️  音频播放失败: {e}")

    def stop(self) -> None:
        """停止当前正在播放的音频（用户新输入打断时调用）。"""
        self._stop_playback = True
        if self._current_stream is not None:
            try:
                self._current_stream.stop()
                self._current_stream.close()
            except Exception:
                pass
            self._current_stream = None
        if HAS_AUDIO:
            try:
                sd.stop()
            except Exception:
                pass


# ==================== 使用示例 ====================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    import config

    async def test():
        tts = SiliconFlowCosyVoice(
            api_key=config.TTS_API_KEY,
            api_url=config.TTS_BASE_URL,
            model=config.TTS_MODEL_NAME,
            voice=config.TTS_VOICE,
        )
        await tts.stream_synthesize(
            text="你好，我正在测试 PCM 流式语音合成和实时嘴型同步效果。",
            tone="",
            speed=1.0,
            gain=0.0,
            response_format="pcm",
            sample_rate=44100,
            play=True,
        )

    asyncio.run(test())
