import asyncio
import pyvts
import random
import math
import time
import json
import os

import config


class VTSController:
    """专用于 defaultboy06V2 模型的 VTube Studio API 控制器。

    直接注入 VTube Studio 内置追踪参数（Input），由 VTS 根据
    defaultboy06v2.vtube.json 中的 ParameterSettings 映射到 Live2D 参数。

    注入参数与 Live2D 映射关系（参考 ParameterSettings）：
        FaceAngleX    → ParamAngleX / ParamBodyAngleX  (InputRange -30~30)
        FaceAngleY    → ParamAngleY / ParamBodyAngleY  (InputRange -20~20)
        FaceAngleZ    → ParamAngleZ / ParamBodyAngleZ  (InputRange -30~30)
        EyeRightX     → ParamEyeBallX                  (InputRange -1~1)
        EyeRightY     → ParamEyeBallY                  (InputRange -1~1)
        EyeOpenLeft   → ParamEyeLOpen                  (InputRange 0~1)
        EyeOpenRight  → ParamEyeROpen                  (InputRange 0~1)
        Brows         → ParamBrowLForm / ParamBrowRForm(InputRange 0~1 → Output -1~1)
        MouthSmile    → ParamMouthForm / ParamEyeLSmile/ ParamEyeRSmile (InputRange 0~1)
        MouthOpen     → ParamMouthOpenY                (InputRange 0~1)
        MouthX        → Paramkuchi                     (InputRange -1~1)
    """

    def __init__(self, host: str = config.VTS_HOST, port: int = config.VTS_PORT):
        self.plugin_info = {
            "plugin_name": "DefaultBoy",
            "developer": "Ruka",
            "authentication_token_path": "./vts_token.txt"
        }
        vts_api_info = {
            "version": "1.0",
            "name": "VTubeStudioPublicAPI",
            "host": host,
            "port": port,
        }
        self.vts = pyvts.vts(plugin_info=self.plugin_info, vts_api_info=vts_api_info)
        self.is_alive = False
        self._idle_task = None
        self._reader_task = None

        # 当前动画状态（内部逻辑值，部分需要在注入时映射到 VTS InputRange）
        self.cur_head_x, self.cur_head_y, self.cur_head_z = 0.0, 0.0, 0.0
        self.cur_eye_x, self.cur_eye_y = 0.0, 0.0
        self.cur_eye_open = 1.0          # 同时用于左右眼
        self.cur_brow_form = 0.0         # 内部范围 -1~1，注入 Brows 时映射为 (form+1)/2
        self.cur_mouth_form = 0.0        # 内部范围 -1~1，注入 MouthSmile 时映射为 (form+1)/2
        self.cur_mouth_open = 0.0        # 直接注入 MouthOpen
        self.cur_mouth_x = 0.0           # 直接注入 MouthX（范围 -1~1）

        # 注意力与眨眼系统
        self.focus_target_x, self.focus_target_y = 0.0, 0.0
        self.last_focus_time = time.perf_counter()
        self.blink_timer = time.perf_counter()
        self.is_blinking = False

        # 随机相位偏移
        self.phase_x = random.uniform(0, math.pi * 2)
        self.phase_y = random.uniform(0, math.pi * 2)
        self.phase_z = random.uniform(0, math.pi * 2)
        self.phase_eye_x = random.uniform(0, math.pi * 2)
        self.phase_eye_y = random.uniform(0, math.pi * 2)
        self.phase_brow = random.uniform(0, math.pi * 2)

        # Agent 动作覆盖参数（key 使用 VTS Input 参数名）
        self._override_params = {}
        self._override_expiry = 0.0

        # 嘴型目标值（由表情动画设置，供 lip sync 叠加使用）
        self._mouth_target = 0.0

        # WebSocket 写入锁（防止并发 send 导致连接断开）
        self._send_lock = asyncio.Lock()

        # 自定义动画标志（防止与 _procedural_idle_loop 冲突）
        self._custom_animating = False

        # lip sync 包络状态（由 TTS 模块写入）
        self._lip_sync_envelope = 0.0

        # 姿态过渡（动画结束 → idle 的平滑插值，避免跳变）
        self._last_sent_params = {}       # inject_now 最后发送的参数
        self._transition_active = False   # 是否正在过渡
        self._transition_start = 0.0      # 过渡开始时间
        self._transition_duration = 0.0   # 过渡持续时间
        self._was_custom_animating = False  # 上一帧是否处于自定义动画

    # ------------------------------------------------------------------
    # 连接与认证
    # ------------------------------------------------------------------
    async def connect_and_auth(self):
        print("正在连接 VTube Studio API")
        try:
            await self.vts.connect()

            # ---------- 鉴权流程 ----------
            # 从 token 文件读取或向 VTS 申请新 token
            await self.vts.request_authenticate_token()
            auth_ok = await self.vts.request_authenticate()

            # Token 无效或被撤销时，强制重新请求
            if not auth_ok:
                print("Token 无效或已被撤销，正在重新申请...")
                token_path = self.plugin_info.get("authentication_token_path", "./vts_token.txt")
                if os.path.exists(token_path):
                    os.remove(token_path)
                    print(f"已删除旧 token 文件: {token_path}")
                # await self.vts.request_authenticate_token(force=True)
                
                await self.vts.request_authenticate_token()
                print("请在 VTube Studio 弹窗中点击【允许】以授权插件...")

                for _ in range(60):
                    auth_ok = await self.vts.request_authenticate()
                    if auth_ok:
                        break
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError("等待 VTube Studio 授权超时，请重试")
            print("连接成功")

            # 无需创建自定义参数，直接注入 VTS 内置 Input 参数即可
            self.is_alive = True
            self._reader_task = asyncio.create_task(self._blackhole_reader())
            self._idle_task = asyncio.create_task(self._procedural_idle_loop())
        except Exception as e:
            print(f"连接失败: {e}")
            raise

    async def _blackhole_reader(self):
        """持续读取 VTS WebSocket 响应，防止接收缓冲区满导致断连。

        遇到连接关闭类异常时标记 is_alive=False 并退出，
        其他异常（如临时网络抖动）则继续循环。
        """
        while self.is_alive:
            try:
                if self.vts.websocket:
                    await self.vts.websocket.recv()
                else:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # 正常取消（如引擎关闭），直接退出
                break
            except Exception as e:
                # 区分连接关闭 vs 其他异常
                err_name = type(e).__name__
                if err_name in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError", "WebSocketException"):
                    print(f"[VTS] WebSocket 连接已关闭 ({err_name})，停止 reader")
                    self.is_alive = False
                    break
                # 其他异常（如临时读取失败）打印后继续
                print(f"[VTS] _blackhole_reader 异常（将继续重试）: {e}")
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _math_lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_param(self, key: str, idle_val: float, t: float) -> float:
        """从 _last_sent_params 向 idle_val 做 lerp 过渡。"""
        from_val = self._last_sent_params.get(key, idle_val)
        return from_val + (idle_val - from_val) * t

    def _smooth_damp(self, current, target, speed, dt):
        if dt <= 0.0:
            return current
        return current + (target - current) * (1.0 - math.exp(-speed * dt))

    def _organic_noise(self, t, speed_multiplier, phase_offset):
        t = t * speed_multiplier
        wave1 = math.sin(t * 0.73 + phase_offset)
        wave2 = math.sin(t * 1.37 + phase_offset * 1.3) * 0.5
        wave3 = math.sin(t * 2.11 + phase_offset * 1.7) * 0.25
        return (wave1 + wave2 + wave3) / 1.75


    # ------------------------------------------------------------------
    # 核心 Idle 动画循环（60fps 实时计算并注入参数）
    # ------------------------------------------------------------------
    async def _procedural_idle_loop(self):
        start_time = time.perf_counter()
        last_time = start_time
        target_fps = 60
        frame_time = 1.0 / target_fps

        while self.is_alive:
            try:
                loop_start = time.perf_counter()
                dt = loop_start - last_time
                last_time = loop_start

                t = loop_start - start_time
                current_time = loop_start

                # ---------- 注意力系统 + 眼球追踪（说话时 eye focus） ----------
                if self._custom_animating:
                    # 说话时注视前方，减少大幅度漂移和 jitter
                    self.focus_target_x = 0.0
                    self.focus_target_y = 0.0
                    self.last_focus_time = current_time
                    eye_jitter_x = self._organic_noise(t, 3.0, self.phase_eye_x) * 0.01
                    eye_jitter_y = self._organic_noise(t, 3.5, self.phase_eye_y) * 0.01
                else:
                    # 正常 idle 模式
                    if current_time - self.last_focus_time > random.uniform(1.5, 4.5):
                        fx = random.gauss(0.0, 0.5)
                        fy = random.gauss(0.0, 0.5)
                        max_radius = 0.1
                        dist = math.hypot(fx, fy)
                        if dist > max_radius:
                            fx = (fx / dist) * max_radius
                            fy = (fy / dist) * max_radius
                        self.focus_target_x = fx
                        self.focus_target_y = fy
                        self.last_focus_time = current_time
                    eye_jitter_x = self._organic_noise(t, 3.0, self.phase_eye_x) * 0.1
                    eye_jitter_y = self._organic_noise(t, 3.5, self.phase_eye_y) * 0.1

                final_tgt_eye_x = self.focus_target_x + eye_jitter_x
                final_tgt_eye_y = self.focus_target_y + eye_jitter_y

                final_eye_dist = math.hypot(final_tgt_eye_x, final_tgt_eye_y)
                if final_eye_dist > 1.0:
                    final_tgt_eye_x = (final_tgt_eye_x / final_eye_dist) * 0.1
                    final_tgt_eye_y = (final_tgt_eye_y / final_eye_dist) * 0.1

                self.cur_eye_x = self._smooth_damp(self.cur_eye_x, final_tgt_eye_x, 35.0, dt)
                self.cur_eye_y = self._smooth_damp(self.cur_eye_y, final_tgt_eye_y, 35.0, dt)

                # ---------- 头部姿态（说话时仍保持 idle 呼吸微动） ----------
                head_breathing_noise_x = self._organic_noise(t, 0.8, self.phase_x) * 5.0
                head_breathing_noise_y = self._organic_noise(t, 0.6, self.phase_y) * 3.0
                head_breathing_noise_z = self._organic_noise(t, 0.5, self.phase_z) * 6.0

                tgt_head_x = (self.cur_eye_x * 25.0) + head_breathing_noise_x
                tgt_head_y = (self.cur_eye_y * 18.0) + head_breathing_noise_y
                tgt_head_z = (self.cur_eye_x * 10.0) + head_breathing_noise_z

                tgt_head_y = max(-20.0, min(20.0, tgt_head_y))

                self.cur_head_x = self._smooth_damp(self.cur_head_x, tgt_head_x, 8.0, dt)
                self.cur_head_y = self._smooth_damp(self.cur_head_y, tgt_head_y, 8.0, dt)
                self.cur_head_z = self._smooth_damp(self.cur_head_z, tgt_head_z, 8.0, dt)

                # ---------- 眨眼系统（说话时保留自然眨眼） ----------
                eye_open = 1.0 + self._organic_noise(t, 5.0, self.phase_y) * 0.05
                if not self.is_blinking and current_time - self.blink_timer > random.uniform(2.0, 4.5):
                    self.is_blinking = True
                    self.blink_timer = current_time
                if self.is_blinking:
                    blink_progress = current_time - self.blink_timer
                    if blink_progress < 0.08:
                        eye_open = self._math_lerp(1.0, 0.0, blink_progress / 0.08)
                    elif blink_progress < 0.23:
                        eye_open = self._math_lerp(0.0, 1.0, (blink_progress - 0.08) / 0.15)
                    else:
                        self.is_blinking = False
                        eye_open = 1.0
                        if random.random() < 0.05:
                            self.blink_timer = current_time - random.uniform(0.3, 0.8)
                eye_open = max(0.0, min(eye_open, 1.0))

                # ---------- 眉毛形态 ----------
                tgt_brow = (self.cur_head_y * 0.06) + ((eye_open - 0.8) * 1.3)
                tgt_brow += self._organic_noise(t, 1.2, self.phase_brow) * 0.1
                tgt_brow = max(-1.0, min(1.0, tgt_brow))
                self.cur_brow_form = self._smooth_damp(self.cur_brow_form, tgt_brow, 12.0, dt)

                # ---------- 嘴型形态（说话时由 override/lip sync 控制，idle 不驱动嘴部） ----------
                if self._custom_animating:
                    tgt_mouth_form = 0.0
                    tgt_mouth_x = 0.0
                else:
                    tgt_mouth_form = self._organic_noise(t, 0.3, self.phase_x) * 1.05
                    tgt_mouth_form = max(-0.8, min(0.4, tgt_mouth_form))
                    tgt_mouth_x = self._organic_noise(t, 0.4, self.phase_x) * 0.55
                self.cur_mouth_form = self._smooth_damp(self.cur_mouth_form, tgt_mouth_form, 5.0, dt)
                self.cur_mouth_x = self._smooth_damp(self.cur_mouth_x, tgt_mouth_x, 2.0, dt)

                # ---------- 准备发送参数 ----------
                current_time = time.perf_counter()
                send_head_x = self.cur_head_x
                send_head_y = self.cur_head_y
                send_head_z = self.cur_head_z
                send_eye_x = self.cur_eye_x
                send_eye_y = self.cur_eye_y
                send_eye_open_l = eye_open
                send_eye_open_r = eye_open
                send_brow = (self.cur_brow_form + 1.0) / 2.0
                send_mouth_smile = (self.cur_mouth_form + 1.0) / 2.0
                send_mouth_open = self.cur_mouth_open
                send_mouth_x = self.cur_mouth_x

                # 记录自定义动画状态，用于检测动画结束并启动过渡
                if self._custom_animating:
                    self._was_custom_animating = True

                # 标记刚结束自定义动画，启动姿态过渡
                if self._was_custom_animating and not self._custom_animating:
                    self._was_custom_animating = False
                    if self._last_sent_params:
                        self._transition_start = current_time
                        self._transition_duration = random.uniform(0.4, 1.0)
                        self._transition_active = True
                        print(f"[VTS] 🌊 姿态过渡启动: {self._transition_duration:.2f}s")

                if self._transition_active:
                    elapsed = current_time - self._transition_start
                    if elapsed >= self._transition_duration:
                        self._transition_active = False
                        self._last_sent_params = {}
                        print("[VTS] 🌊 姿态过渡完成")
                    else:
                        t = min(1.0, elapsed / self._transition_duration)
                        send_head_x    = self._lerp_param("FaceAngleX",   send_head_x,    t)
                        send_head_y    = self._lerp_param("FaceAngleY",   send_head_y,    t)
                        send_head_z    = self._lerp_param("FaceAngleZ",   send_head_z,    t)
                        send_eye_x     = self._lerp_param("EyeRightX",    send_eye_x,     t)
                        send_eye_y     = self._lerp_param("EyeRightY",    send_eye_y,     t)
                        send_eye_open_l = self._lerp_param("EyeOpenLeft", send_eye_open_l, t)
                        send_eye_open_r = self._lerp_param("EyeOpenRight", send_eye_open_r, t)
                        send_brow      = self._lerp_param("Brows",        send_brow,      t)
                        send_mouth_smile = self._lerp_param("MouthSmile", send_mouth_smile, t)
                        send_mouth_open = self._lerp_param("MouthOpen",   send_mouth_open, t)
                        send_mouth_x   = self._lerp_param("MouthX",       send_mouth_x,   t)

                # ---------- 应用 override（动画/lip sync 只覆盖嘴部+眉毛） ----------
                for key, (val, expiry) in list(self._override_params.items()):
                    if current_time < expiry:
                        if key == "MouthSmile":   send_mouth_smile = val
                        elif key == "MouthOpen":  send_mouth_open = val
                        elif key == "MouthX":     send_mouth_x = val
                        elif key == "Brows":      send_brow = val
                    else:
                        del self._override_params[key]

                # ---------- 打包注入 VTube Studio ----------
                if self.is_alive and self.vts.websocket:
                    inject_data = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "InjectParams",
                        "messageType": "InjectParameterDataRequest",
                        "data": {
                            "faceFound": True,
                            "parameterValues": [
                                {"id": "FaceAngleX",   "value": send_head_x},
                                {"id": "FaceAngleY",   "value": send_head_y},
                                {"id": "FaceAngleZ",   "value": send_head_z},
                                {"id": "EyeRightX",    "value": send_eye_x},
                                {"id": "EyeRightY",    "value": send_eye_y},
                                {"id": "EyeOpenLeft",  "value": send_eye_open_l},
                                {"id": "EyeOpenRight", "value": send_eye_open_r},
                                {"id": "Brows",        "value": send_brow},
                                {"id": "MouthSmile",   "value": send_mouth_smile},
                                {"id": "MouthOpen",    "value": send_mouth_open},
                                {"id": "MouthX",       "value": send_mouth_x},
                            ]
                        }
                    }
                    try:
                        async with self._send_lock:
                            if self.vts.websocket:
                                await self.vts.websocket.send(json.dumps(inject_data))
                    except Exception as e:
                        err_name = type(e).__name__
                        if err_name in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"):
                            print(f"[VTS] WebSocket 发送时连接已关闭 ({err_name})")
                            self.is_alive = False
                        # 其他异常（如临时发送失败）静默处理，避免刷屏

                elapsed = time.perf_counter() - loop_start
                await asyncio.sleep(max(0.001, frame_time - elapsed))

            except Exception as e:
                print(f"_procedural_idle_loop 异常: {e}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # 立即发送参数（供外部直接调用，不等待 idle 循环）
    # ------------------------------------------------------------------
    async def inject_now(self, param_values: dict):
        """设置参数覆盖，由 idle loop 在下一帧统一注入。

        动画函数和 lip sync 通过此机制覆盖嘴部/眉毛参数，
        头部和眼睛始终由 idle loop 控制，保持自然呼吸和 focus。

        Args:
            param_values: 参数名 -> 数值 的字典，例如 {"MouthOpen": 0.5}
        """
        if not self.is_alive:
            return
        expiry = time.perf_counter() + 0.3  # 1.0s 过期，确保覆盖注入间隔
        for k, v in param_values.items():
            self._override_params[k] = (v, expiry)
        # 更新最后发送的参数，供姿态过渡使用
        self._last_sent_params.update(param_values)

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    async def close(self):
        self.is_alive = False
        if self._idle_task:
            self._idle_task.cancel()
        if self._reader_task:
            self._reader_task.cancel()
        if self.vts.websocket:
            await self.vts.close()
