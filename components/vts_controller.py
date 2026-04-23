import asyncio
import pyvts
import random
import math
import time
import json


class VTSController:
    def __init__(self, host: str = "localhost", port: int = 25565):
        self.plugin_info = {
            "plugin_name": "Sentia_AI_Core",
            "developer": "YourName",
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

        self.custom_params = [
            "Sentia_AngleX", "Sentia_AngleY", "Sentia_AngleZ",
            "Sentia_EyeX", "Sentia_EyeY",
            "Sentia_EyeLOpen", "Sentia_EyeROpen",
            "Sentia_BrowLY", "Sentia_BrowRY", "Sentia_BrowLForm", "Sentia_BrowRForm",
            "Sentia_BodyX", "Sentia_BodyY", "Sentia_BodyZ",
            "Sentia_MouthOpenY", "Sentia_MouthForm"
        ]

        self.cur_head_x, self.cur_head_y, self.cur_head_z = 0.0, 0.0, 0.0
        self.cur_body_x, self.cur_body_y, self.cur_body_z = 0.0, 0.0, 0.0
        self.cur_eye_x, self.cur_eye_y = 0.0, 0.0
        self.cur_brow_y = 0.0
        self.cur_mouth_open = 0.0
        self.cur_mouth_form = 0.5

        self.focus_target_x, self.focus_target_y = 0.0, 0.0
        self.last_focus_time = time.perf_counter()
        self.blink_timer = time.perf_counter()
        self.is_blinking = False

        # 随机相位偏移，让每个实例的动画略有不同
        self.phase_x = random.uniform(0, math.pi * 2)
        self.phase_y = random.uniform(0, math.pi * 2)
        self.phase_z = random.uniform(0, math.pi * 2)
        self.phase_eye_x = random.uniform(0, math.pi * 2)
        self.phase_eye_y = random.uniform(0, math.pi * 2)

        # Agent 动作覆盖参数
        self._override_params = {}
        self._override_expiry = 0.0


    async def connect_and_auth(self):
        print("正在连接 VTube Studio API")
        try:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            await self.vts.request_authenticate()
            print("连接成功")

            for param in self.custom_params:
                req_data = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "ParamInit",
                    "messageType": "ParameterCreationRequest",
                    "data": {
                        "parameterName": param,
                        "explanation": "Sentia AI Parameter",
                        "min": -30.0,
                        "max": 30.0,
                        "defaultValue": 0.0
                    }
                }
                await self.vts.request(req_data)

            self.is_alive = True
            self._reader_task = asyncio.create_task(self._blackhole_reader())
            self._idle_task = asyncio.create_task(self._procedural_soul_loop())
        except Exception as e:
            print(f"连接失败: {e}")
            raise

    async def _blackhole_reader(self):
        while self.is_alive:
            try:
                if self.vts.websocket:
                    await self.vts.websocket.recv()
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                break

    # 算法
    def _math_lerp(self, a, b, t):
        return a + (b - a) * t

    def _smooth_damp(self, current, target, speed, dt):
        if dt <= 0.0: return current
        return current + (target - current) * (1.0 - math.exp(-speed * dt))

    def _organic_noise(self, t, speed_multiplier, phase_offset):
        t = t * speed_multiplier
        wave1 = math.sin(t * 0.73 + phase_offset)
        wave2 = math.sin(t * 1.37 + phase_offset * 1.3) * 0.5
        wave3 = math.sin(t * 2.11 + phase_offset * 1.7) * 0.25
        return (wave1 + wave2 + wave3) / 1.75

    # 口型
    async def animate_speech_lip_sync(self, audio_samples, sample_rate, emotion="Neutral"):
        if not self.is_alive:
            return

        fps = 60
        chunk_size = int(sample_rate / fps)
        total_chunks = len(audio_samples) // chunk_size

        start_time = time.perf_counter()
        circle_speed = 15.0

        for i in range(total_chunks):
            t = (i + 1) / fps
            theta = t * circle_speed

            target_open = (math.sin(theta) * 0.5 + 0.5) * 0.8

            target_form = math.cos(theta) * 0.9

            # 无需阻尼，直接暴力赋值！我们要的就是这种机械、鬼畜的几何感！
            self.cur_mouth_open = target_open
            self.cur_mouth_form = target_form

            expected_time = t
            elapsed = time.perf_counter() - start_time
            if expected_time > elapsed:
                await asyncio.sleep(expected_time - elapsed)

        self.cur_mouth_open = 0.0
        self.cur_mouth_form = 0.8


    async def _procedural_soul_loop(self):
        start_time = time.perf_counter()
        last_time = start_time
        target_fps = 1   # 之前为60，次数太多调为1
        frame_time = 1.0 / target_fps

        while self.is_alive:
            try:
                loop_start = time.perf_counter()
                dt = loop_start - last_time
                last_time = loop_start

                t = loop_start - start_time
                current_time = loop_start

                #注意力
                if current_time - self.last_focus_time > random.uniform(1.5, 4.5):
                    # 高斯分布中心严格为 0，绝对对称
                    fx = random.gauss(0.0, 0.5)
                    fy = random.gauss(0.0, 0.5)

                    max_radius = 1.0
                    dist = math.hypot(fx, fy)
                    if dist > max_radius:
                        fx = (fx / dist) * max_radius
                        fy = (fy / dist) * max_radius

                    # self.focus_target_x = fx
                    # self.focus_target_y = fy
                    self.last_focus_time = current_time

                #眼球
                eye_jitter_x = self._organic_noise(t, 3.0, self.phase_eye_x) * 0.1
                eye_jitter_y = self._organic_noise(t, 3.5, self.phase_eye_y) * 0.1

                final_tgt_eye_x = self.focus_target_x + eye_jitter_x
                final_tgt_eye_y = self.focus_target_y + eye_jitter_y

                final_eye_dist = math.hypot(final_tgt_eye_x, final_tgt_eye_y)
                if final_eye_dist > 1.0:
                    final_tgt_eye_x = (final_tgt_eye_x / final_eye_dist) * 1.0
                    final_tgt_eye_y = (final_tgt_eye_y / final_eye_dist) * 1.0

                self.cur_eye_x = self._smooth_damp(self.cur_eye_x, final_tgt_eye_x, 35.0, dt)
                self.cur_eye_y = self._smooth_damp(self.cur_eye_y, final_tgt_eye_y, 35.0, dt)

                #头部
                head_breathing_noise_x = self._organic_noise(t, 0.8, self.phase_x) * 5.0
                head_breathing_noise_y = self._organic_noise(t, 0.6, self.phase_y) * 3.0

                # tgt_head_x = (self.cur_eye_x * 25.0) + head_breathing_noise_x
                # tgt_head_y = (self.cur_eye_y * 18.0) + head_breathing_noise_y
                # tgt_head_z = (self.cur_eye_x * 10.0) + self._organic_noise(t, 0.5, self.phase_z) * 6.0
                tgt_head_x = head_breathing_noise_x
                tgt_head_y = head_breathing_noise_y
                tgt_head_z = self._organic_noise(t, 0.5, self.phase_z) * 6.0

                self.cur_head_x = self._smooth_damp(self.cur_head_x, tgt_head_x, 8.0, dt)
                self.cur_head_y = self._smooth_damp(self.cur_head_y, tgt_head_y, 8.0, dt)
                self.cur_head_z = self._smooth_damp(self.cur_head_z, tgt_head_z, 8.0, dt)

                #身体
                breathing_y = math.sin(t * 2.0 + self.phase_y) * 1.0
                # tgt_body_x = self.cur_head_x * 0.4
                # tgt_body_y = self.cur_head_y * 0.3 + breathing_y
                # tgt_body_z = self.cur_head_z * 0.3
                tgt_body_x = 0.0
                tgt_body_y = breathing_y
                tgt_body_z = 0.0

                self.cur_body_x = self._smooth_damp(self.cur_body_x, tgt_body_x, 4.0, dt)
                self.cur_body_y = self._smooth_damp(self.cur_body_y, tgt_body_y, 4.0, dt)
                self.cur_body_z = self._smooth_damp(self.cur_body_z, tgt_body_z, 4.0, dt)

                #眨眼
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
                        if random.random() < 0.2:
                            self.blink_timer = current_time - random.uniform(0.3, 0.8)
                eye_open = max(0.0, min(eye_open, 1.0))

                # 眉毛
                tgt_brow_y = (self.cur_head_y * 0.08) + ((eye_open - 0.8) * 0.5)
                tgt_brow_y += self._organic_noise(t, 1.2, self.phase_z) * 0.1
                tgt_brow_y = max(-1.0, min(tgt_brow_y, 1.0))
                self.cur_brow_y = self._smooth_damp(self.cur_brow_y, tgt_brow_y, 15.0, dt)

                # 检查 Agent 动作覆盖参数（由 control_body 工具设置）
                current_time = time.perf_counter()
                if current_time < self._override_expiry:
                    send_head_x = self._override_params.get("Sentia_AngleX", self.cur_head_x)
                    send_head_y = self._override_params.get("Sentia_AngleY", self.cur_head_y)
                    send_head_z = self._override_params.get("Sentia_AngleZ", self.cur_head_z)
                    send_body_x = self._override_params.get("Sentia_BodyX", self.cur_body_x)
                    send_body_y = self._override_params.get("Sentia_BodyY", self.cur_body_y)
                    send_eye_x = self._override_params.get("Sentia_EyeX", self.cur_eye_x)
                    send_eye_y = self._override_params.get("Sentia_EyeY", self.cur_eye_y)
                    send_eye_open = self._override_params.get("Sentia_EyeLOpen", eye_open)
                    send_mouth_open = self._override_params.get("Sentia_MouthOpenY", self.cur_mouth_open)
                    send_mouth_form = self._override_params.get("Sentia_MouthForm", self.cur_mouth_form)
                else:
                    send_head_x = self.cur_head_x
                    send_head_y = self.cur_head_y
                    send_head_z = self.cur_head_z
                    send_body_x = self.cur_body_x
                    send_body_y = self.cur_body_y
                    send_eye_x = self.cur_eye_x
                    send_eye_y = self.cur_eye_y
                    send_eye_open = eye_open
                    send_mouth_open = self.cur_mouth_open
                    send_mouth_form = self.cur_mouth_form

                #打包发送
                inject_data = {
                    "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
                    "requestID": "InjectParams", "messageType": "InjectParameterDataRequest",
                    "data": {
                        "faceFound": True,
                        "parameterValues": [
                            {"id": "Sentia_AngleX", "value": send_head_x},
                            {"id": "Sentia_AngleY", "value": send_head_y},
                            {"id": "Sentia_AngleZ", "value": send_head_z},
                            {"id": "Sentia_BodyX", "value": send_body_x},
                            {"id": "Sentia_BodyY", "value": send_body_y},
                            {"id": "Sentia_EyeX", "value": send_eye_x},
                            {"id": "Sentia_EyeY", "value": send_eye_y},
                            {"id": "Sentia_EyeLOpen", "value": send_eye_open},
                            {"id": "Sentia_EyeROpen", "value": send_eye_open},
                            {"id": "Sentia_MouthOpenY", "value": send_mouth_open},
                            {"id": "Sentia_MouthForm", "value": send_mouth_form}
                        ]
                    }
                }
                if self.vts.websocket:
                    await self.vts.websocket.send(json.dumps(inject_data))

                elapsed = time.perf_counter() - loop_start
                await asyncio.sleep(max(0.001, frame_time - elapsed))

            except Exception as e:
                await asyncio.sleep(1)

    async def close(self):
        self.is_alive = False
        if self._idle_task: 
            self._idle_task.cancel()
        if self.vts.websocket:
            await self.vts.close()