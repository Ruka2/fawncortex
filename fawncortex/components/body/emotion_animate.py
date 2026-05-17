"""
VTube Studio 表情动画库
存放各种持续播放的自定义表情动画函数。
每个动画都以 60fps 持续向 VTS WebSocket 发送 InjectParameterDataRequest，
"""

import asyncio
import math
import random
import time

# 全局可配置帧率（默认 60fps）
DEFAULT_FPS = 60

# 表情对应的基础嘴型值（用于 lip sync 叠加模式）
# 大笑基准高，生气基准紧闭，不同表情下相同音量嘴张程度不同
EMOTION_MOUTH_BASE = {
    "open_mouse": 1.00,   # 大笑
    "smile":      0.40,   # 微笑
    "angry":      0.00,   # 生气：紧闭
    "surprised":  1.00,   # 惊讶
    "smirk":      0.15,   # 坏笑
    "confused":   0.05,   # 疑惑
    "neural":     0.00,   # 无表情
    "lean_left":  0.00,   # 身体动作不影响嘴型
    "lean_right": 0.00,
    "look_up":    0.00,
    "look_down":  0.00,
    "wink_left":  0.00,
    "wink_right": 0.00,
    "close_eyes": 0.00,
}

# 可用动态表情列表
AVAILABLE_ACTIONS = {
    "emotion": ["neural", "smile", "angry", "surprised", "confused"],  # 表情组合
    "eye_motion": ["wink_left", "wink_right", "close_eyes"],                   # 眼睛动作
    "body_motion": ["lean_left", "lean_right", "look_up", "look_down"],        # 身体姿态
}

# AVAILABLE_ACTIONS = {
#     "emotion": ["neural", "smile", "angry", "surprised", "smirk", "confused", "open_mouse"],  # 表情组合
#     "eye_motion": ["wink_left", "wink_right", "close_eyes"],                   # 眼睛动作
#     "body_motion": ["lean_left", "lean_right", "look_up", "look_down"],        # 身体姿态
# }

def _breath_noise(t: float) -> dict:
    """生成微量呼吸噪声，模拟 idle 时的有机微动。
    """
    # 头部呼吸（多频叠加，模拟 _organic_noise）
    breath_x = math.sin(t * 0.70 * 2 * math.pi) * 0.80 + math.sin(t * 1.30 * 2 * math.pi) * 0.40
    breath_y = math.sin(t * 0.50 * 2 * math.pi) * 0.50 + math.sin(t * 1.10 * 2 * math.pi) * 0.30
    breath_z = math.sin(t * 0.60 * 2 * math.pi) * 0.60 + math.sin(t * 0.90 * 2 * math.pi) * 0.30

    # 眼球轻微 jitter
    eye_jitter_x = math.sin(t * 2.50 * 2 * math.pi) * 0.030 + math.sin(t * 4.00 * 2 * math.pi) * 0.015
    eye_jitter_y = math.sin(t * 3.00 * 2 * math.pi) * 0.020 + math.sin(t * 4.50 * 2 * math.pi) * 0.010

    # 嘴部呼吸式张合
    mouth_breath = math.sin(t * 0.60 * 2 * math.pi) * 0.04

    # 眉毛微动
    brow_breath = math.sin(t * 0.40 * 2 * math.pi) * 0.06

    # 嘴型微动
    mouth_form_breath = math.sin(t * 0.50 * 2 * math.pi) * 0.00

    # 嘴部水平微偏移
    mouth_x_breath = math.sin(t * 0.05 * 2 * math.pi) * 0.00

    return {
        "angle_x": breath_x,
        "angle_y": breath_y,
        "angle_z": breath_z,
        "eye_x": eye_jitter_x,
        "eye_y": eye_jitter_y,
        "mouth_open": mouth_breath,
        "brow_form": brow_breath,
        "mouth_form": mouth_form_breath,
        "mouth_x": mouth_x_breath,
    }


async def animate_open_mouse(vts, duration: float = 5.0):
    """张嘴动画：嘴巴从闭合慢慢张大，伴随呼吸微动。

    动作设计：
    - 嘴巴从 0 慢慢张开到最大（0.6s 张开过渡），然后保持
    - 眼睛正常睁开（1.0），伴随随机眨眼
    - 眉毛开心上扬（1.0）
    - 眼球看向斜上方，带轻微 jitter
    - 叠加微量呼吸
    - 0.25s 渐入 + 0.4s 渐出，过渡平滑
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放大笑动画")
        return

    print(f"  😂 启动大笑动画 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    # 预生成随机眨眼时间点（1~2 次）
    blink_times = []
    if duration > 1.2:
        n_blinks = random.randint(1, 2)
        for _ in range(n_blinks):
            bt = random.uniform(0.4, duration - 0.6)
            blink_times.append(bt)

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.25)
            fade_out = min(1.0, (duration - t) / 0.40)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            # ---- 嘴部：从闭合慢慢张大（0.6s 过渡）----
            open_progress = min(1.0, t / 0.60)
            mouth_open = open_progress * fade + br["mouth_open"]

            # 大笑嘴型：持续微笑，嘴角微动
            mouth_form_raw = 0.80 + math.sin(t * 2.8 * 2 * math.pi) * 0.08
            mouth_form = -1.0 + (mouth_form_raw + 1.0) * fade + br["mouth_form"]

            # ---- 眼睛：正常睁开 + 眨眼 ----
            eye_open = 1.0

            # 眨眼覆盖
            blink_val = eye_open
            for bt in blink_times:
                delta = t - bt
                if 0.0 <= delta < 0.06:
                    blink_val = 0.02
                    break
                elif 0.06 <= delta < 0.16:
                    progress = (delta - 0.06) / 0.10
                    blink_val = 0.02 + (eye_open - 0.02) * progress
                    break

            # ---- 眉毛：上扬 ----
            brow_form = 1.0 * fade + br["brow_form"]

            # ---- 头部：后仰 + 随笑声摇摆 + 呼吸 ----
            angle_y_raw = math.sin(t * 0.6 * 2 * math.pi) * 1.0
            angle_y = angle_y_raw * fade + br["angle_y"]

            angle_z_raw = math.sin(t * 0.4 * 2 * math.pi) * 1.0
            angle_z = angle_z_raw * fade + br["angle_z"]

            # ---- 眼球：斜上方开心眼神 + jitter ----
            eye_x = (0.05 + math.sin(t * 2.5 * 2 * math.pi) * 0.04) * fade + br["eye_x"]
            eye_y = (-0.14 + math.sin(t * 1.9 * 2 * math.pi) * 0.03) * fade + br["eye_y"]

            # ---- 嘴X：轻微偏移 + 呼吸 ----
            mouth_x = math.sin(t * 1.2 * 2 * math.pi) * 0.10 * fade + br["mouth_x"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, blink_val)),
                "EyeOpenRight": max(0.0, min(1.0, blink_val)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False


async def animate_smile(vts, duration: float = 5.0):
    """微笑动画：温和持续的表情，模拟自然微笑状态。
    适合表达友好、温柔、安静的愉悦情绪。

    动作设计：
    - 嘴巴几乎不动，仅轻微呼吸式波动（mouth_open 0.05~0.12）
    - 嘴角持续上扬（mouth_form 0.50~0.72），带缓慢波动
    - 眼睛轻微眯起（0.78~0.92），温柔而不夸张
    - 眉毛轻微上扬（0.20~0.40），比大笑柔和
    - 头部轻微晃动，幅度很小（angle_y ±2.5）
    - 眼球缓慢漂移，幅度极小，营造放松感
    - 0.5s 柔和渐入 + 0.6s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放微笑动画")
        return

    print(f"  😊 启动微笑动画 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    # 微笑期间眨眼更少、更慢
    blink_times = []
    if duration > 2.0:
        n_blinks = random.randint(1, 3)
        for _ in range(n_blinks):
            bt = random.uniform(0.8, duration - 0.8)
            blink_times.append(bt)

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            # ---- 柔和渐入渐出 ----
            fade_in = min(1.0, t / 0.50)
            fade_out = min(1.0, (duration - t) / 0.60)
            fade = fade_in * fade_out

            # ---- 嘴部：几乎不动，轻微呼吸波动 ----
            # 很慢的 0.9Hz 节奏，幅度极小
            mouth_open = 0.05

            # 持续微笑嘴角，缓慢波动
            mouth_form = -1.0

            # ---- 眼睛：温柔微眯 + 慢眨眼 ----
            eye_open = 0.45

            # 慢眨眼覆盖
            blink_val = eye_open
            for bt in blink_times:
                delta = t - bt
                if 0.0 <= delta < 0.08:
                    blink_val = 0.03
                    break
                elif 0.08 <= delta < 0.22:
                    progress = (delta - 0.08) / 0.14
                    blink_val = 0.03 + (eye_open - 0.03) * progress
                    break

            # ---- 眉毛：轻微上扬，非常缓慢 ----
            brow_form_raw = 0.30 + math.sin(t * 0.5 * 2 * math.pi) * 0.08
            brow_form = brow_form_raw * fade

            # ---- 头部：轻微晃动，幅度很小 ----
            angle_y_raw = math.sin(t * 0.6 * 2 * math.pi) * 1.0
            angle_y = angle_y_raw * fade

            angle_z_raw = math.sin(t * 0.4 * 2 * math.pi) * 1.0
            angle_z = angle_z_raw * fade

            # ---- 眼球：极缓慢漂移，放松感 ----
            eye_x = (0.02 + math.sin(t * 0.35 * 2 * math.pi) * 0.03) * fade
            eye_y = (-0.05 + math.sin(t * 0.25 * 2 * math.pi) * 0.02) * fade

            # ---- 嘴X：极轻微偏移 ----
            mouth_x = math.sin(t * 0.5 * 2 * math.pi) * 0.06 * fade

            # ---- 构建 VTS 参数并发送 ----
            params = {
                "FaceAngleX": 0.0,
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, blink_val)),
                "EyeOpenRight": max(0.0, min(1.0, blink_val)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😊 微笑动画结束")


async def animate_angry(vts, duration: float = 3.0):
    """生气动画：眉毛下压，眼睛瞪视，表情紧绷。

    动作设计：
    - brow_form = -1（眉毛下压，触发 VTS 生气眉眼形态）
    - 眼睛瞪大直视（eye_open 0.90~0.98）
    - 嘴巴紧闭或微张（mouth_open 0.03~0.08）
    - 嘴型平直或微下（mouth_form -0.3~0.0）
    - 头部微微前倾（angle_y 2~6），带紧张抖动
    - 眼球直视前方，带轻微愤怒抖动
    - 0.3s 快速渐入 + 0.4s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放生气动画")
        return

    print(f"  😠 启动生气动画 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.30)
            fade_out = min(1.0, (duration - t) / 0.40)
            fade = fade_in * fade_out

            # 眉毛下压（核心：brow_form = -1 触发生气眉眼）
            brow_form = -1.0 * fade

            # 眼睛瞪大直视，带极轻微紧张抖动
            eye_open = (0.4 + math.sin(t * 8.0 * 2 * math.pi) * 0.03) * fade
            eye_open = max(0.0, min(1.0, eye_open))

            # 嘴巴紧闭
            mouth_open = -0.05 * fade

            # 嘴型平直微下
            mouth_form = -0.2 * fade

            # 头部微微前倾 + 紧张抖动
            angle_y_raw = 4.0 + math.sin(t * 6.0 * 2 * math.pi) * 0.5
            angle_y = angle_y_raw * fade

            # 眼球直视前方，带愤怒 jitter
            eye_x = (math.sin(t * 7.0 * 2 * math.pi) * 0.03) * fade
            eye_y = (math.sin(t * 5.5 * 2 * math.pi) * 0.02) * fade

            # 嘴X微偏
            mouth_x = (math.sin(t * 3.0 * 2 * math.pi) * 0.08) * fade

            params = {
                "FaceAngleX": 0.0,
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": 0.0,
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😠 生气动画结束")


async def animate_wink_left(vts, duration: float = 1.5):
    """左眼眨眼（wink）：左眼闭合，右眼正常，带俏皮感。

    动作设计：
    - 左眼完全闭合（EyeOpenLeft = 0）
    - 右眼正常睁开（0.85~1.0）
    - 嘴角微微上扬，眉毛轻微挑动
    - 头部轻微歪向右侧（angle_z 负值）
    - 0.15s 快速渐入 + 0.2s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放左眼 wink")
        return

    print(f"  😉 启动左眼 wink (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.15)
            fade_out = min(1.0, (duration - t) / 0.20)
            fade = fade_in * fade_out

            # 左眼闭合，右眼正常
            eye_open_left = 0.0
            eye_open_right = 0.90 * fade

            # 嘴角微微上扬
            mouth_form = 0.4 * fade
            mouth_open = 0.08 * fade

            # 眉毛：左眉微挑（但 brow_form 是整体参数，用轻微上扬模拟）
            brow_form = 0.25 * fade

            # 头部轻微歪向右侧（angle_z 负）
            angle_z = (-1.0 + math.sin(t * 0.5 * 1 * math.pi) * 1.0) * fade
            angle_y = (-1.0 + math.sin(t * 0.5 * 1 * math.pi) * 1.0) * fade

            # 眼球微微看向左侧（配合 wink）
            eye_x = (-0.10 + math.sin(t * 3.0 * 2 * math.pi) * 0.03) * fade
            eye_y = (-0.05 + math.sin(t * 2.5 * 2 * math.pi) * 0.02) * fade

            mouth_x = math.sin(t * 2.0 * 2 * math.pi) * 0.06 * fade

            params = {
                "FaceAngleX": 0.0,
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open_left)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open_right)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😉 左眼 wink 结束")


async def animate_wink_right(vts, duration: float = 1.5):
    """右眼眨眼（wink）：右眼闭合，左眼正常，带俏皮感。

    动作设计：
    - 右眼完全闭合（EyeOpenRight = 0）
    - 左眼正常睁开（0.85~1.0）
    - 嘴角微微上扬，眉毛轻微挑动
    - 头部轻微歪向左侧（angle_z 正值）
    - 0.15s 快速渐入 + 0.2s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放右眼 wink")
        return

    print(f"  😉 启动右眼 wink (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.15)
            fade_out = min(1.0, (duration - t) / 0.20)
            fade = fade_in * fade_out

            # 右眼闭合，左眼正常
            eye_open_left = 0.90 * fade
            eye_open_right = 0.0

            # 嘴角微微上扬
            mouth_form = 0.4 * fade
            mouth_open = 0.08 * fade

            # 眉毛轻微上扬
            brow_form = 0.25 * fade

            # 头部轻微歪向左侧（angle_z 正）
            angle_z = (-1.0 + math.sin(t * 0.5 * 1 * math.pi) * 1.0) * fade
            angle_y = (-1.0 + math.sin(t * 0.5 * 1 * math.pi) * 1.0) * fade

            # 眼球微微看向右侧
            eye_x = (0.10 + math.sin(t * 3.0 * 2 * math.pi) * 0.03) * fade
            eye_y = (-0.05 + math.sin(t * 2.5 * 2 * math.pi) * 0.02) * fade

            mouth_x = math.sin(t * 2.0 * 2 * math.pi) * 0.06 * fade

            params = {
                "FaceAngleX": 0.0,
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open_left)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open_right)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😉 右眼 wink 结束")


async def animate_close_eyes(vts, duration: float = 2.0):
    """双眼闭眼：双目完全闭合，表情放松。

    动作设计：
    - 双眼完全闭合（EyeOpenLeft = 0, EyeOpenRight = 0）
    - 嘴巴放松微张（mouth_open 0.05~0.10）
    - 嘴型自然（mouth_form 0.0~0.2）
    - 眉毛放松（brow_form 0.0~0.1）
    - 头部微微放松晃动
    - 0.3s 柔和渐入 + 0.3s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放闭眼动画")
        return

    print(f"  😌 启动双眼闭眼 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.30)
            fade_out = min(1.0, (duration - t) / 0.30)
            fade = fade_in * fade_out

            # 双眼闭合
            eye_open_left = 0.0
            eye_open_right = 0.0

            # 嘴巴放松
            mouth_open = 0
            mouth_form = 0

            # 眉毛放松
            brow_form = 1 * fade

            # 头部放松微晃
            angle_y = (math.sin(t * 0.4 * 2 * math.pi) * 1.5) * fade
            angle_z = (math.sin(t * 0.3 * 2 * math.pi) * 2.0) * fade

            # 眼球在闭眼状态下回到中心
            eye_x = 0.0
            eye_y = 0.0

            mouth_x = math.sin(t * 0.5 * 2 * math.pi) * 0.04 * fade

            params = {
                "FaceAngleX": 0.0,
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": 0.0,
                "EyeRightY": 0.0,
                "EyeOpenLeft": 0.0,
                "EyeOpenRight": 0.0,
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😌 双眼闭眼结束")


async def animate_confused(vts, duration: float = 3.0):
    """疑惑动画：身体侧弯，眉毛不对称下压，嘴巴偏移，营造困惑感。

    动作设计：
    - 身体侧弯 angle_x = ±20（随机左/右侧弯）
    - brow_form 随机负数（-0.6 ~ -1.0），营造皱眉困惑感
    - mouth_x 随机最大值或最小值（-1.0 或 1.0），嘴巴偏向一侧
    - 眼睛睁大（0.85~0.95），带困惑的轻微晃动
    - 头部微微倾斜，带不解的摇摆
    - 0.4s 渐入 + 0.5s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放疑惑动画")
        return

    # 随机决定侧弯方向、嘴巴偏移方向、眉毛深度
    side_direction = random.choice([-1.0, 1.0])  # -1 左弯, 1 右弯
    mouth_x_target = random.choice([-0.3, 0.3])
    brow_target = random.uniform(-1.0, -0.6)

    print(f"  🤔 启动疑惑动画 (duration={duration:.1f}s)")
    print(f"     侧弯方向: {'右' if side_direction > 0 else '左'}, "
          f"嘴偏: {'右' if mouth_x_target > 0 else '左'}, "
          f"皱眉: {brow_target:.2f}")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.40)
            fade_out = min(1.0, (duration - t) / 0.50)
            fade = fade_in * fade_out

            # 身体侧弯（核心：angle_z = ±20）
            angle_x = (math.sin(t * 0.3 * 2 * math.pi) * 2.0) * fade
            
            angle_z_raw = 20.0 * side_direction
            angle_z = angle_z_raw * fade

            # 眉毛随机负数下压
            brow_form = brow_target * fade

            # 嘴巴随机偏向一侧
            mouth_x = mouth_x_target * fade

            # 眼睛睁大，带困惑抖动
            eye_open = (0.88 + math.sin(t * 4.0 * 2 * math.pi) * 0.04) * fade
            eye_open = max(0.0, min(1.0, eye_open))

            # 嘴型微张，困惑感
            mouth_open = 0.0 * fade
            mouth_form = -0.1 * fade

            # 头部倾斜 + 困惑摇摆
            angle_y = (1.0 + math.sin(t * 1.0 * 2 * math.pi) * 1.0) * fade
            # angle_z = (math.sin(t * 1.0 * 2 * math.pi) * 4.0) * fade

            # 眼球：困惑地左右看
            eye_x = (math.sin(t * 1.8 * 2 * math.pi) * 0.05) * fade
            eye_y = (-0.08 + math.sin(t * 1.2 * 2 * math.pi) * 0.1) * fade

            params = {
                "FaceAngleX": max(-30.0, min(30.0, angle_x)),
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, eye_x)),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  🤔 疑惑动画结束")


async def animate_lean_left(vts, duration: float = 3.0):
    """身体向左倾斜：angle_z = -30，无表情，面部保持 neutral。

    动作设计：
    - angle_z = -30（向左侧弯）
    - 眼睛正常睁开（1.0），嘴巴闭合，眉毛平直
    - 眼球回到中心
    - 0.3s 渐入 + 0.3s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放左倾动画")
        return

    print(f"  ⬅️  启动左倾 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.30)
            fade_out = min(1.0, (duration - t) / 0.30)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            angle_z = -30.0 * fade + br["angle_z"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, br["angle_y"])),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, br["eye_y"])),
                "EyeOpenLeft": 1.0,
                "EyeOpenRight": 1.0,
                "Brows": max(0.0, min(1.0, (br["brow_form"] + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (br["mouth_form"] + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, br["mouth_open"])),
                "MouthX": 0.0,
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  ⬅️  左倾结束")


async def animate_lean_right(vts, duration: float = 3.0):
    """身体向右倾斜：angle_z = 30，无表情，面部保持 neutral。

    动作设计：
    - angle_z = 30（向右侧弯）
    - 眼睛正常睁开（1.0），嘴巴闭合，眉毛平直
    - 眼球回到中心
    - 0.3s 渐入 + 0.3s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放右倾动画")
        return

    print(f"  ➡️  启动右倾 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.30)
            fade_out = min(1.0, (duration - t) / 0.30)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            angle_z = 30.0 * fade + br["angle_z"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, br["angle_y"])),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, br["eye_y"])),
                "EyeOpenLeft": 1.0,
                "EyeOpenRight": 1.0,
                "Brows": max(0.0, min(1.0, (br["brow_form"] + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (br["mouth_form"] + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, br["mouth_open"])),
                "MouthX": 0.0,
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  ➡️  右倾结束")


async def animate_look_up(vts, duration: float = 2.0):
    """向上看：眼球上移，头部微仰。

    动作设计：
    - eye_y = 1.0（眼球最上）
    - angle_y = 8（头部微微上扬）
    - 眼睛睁大（1.0）
    - 0.25s 渐入 + 0.25s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放向上看")
        return

    print(f"  👆 启动向上看 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.25)
            fade_out = min(1.0, (duration - t) / 0.25)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            eye_y = 1.0 * fade + br["eye_y"]
            angle_y = 8.0 * fade + br["angle_y"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, br["angle_z"])),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": 1.0,
                "EyeOpenRight": 1.0,
                "Brows": max(0.0, min(1.0, (br["brow_form"] + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (br["mouth_form"] + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, br["mouth_open"])),
                "MouthX": 0.0,
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  👆 向上看结束")


async def animate_look_down(vts, duration: float = 2.0):
    """向下看：眼球下移，头部微垂。

    动作设计：
    - eye_y = -1.0（眼球最下）
    - angle_y = -8（头部微微低垂）
    - 眼睛稍微眯起（0.75）
    - 0.25s 渐入 + 0.25s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放向下看")
        return

    print(f"  👇 启动向下看 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.25)
            fade_out = min(1.0, (duration - t) / 0.25)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            eye_y = -1.0 * fade + br["eye_y"]
            angle_y = -8.0 * fade + br["angle_y"]
            eye_open = 0.75 * fade + 0.25

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, br["angle_z"])),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, eye_y)),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open)),
                "Brows": max(0.0, min(1.0, (br["brow_form"] + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (br["mouth_form"] + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, br["mouth_open"])),
                "MouthX": 0.0,
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  👇 向下看结束")


async def animate_surprised(vts, duration: float = 2.5):
    """惊讶动画：眉毛最高上扬，双眼睁最大，嘴巴先大张后慢慢闭上，头部先仰后垂。

    动作设计：
    - brow_form = 1（眉毛最高上扬）
    - eye_open = 1.0（双眼睁到最大）
    - mouth_open 先 1.0 后衰减到 0.2（嘴巴先大张再慢慢闭上）
    - mouth_form = 0.0（嘴型呈 O 形）
    - 头部先后仰（angle_y ≈ -10）再往前垂下（angle_y ≈ +8），然后回正
    - 叠加微量呼吸
    - 0.2s 快速渐入 + 0.3s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放惊讶动画")
        return

    print(f"  😲 启动惊讶动画 (duration={duration:.1f}s)")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.20)
            fade_out = min(1.0, (duration - t) / 0.30)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            # 眉毛最高上扬
            brow_form = 1.0 * fade + br["brow_form"]

            # 双眼睁最大
            eye_open = 1.0 * fade

            # 嘴巴先大张后慢慢闭上（从 1.0 线性衰减到 0.2）
            progress = t / duration if duration > 0 else 0.0
            mouth_open_raw = max(0.2, 1.0 - progress * 0.8)
            mouth_open = mouth_open_raw * fade + br["mouth_open"]

            # 嘴型呈 O 形
            mouth_form = 0.0 + br["mouth_form"]

            # 头部先后仰再往前垂下（正弦半波：从 -10 → +8 → -10）
            head_wave = math.sin(progress * math.pi) * 18.0
            angle_y = (-10.0 + head_wave) * fade + br["angle_y"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, angle_y)),
                "FaceAngleZ": max(-30.0, min(30.0, br["angle_z"])),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, -0.1 * fade + br["eye_y"])),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, br["mouth_x"])),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😲 惊讶动画结束")


async def animate_smirk(vts, duration: float = 3.0):
    """坏笑动画：眉毛下压，眼睛半眯，嘴巴偏向一侧，带痞气。

    动作设计：
    - brow_form = -1（眉毛下压，坏坏的）
    - eye_open = 0.65（眼睛稍微闭）
    - mouth_x = ±1.0（嘴巴随机偏向一侧）
    - mouth_form = 0.6（嘴角上扬）
    - 头部轻微倾斜
    - 0.3s 渐入 + 0.4s 渐出
    """
    if not vts.vts.websocket:
        print("  ❌ VTS 未连接，无法播放坏笑动画")
        return

    mouth_x_target = random.choice([-0.3, 0.3])
    print(f"  😏 启动坏笑动画 (duration={duration:.1f}s, 嘴偏: {'右' if mouth_x_target > 0 else '左'})")

    vts._custom_animating = True
    anim_start = time.perf_counter()
    frame_time = 1.0 / DEFAULT_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            t = loop_start - anim_start
            if t >= duration:
                break

            fade_in = min(1.0, t / 0.30)
            fade_out = min(1.0, (duration - t) / 0.40)
            fade = fade_in * fade_out

            br = _breath_noise(t)

            brow_form = -1.0 * fade + br["brow_form"]
            eye_open = 0.35 * fade
            mouth_x = mouth_x_target * fade + br["mouth_x"]
            mouth_form = 0.00 * fade + br["mouth_form"]
            mouth_open = 0.00 * fade + br["mouth_open"]
            angle_z = (mouth_x_target * 5.0) * fade + br["angle_z"]

            params = {
                "FaceAngleX": max(-30.0, min(30.0, br["angle_x"])),
                "FaceAngleY": max(-20.0, min(20.0, br["angle_y"])),
                "FaceAngleZ": max(-30.0, min(30.0, angle_z)),
                "EyeRightX": max(-1.0, min(1.0, br["eye_x"])),
                "EyeRightY": max(-1.0, min(1.0, br["eye_y"])),
                "EyeOpenLeft": max(0.0, min(1.0, eye_open)),
                "EyeOpenRight": max(0.0, min(1.0, eye_open)),
                "Brows": max(0.0, min(1.0, (brow_form + 1.0) / 2.0)),
                "MouthSmile": max(0.0, min(1.0, (mouth_form + 1.0) / 2.0)),
                "MouthOpen": max(0.0, min(1.0, mouth_open)),
                "MouthX": max(-1.0, min(1.0, mouth_x)),
            }
            await vts.inject_now(params)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_time - elapsed))

    finally:
        vts._custom_animating = False
        print("  😏 坏笑动画结束")
