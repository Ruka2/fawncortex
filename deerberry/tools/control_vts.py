"""
VTube Studio 虚拟形象动作控制工具

提供两类接口：
1. express_emotion(action, duration, intensity) —— 高层预设表情/动作接口
2. control_body(...) —— 底层通用参数控制接口

Agent 侧推荐调用 express_emotion()，只需传入动作名称关键词即可。
"""

import time
from typing import Optional

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

from deerberry.components.visual.vts_controller import VTSController

# 模块级 VTS 控制器引用，由外部通过 set_vts_controller() 注入
_vts: Optional[VTSController] = None


# =============================================================================
# 参数有效范围映射
# =============================================================================

_PARAM_RANGES = {
    "angle_x": (-30.0, 30.0),
    "angle_y": (-30.0, 30.0),
    "angle_z": (-30.0, 30.0),
    "body_x": (-30.0, 30.0),
    "body_y": (-30.0, 30.0),
    "body_z": (-30.0, 30.0),
    "eye_x": (-1.0, 1.0),
    "eye_y": (-1.0, 1.0),
    "eye_l_open": (0.0, 1.0),
    "eye_r_open": (0.0, 1.0),
    "mouth_open": (0.0, 0.9),
    "mouth_form": (-0.8, 1.0),
}


# =============================================================================
# 预设姿态参数库
# =============================================================================
# 每个姿态只设置关键参数，未设置的参数由 VTSController 的 idle 循环接管

_ACTION_PRESETS = {
    # ---- 表情类 ----
    "smile": {
        "mouth_form": 0.75,
        "mouth_open": 0.05,
        "eye_l_open": 0.8,
        "eye_r_open": 0.8,
        "angle_y": -3,
    },
    "happy": {
        "mouth_form": 0.85,
        "mouth_open": 0.2,
        "eye_l_open": 0.7,
        "eye_r_open": 0.7,
        "angle_y": -5,
        "body_y": 3,
    },
    "laugh": {
        "mouth_form": 0.95,
        "mouth_open": 0.45,
        "eye_l_open": 0.55,
        "eye_r_open": 0.55,
        "angle_y": -8,
        "body_y": 4,
        "angle_z": 3,
    },
    "sad": {
        "mouth_form": -0.3,
        "mouth_open": 0.1,
        "eye_l_open": 0.7,
        "eye_r_open": 0.7,
        "angle_y": 5,
        "eye_y": 0.3,
    },
    "cry": {
        "mouth_form": -0.5,
        "mouth_open": 0.2,
        "eye_l_open": 0.5,
        "eye_r_open": 0.5,
        "angle_y": 8,
        "eye_y": 0.4,
        "body_y": -2,
    },
    "angry": {
        "mouth_form": -0.2,
        "mouth_open": 0.15,
        "eye_l_open": 0.95,
        "eye_r_open": 0.95,
        "eye_x": 0.0,
        "angle_y": -2,
    },
    "surprise": {
        "mouth_form": 0.1,
        "mouth_open": 0.55,
        "eye_l_open": 1.0,
        "eye_r_open": 1.0,
        "eye_y": -0.25,
        "angle_y": -8,
    },
    "shy": {
        "mouth_form": 0.4,
        "mouth_open": 0.05,
        "eye_l_open": 0.65,
        "eye_r_open": 0.65,
        "eye_y": -0.25,
        "angle_y": 4,
        "angle_z": 4,
    },
    "sleepy": {
        "eye_l_open": 0.35,
        "eye_r_open": 0.35,
        "mouth_open": 0.1,
        "mouth_form": 0.05,
        "angle_y": 5,
    },
    "disgust": {
        "mouth_form": -0.25,
        "mouth_open": 0.1,
        "eye_l_open": 0.6,
        "eye_r_open": 0.6,
        "eye_y": 0.25,
    },
    "neutral": {},

    # ---- 眼部类 ----
    "blink": {
        "eye_l_open": 0.0,
        "eye_r_open": 0.0,
    },
    "close_eyes": {
        "eye_l_open": 0.0,
        "eye_r_open": 0.0,
    },
    "wink": {
        "eye_l_open": 0.0,
        "eye_r_open": 1.0,
        "mouth_form": 0.6,
        "angle_z": 4,
    },

    # ---- 身体类 ----
    "lean_left": {
        "body_x": -10,
        "angle_x": -12,
        "angle_z": 4,
    },
    "lean_right": {
        "body_x": 10,
        "angle_x": 12,
        "angle_z": -4,
    },
    "nod": {
        "angle_y": 8,
    },
    "tilt": {
        "angle_z": 12,
        "eye_y": -0.15,
        "mouth_form": 0.35,
    },

    # ---- 嘴部类 ----
    "talk": {
        "mouth_open": 0.45,
        "mouth_form": 0.5,
    },
}


# =============================================================================
# 内部工具函数
# =============================================================================

def _apply_intensity(params: dict, intensity: float) -> dict:
    """对参数应用强度缩放。

    仅对角度、身体幅度和嘴巴张开度进行缩放；
    眼部开合和嘴型形态保持原始值（不适合线性缩放）。
    """
    scalable = {
        "angle_x", "angle_y", "angle_z",
        "body_x", "body_y", "body_z",
        "mouth_open",
    }
    scaled = {}
    for key, value in params.items():
        lo, hi = _PARAM_RANGES[key]
        if key in scalable:
            scaled[key] = max(lo, min(hi, value * intensity))
        else:
            scaled[key] = max(lo, min(hi, value))
    return scaled


def _clamp_params(params: dict) -> dict:
    """将所有参数裁剪到有效范围内。"""
    clamped = {}
    for key, value in params.items():
        lo, hi = _PARAM_RANGES[key]
        clamped[key] = max(lo, min(hi, value))
    return clamped


def _build_override_dict(params: dict) -> dict:
    """将内部参数名映射为 VTube Studio 的参数 ID。"""
    return {
        "Sentia_AngleX": params.get("angle_x", 0.0),
        "Sentia_AngleY": params.get("angle_y", 0.0),
        "Sentia_AngleZ": params.get("angle_z", 0.0),
        "Sentia_BodyX": params.get("body_x", 0.0),
        "Sentia_BodyY": params.get("body_y", 0.0),
        "Sentia_EyeX": params.get("eye_x", 0.0),
        "Sentia_EyeY": params.get("eye_y", 0.0),
        "Sentia_EyeLOpen": params.get("eye_l_open", 1.0),
        "Sentia_EyeROpen": params.get("eye_r_open", 1.0),
        "Sentia_MouthOpenY": params.get("mouth_open", 0.0),
        "Sentia_MouthForm": params.get("mouth_form", 0.5),
    }


# =============================================================================
# 表情解析辅助
# =============================================================================

_KNOWN_ACTIONS = {
    "smile", "happy", "laugh", "sad", "cry", "angry",
    "surprise", "shy", "sleepy", "disgust", "neutral",
    "blink", "close_eyes", "wink",
    "lean_left", "lean_right", "nod", "tilt",
    "talk",
}


def parse_action(text: str) -> str:
    """从 emotion_agent 的输出中提取动作名称，失败则返回 'smile'。"""
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        word = word.strip()
        if word in _KNOWN_ACTIONS:
            return word
    return "smile"


# =============================================================================
# 对外接口
# =============================================================================

def set_vts_controller(vts: VTSController) -> None:
    """注入已连接并认证完成的 VTSController 实例。"""
    global _vts
    _vts = vts


def express_emotion(action: str, duration: float = 5.0, intensity: float = 1.0) -> ToolResponse:
    """设置虚拟形象的表情和动作姿态。

    通过关键词匹配预设姿态库，自动配置 VTube Studio 的身体参数。
    Agent 侧只需传入 action 名称即可触发对应表情/动作。

    Args:
        action: 动作/表情名称。支持的名称：
            - 表情类: smile(微笑), happy(开心), laugh(大笑), sad(难过),
                      cry(哭泣), angry(生气), surprise(惊讶), shy(害羞),
                      sleepy(困倦), disgust(嫌弃), neutral(默认)
            - 眼部类: blink(眨眼), close_eyes(闭眼), wink(单眼wink)
            - 身体类: lean_left(左前倾), lean_right(右前倾),
                      nod(点头), tilt(歪头/疑惑)
            - 嘴部类: talk(说话嘴型)
        duration: 动作持续时间（秒），默认 5.0
        intensity: 动作强度倍率(0.0~1.5)，默认 1.0。
                   仅影响角度和身体幅度，不影响眼部开合和嘴型。

    Returns:
        动作执行结果描述
    """
    # 异常处理：VTSController 未连接或已断开
    global _vts
    if _vts is None or not _vts.is_alive:
        return ToolResponse(
            # content=[TextBlock(type="text", text="VTS 未连接，请先调用 set_vts_controller() 注入已连接的 VTSController")]
            content=[TextBlock(type="text", text="VTS未连接")]
        )
    # 异常处理：不支持的动作名称
    action = action.lower().strip()
    if action not in _ACTION_PRESETS:
        supported = ", ".join(sorted(_ACTION_PRESETS.keys()))
        return ToolResponse(
            content=[TextBlock(type="text", text=f"不支持的 action '{action}'。支持的动作: {supported}")]
        )
    # 异常处理：兜底正则匹配参数
    action = parse_action(action)

    # 获取预设 → 强度缩放 → 裁剪边界
    preset = _ACTION_PRESETS[action]
    params = _apply_intensity(dict(preset), intensity)
    params = _clamp_params(params)

    # 写入覆盖队列，由 VTSController._procedural_soul_loop 统一发送
    _vts._override_params = _build_override_dict(params)
    _vts._override_expiry = time.perf_counter() + max(0.5, duration)

    # 生成描述文本
    parts = [f"🎭 已设置姿态: {action}"]
    if params:
        detail = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(params.items()) if v != 0.0
        )
        if detail:
            parts.append(f"({detail})")
    parts.append(f"持续 {duration:.1f}s")
    return ToolResponse(
        content=[TextBlock(type="text", text=" | ".join(parts))]
    )


def control_body(
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
    body_x: float = 0.0,
    body_y: float = 0.0,
    eye_x: float = 0.0,
    eye_y: float = 0.0,
    eye_l_open: float = 1.0,
    eye_r_open: float = 1.0,
    mouth_open: float = 0.0,
    mouth_form: float = 0.5,
) -> ToolResponse:
    """底层通用身体控制接口。直接设置所有身体参数。

    如需使用预设表情，推荐使用 express_emotion() 函数。

    Args:
        angle_x: 头部左右旋转（Sentia_AngleX）。范围 [-30, 30]。
        angle_y: 头部上下俯仰（Sentia_AngleY）。范围 [-30, 30]。
        angle_z: 头部前后翻滚（Sentia_AngleZ）。范围 [-30, 30]。
        body_x: 身体左右摆动（Sentia_BodyX）。范围 [-30, 30]。
        body_y: 身体前后倾斜（Sentia_BodyY）。范围 [-30, 30]。
        eye_x: 眼球水平位置（Sentia_EyeX）。范围 [-1.0, 1.0]。
        eye_y: 眼球垂直位置（Sentia_EyeY）。范围 [-1.0, 1.0]。
        eye_l_open: 左眼开合（Sentia_EyeLOpen）。范围 [0.0, 1.0]。
        eye_r_open: 右眼开合（Sentia_EyeROpen）。范围 [0.0, 1.0]。
        mouth_open: 嘴巴张开程度（Sentia_MouthOpenY）。范围 [0.0, 0.9]。
        mouth_form: 嘴型表情形态（Sentia_MouthForm）。范围 [-0.8, 1.0]。
    """
    global _vts
    if _vts is None or not _vts.is_alive:
        return ToolResponse(
            content=[TextBlock(type="text", text="VTS 未连接，请先调用 set_vts_controller() 注入已连接的 VTSController")]
        )

    # 边界裁剪
    angle_x = max(-30.0, min(30.0, angle_x))
    angle_y = max(-30.0, min(30.0, angle_y))
    angle_z = max(-30.0, min(30.0, angle_z))
    body_x = max(-30.0, min(30.0, body_x))
    body_y = max(-30.0, min(30.0, body_y))
    eye_x = max(-1.0, min(1.0, eye_x))
    eye_y = max(-1.0, min(1.0, eye_y))
    eye_l_open = max(0.0, min(1.0, eye_l_open))
    eye_r_open = max(0.0, min(1.0, eye_r_open))
    mouth_open = max(0.0, min(0.9, mouth_open))
    mouth_form = max(-0.8, min(1.0, mouth_form))

    _vts._override_params = {
        "Sentia_AngleX": angle_x,
        "Sentia_AngleY": angle_y,
        "Sentia_AngleZ": angle_z,
        "Sentia_BodyX": body_x,
        "Sentia_BodyY": body_y,
        "Sentia_EyeX": eye_x,
        "Sentia_EyeY": eye_y,
        "Sentia_EyeLOpen": eye_l_open,
        "Sentia_EyeROpen": eye_r_open,
        "Sentia_MouthOpenY": mouth_open,
        "Sentia_MouthForm": mouth_form,
    }
    _vts._override_expiry = time.perf_counter() + 2.0

    return ToolResponse(
        content=[TextBlock(type="text", text=(
            f"已控制身体: "
            f"头部=({angle_x:.1f}, {angle_y:.1f}, {angle_z:.1f}), "
            f"身体=({body_x:.1f}, {body_y:.1f}), "
            f"眼球=({eye_x:.1f}, {eye_y:.1f}), "
            f"眼睛开合=({eye_l_open:.1f}, {eye_r_open:.1f}), "
            f"嘴巴=({mouth_open:.1f}, {mouth_form:.1f})"
        ))]
    )
