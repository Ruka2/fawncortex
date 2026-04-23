# VTube Studio 通信接口分析与参数映射指南

> 基于 `components/vts_controller.py` 代码分析

---

## 1. 整体架构

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│   Python 程序   │ ◄────────────────► │  VTube Studio    │
│  (vts_controller)│    JSON API       │   (Live2D 渲染)   │
└─────────────────┘                    └──────────────────┘
         │                                      │
         │  1. 创建自定义参数                      │
         │  2. 注入参数数值  ────────────────────►│  3. 驱动 Live2D 模型
         │                                      │
    [Idle 动画算法]                        [ParamAngleX]
    [语音口型同步]                         [ParamEyeBallX]
    [情感表达系统]                         [ParamMouthOpenY]
```

**通信协议**：VTube Studio 公开了一套基于 **WebSocket** 的 JSON API，外部程序通过发送特定格式的 JSON 消息来控制模型。

**核心库**：代码中使用的是社区封装的 `pyvts` 库，它简化了 WebSocket 连接、认证和消息构造的过程。

---

## 2. 连接与认证流程

```python
async def connect_and_auth(self):
    await self.vts.connect()                    # 1. 建立 WebSocket 连接
    await self.vts.request_authenticate_token()  # 2. 请求认证 Token
    await self.vts.request_authenticate()        # 3. 用 Token 完成认证
```

**认证 Token 持久化**：
- Token 保存在 `./vts_token.txt`
- 首次连接需要用户在 VTube Studio 中点击确认授权
- 之后 Token 可重复使用，无需再次确认

---

## 3. 自定义参数创建

代码在连接成功后，会向 VTube Studio **注册一组自定义参数**：

```python
self.custom_params = [
    "Sentia_AngleX", "Sentia_AngleY", "Sentia_AngleZ",   # 头部旋转 (俯仰/偏航/翻滚)
    "Sentia_EyeX", "Sentia_EyeY",                          # 眼球视线 (水平/垂直)
    "Sentia_EyeLOpen", "Sentia_EyeROpen",                  # 左右眼开合
    "Sentia_BrowLY", "Sentia_BrowRY",                      # 左右眉毛高度
    "Sentia_BrowLForm", "Sentia_BrowRForm",                # 左右眉毛形态
    "Sentia_BodyX", "Sentia_BodyY", "Sentia_BodyZ",        # 身体旋转
    "Sentia_MouthOpenY", "Sentia_MouthForm"                # 嘴巴开合 / 嘴型
]
```

**创建参数的请求格式**：

```json
{
    "apiName": "VTubeStudioPublicAPI",
    "apiVersion": "1.0",
    "requestID": "ParamInit",
    "messageType": "ParameterCreationRequest",
    "data": {
        "parameterName": "Sentia_AngleX",
        "explanation": "Sentia AI Parameter",
        "min": -30.0,
        "max": 30.0,
        "defaultValue": 0.0
    }
}
```

> ⚠️ **关键**：参数创建后，还需要在 VTube Studio 中将它们**绑定到 Live2D 模型的实际参数**，否则数值注入后不会驱动模型动作（见第 5 节）。

---

## 4. 参数注入（实时驱动）

代码通过一个 **60fps 的异步循环** (`_procedural_soul_loop`) 持续计算动画数值，并通过 `InjectParameterDataRequest` 发送到 VTube Studio。

**注入消息的 JSON 格式**：

```json
{
    "apiName": "VTubeStudioPublicAPI",
    "apiVersion": "1.0",
    "requestID": "InjectParams",
    "messageType": "InjectParameterDataRequest",
    "data": {
        "faceFound": true,
        "parameterValues": [
            {"id": "Sentia_AngleX", "value": 5.2},
            {"id": "Sentia_AngleY", "value": -3.1},
            {"id": "Sentia_MouthOpenY", "value": 0.6},
            ...
        ]
    }
}
```

**当前实际注入的参数**（代码中 `inject_data` 部分）：

| 注入参数 | 数值来源 | 作用 |
|---------|---------|------|
| `Sentia_AngleX` | `cur_head_x` | 头部左右旋转 |
| `Sentia_AngleY` | `cur_head_y` | 头部上下俯仰 |
| `Sentia_AngleZ` | `cur_head_z` | 头部前后翻滚 |
| `Sentia_BodyX` | `cur_body_x` | 身体左右摆动 |
| `Sentia_BodyY` | `cur_body_y` | 身体前后倾斜 |
| `Sentia_EyeX` | `cur_eye_x` | 眼球左右移动 |
| `Sentia_EyeY` | `cur_eye_y` | 眼球上下移动 |
| `Sentia_EyeLOpen` | `eye_open` | 左眼开合（0~1） |
| `Sentia_EyeROpen` | `eye_open` | 右眼开合（0~1） |
| `Sentia_MouthOpenY` | `cur_mouth_open` | 嘴巴张开程度 |
| `Sentia_MouthForm` | `cur_mouth_form` | 嘴巴形态（-1~1） |

> **注意**：`Sentia_BrowLY`, `Sentia_BrowRY`, `Sentia_BrowLForm`, `Sentia_BrowRForm`, `Sentia_BodyZ` 虽然在创建时注册了，但当前注入逻辑中没有发送它们（代码中注释了或未包含在 `inject_data` 里）。如果需要眉毛动画，需要补充到注入列表。

---

## 5. 如何在 VTube Studio 中绑定参数（关键步骤）

### 5.1 准备工作

1. 启动 VTube Studio
2. 加载你的 Live2D 模型
3. 确保模型已解锁编辑（模型设置 → 允许编辑）

### 5.2 绑定自定义参数到 Live2D 参数

**步骤 1：打开参数面板**
- 进入 VTube Studio 的 **"模型设置"** → **"参数"** 标签页
- 或者点击界面上的 **"参数"** 按钮

**步骤 2：找到 Live2D 原生参数**
Live2D 模型自带的标准参数名称通常如下：

| Live2D 参数名 | 中文说明 |
|-------------|---------|
| `ParamAngleX` | 头部左右角度 |
| `ParamAngleY` | 头部上下角度 |
| `ParamAngleZ` | 头部旋转角度 |
| `ParamEyeBallX` | 眼球左右 |
| `ParamEyeBallY` | 眼球上下 |
| `ParamEyeLOpen` | 左眼开合 |
| `ParamEyeROpen` | 右眼开合 |
| `ParamBrowLY` | 左眉上下 |
| `ParamBrowRY` | 右眉上下 |
| `ParamBrowLAngle` / `ParamBrowLForm` | 左眉角度/形态 |
| `ParamBrowRAngle` / `ParamBrowRForm` | 右眉角度/形态 |
| `ParamBodyAngleX` | 身体左右 |
| `ParamBodyAngleY` | 身体前后 |
| `ParamBodyAngleZ` | 身体旋转 |
| `ParamMouthOpenY` | 嘴巴张开 |
| `ParamMouthForm` | 嘴型（あいうえお） |

**步骤 3：创建绑定**

对于每个需要驱动的参数，执行以下操作：

1. 在参数列表中找到对应的 Live2D 参数（如 `ParamAngleX`）
2. 点击该参数，展开**"输入"**或**"绑定"**区域
3. 选择 **"添加输入"** → **"自定义参数"**
4. 在下拉列表中选择代码创建的参数，如 `Sentia_AngleX`
5. 设置映射范围（通常保持默认 1:1 即可）

**推荐绑定映射表**：

| VTube Studio 自定义参数 | Live2D 参数 | 建议权重 |
|----------------------|------------|---------|
| `Sentia_AngleX` | `ParamAngleX` | 1.0 |
| `Sentia_AngleY` | `ParamAngleY` | 1.0 |
| `Sentia_AngleZ` | `ParamAngleZ` | 1.0 |
| `Sentia_EyeX` | `ParamEyeBallX` | 1.0 |
| `Sentia_EyeY` | `ParamEyeBallY` | 1.0 |
| `Sentia_EyeLOpen` | `ParamEyeLOpen` | 1.0 |
| `Sentia_EyeROpen` | `ParamEyeROpen` | 1.0 |
| `Sentia_BrowLY` | `ParamBrowLY` | 1.0 |
| `Sentia_BrowRY` | `ParamBrowRY` | 1.0 |
| `Sentia_BodyX` | `ParamBodyAngleX` | 1.0 |
| `Sentia_BodyY` | `ParamBodyAngleY` | 1.0 |
| `Sentia_MouthOpenY` | `ParamMouthOpenY` | 1.0 |
| `Sentia_MouthForm` | `ParamMouthForm` | 1.0 |

> **提示**：绑定后建议点击 **"测试"** 按钮，确认参数变化时模型有反应。

### 5.3 保存配置

绑定完成后，VTube Studio 会自动保存。下次启动时绑定关系依然有效。

---

## 6. 动画系统详解

### 6.1 Idle 动画循环 (`_procedural_soul_loop`)

一个 **60fps** 的实时算法动画系统，让模型在静止时也有生命力：

```
注意力系统 ──► 注视点目标 (focus_target_x, focus_target_y)
      │
      ▼
眼球追踪 ──► 眼球看向注视点 + 有机微抖动
      │
      ▼
头部跟随 ──► 头部转向眼球方向 + 呼吸起伏
      │
      ▼
身体联动 ──► 身体跟随头部 + 呼吸动画
      │
      ▼
眨眼系统 ──► 随机间隔眨眼 + 物理插值
      │
      ▼
眉毛微动 ──► 基于头部姿态和眼睛状态
      │
      ▼
WebSocket 注入 ──► 所有参数值发送到 VTube Studio
```

**核心算法**：
- `_smooth_damp`：指数平滑阻尼，让动作自然过渡（类似 Unity 的 SmoothDamp）
- `_organic_noise`：多层正弦波叠加的有机噪声，模拟生物的自然抖动
- 注意力刷新：每 1.5~4.5 秒随机换一个注视点

### 6.2 语音口型同步 (`animate_speech_lip_sync`)

```python
# 根据音频长度，以 60fps 驱动嘴巴
for i in range(total_chunks):
    # 正弦波模拟口型开合
    target_open = (math.sin(theta) * 0.5 + 0.5) * 0.8
    target_form = math.cos(theta) * 0.9
```

**当前实现特点**：
- 基于音频时长生成正弦波动画（几何感较强）
- 不是真正的音频振幅分析（如 RMS 能量）
- 如果需要更自然的口型，可以接入真正的音频特征提取

---

## 7. 扩展建议

### 7.1 接入真实口型

将 `animate_speech_lip_sync` 中的正弦波替换为基于音频 RMS 能量的分析：

```python
import numpy as np

# 计算音频块的 RMS 能量
rms = np.sqrt(np.mean(audio_chunk**2))
# 映射到 mouth_open 范围
target_open = min(1.0, rms * 3.0)
```

### 7.2 接入情感系统

可以新增自定义参数（如 `Sentia_EmotionJoy`, `Sentia_EmotionSad`），并通过 LLM 的情感分析结果来驱动：

```python
# 根据 LLM 回复的情感标签调整动画
if emotion == "Happy":
    cur_brow_y += 0.3  # 眉毛上扬
    cur_mouth_form = 0.8  # 微笑
```

### 7.3 补充未注入的参数

当前 `inject_data` 中缺少 `Sentia_BrowLY/RY` 和 `Sentia_BodyZ`，如果需要完整的眉毛和身体动画，需要在 `_procedural_soul_loop` 末尾的 `inject_data` 中补充：

```python
{"id": "Sentia_BrowLY", "value": self.cur_brow_y},
{"id": "Sentia_BrowRY", "value": self.cur_brow_y},
{"id": "Sentia_BodyZ", "value": self.cur_body_z},
```

---

## 8. 快速检查清单

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | 启动 VTube Studio 并加载模型 | ☐ |
| 2 | 运行 Python 程序完成连接和认证 | ☐ |
| 3 | 检查 VTube Studio → 参数面板，确认自定义参数已出现 | ☐ |
| 4 | 将每个 `Sentia_xxx` 参数绑定到对应的 Live2D 参数 | ☐ |
| 5 | 点击"测试"验证绑定生效 | ☐ |
| 6 | 观察模型是否有 Idle 动画（头部微动、眨眼） | ☐ |
| 7 | 调用 `animate_speech_lip_sync` 测试口型 | ☐ |

---

## 参考资源

- [VTube Studio 官方 API 文档](https://github.com/DenchiSoft/VTubeStudio)
- [pyvts 库 GitHub](https://github.com/Genteki/pyvts)
- Live2D 参数命名规范：`ParamAngleX`, `ParamEyeBallX`, `ParamMouthOpenY` 等
