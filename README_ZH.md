# FawnCortex

> 一个由多智能体协作的 AI 对话系统，主要解决级联系统下的实时对话挑战。

FawnCortex （像小鹿一样思考） 是一个解决对话任务的智能体应用，采用**前台快速响应 + 后台深度思考**的管道设计，通过设定专用于对话任务的子智能体（对话智能体、表情智能体、大脑中枢、反思判断）来进行协同工作，通过截断增量大脑中枢的思考过程，以此来实现低延迟对话与高质量推理的平衡。

项目内使用 [AgentScope](https://github.com/agentscope-ai/agentscope) 框架来搭载基础的智能体工作流，同时项目也内置长期记忆、联网搜索、论文检索等智能体相关实现。对于语音交互而言，语音识别和语音合成目前采用 API 调用方式实现，因此需要对本项目进行模型选型时需要对调用/推理进行一定的改造。

因此，本质上，项目仍然是聚焦在对话交互时的上下文管理所进行讨论，项目方法论贡献在于上下文管道的参考，或是相关语音交互项目的学习分享，欢迎讨论分享。

---



## 项目特点

- **智能体协作架构**
  - **ChatAgent**：前台对话智能体，快速生成面向用户的自然语言回复
  - **EmotionAgent**：表情与语气控制器，驱动虚拟形象动作与语音风格
  - **BrainAgent**：后台大脑中枢，基于 ReAct 模式进行深度推理与工具调用，并将<u>增量</u>思考过程加入到对话智能体
  - **ReflectionAgent**：反思判断器，用于每一轮对话智能体评估回复质量，判断是否触发对话
- **异步编排 Pipeline**
  - 前台并行运行 ChatAgent + EmotionAgent，保证对话时仅解耦为一次LLM调用，不参入任何AgentLoop
  - 后台异步运行 BrainAgent，基于时间阈值持续将思考过程加入到ChatAgent上下文中，辅助对话智能体生成高价值内容
  - OutputScheduler 输出编排器用于管理输出组件的优先级队列，调度 TTS 语音播报与表情输出
- **交互提升**
  - 项目在表情智能体的下游任务中，增加 Vtube Studio 的live2d形象控制，用于参考使用
- **Web UI 监控上下文**
  - 除主要交互网页 `/live` 外，服务端包含监控页面 `/index` 用于观察和分析对话上下文

---



## 快速开始

### 环境要求

- Python >= 3.11
- （可选）VTube Studio（如需虚拟形象联动）

### 1. 快速启动

```bash
git clone <仓库地址>
cd fawncortex
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

项目大量使用 API 和 全局参数配置，需要先通过 `.env` 文件加载 API Key 等配置，项目根目录已提供 `.env` 模板（未上传真实密钥，请自行创建）：

```bash
cp .env.example .env
```

> 配置表内支持为不同智能体配置不同 LLM 模型，以此实现成本与效果权衡。

### 3. 运行项目

#### 模式一：纯文本命令行（推荐学习管道）

```bash
python chat_cli.py
```

- 直接在终端进行文字对话**（※推荐用于学习信息流）**
- 完整展示前台响应 → 后台思考 → 中间汇报 → 总结补充 的完整信息管道
- 支持 TTS 语音播报与 VTS 虚拟形象（若已连接 VTube Studio）

#### 模式二：Web 服务

```bash
python server.py
```

访问：

- 监控面板：`http://localhost:8080/`
- 对话交互页：`http://localhost:8080/live`

Web 服务支持：

- 文字输入与 ASR 语音输入（前端包含 silero_vad VAD）
- TTS 音频实时推流播放
- Agent 内部事件实时可视化（思考快照、表情更新、工具调用等）



## 项目结构

```
.
├── chat_cli.py                   # 纯文本命令行入口（推荐用于学习管道流程）
├── server.py                     # Web 服务端入口
├── web_scheduler.py              # Web 版核心引擎
├── config.py                     # 全局配置文件
├── requirements.txt              # Python 依赖
├── .env                          # 环境变量（API Key 等）
│
├── fawncortex/
│   ├── agent/                    # 智能体定义
│   │   ├── chat_agent.py         # 前台对话智能体
│   │   ├── emotion_agent.py      # 表情/语气智能体
│   │   ├── brain_agent.py        # 后台大脑智能体（ReAct + Tool Calling）
│   │   └── reflection_agent.py   # 反思判断智能体
│   ├── base/                     # 基础组件
│   │   ├── simple_agent.py       # 简易 Agent 基类（封装AgentScope类别）
│   │   └── memory.py             # 长期记忆封装（mem0）
│   ├── pipeline/                 # 管道与调度
│   │   ├── front_stage_pipeline.py   # 前台并行管道
│   │   ├── back_stage_midway.py      # 中间汇报与总结
│   │   ├── event_controller.py       # 事件总线与后台 BrainAgent
│   │   └── output_scheduler.py       # TTS / VTS 输出调度器
│   ├── tools/                    # 工具集（可自行进行新增，需要参考AgentScope框架）
│   │   ├── search_memory.py      # 记忆检索/记录
│   │   ├── online_search.py      # 联网搜索
│   │   ├── paper_search.py       # 论文搜索与阅读
│   │   ├── get_current_time.py   # 获取当前时间
│   │   └── weather.py            # 天气查询
│   ├── components/               # 外部组件适配
│   │   ├── voice/
│   │   │   ├── tts.py            # TTS 语音合成
│   │   │   └── asr.py            # ASR 语音识别
│   │   ├── body/
│   │   │   ├── vts_controller.py     # VTube Studio 控制器
│   │   │   └── emotion_animate.py    # 表情动画映射
│   │   └── webui/static/         # Web 前端静态文件
│   └── logger/                   # 日志与延迟追踪
│
├── data/                         # 运行时数据（数据库、缓存音频）
└── logs/                         # 运行日志
```

> **入门提示**：
> - 想快速理解系统管道和对话上下文的信息流转 → 阅读 `chat_cli.py`
> - 想快速体验对话交互应用 → 运行 `python server.py`
> - `web_scheduler.py` 是 Web 服务的核心引擎，**不直接运行**

---



## 项目瓶颈 & 待办

### 对话交互任务

项目仍然基于半双工以及级联的方式来讨论对话交互任务，并未对端到端对话任务进行任务定义和建模。

### 反思智能体瓶颈

反思智能体不直接作为一个独立的智能体，没有独立的上下文管理机制（依靠对话智能体的对话历史），因此反思智能体目前没有经验反馈（few-shot）或其它提升效果性能（模型微调）的实现，本项目实现目前仅作为分类器使用，需要甄别。

因为本项目低延迟的核心初衷，对话交互时的反思推理不应该占用对话主链路，而增加AgentLoop或反馈到子智能体的设计会大幅度增加首字延迟（同理，VTS身体动作执行成功后也没有信息管道容许智能体知道自己做出了行为/动作），故反思智能体是本项目对话链路中最关键但却又最薄弱的机制。

### 多智能体局限

本项目并不直接讨论**记忆智能体**和**ASR+VAD+TTS管道的实践**，原因在于大脑智能体在AgentLoop中已经实现了记忆智能体的检索和存储的功能，并且本项目直接使用[mem0](https://github.com/mem0ai/mem0)记忆框架来快速实现记忆功能。对于半双工而言，记忆存储的实现和AgentLoop的内核设计可以参考其它优秀的项目，本项目只讨论对话信息的上下文管理。



## 配置详解

所有配置集中在 `config.py`，支持通过环境变量覆盖。关键配置项说明：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `FAWNCORTEX_PORT` | Web 服务端口 | `8259` |
| `AGENT_NAME` / `USER_NAME` | 智能体与用户的默认名称 | `AgentName` / `UserName` |
| `BRAIN_TIMEOUT` | BrainAgent 最大思考时间（秒）<br />即对话交互中对话容忍大脑智能体的思考时间 | `300` |
| `BRAIN_CUT_TIME_DURATION` | 中间汇报截断阈值（秒）<br />即每轮增量思考过程后，经过多少秒切割发送到对话智能体上触发对话 | `5` |
| `STREAM` | 是否开启 LLM 流式输出<br />必须开启，否则增量思考过程无法触发对话 | `True` |
| `LLM_ROLE_GENERATE_KWARGS` | 角色专属生成参数（如是否启用 thinking）<br />需要基于所使用 LLM 模型的推理范式进行使用，推荐对话智能体禁用深度思考模式 | 按智能体角色配置 |

更多细节请查看 `config.py` 内注释。

---



## License

[MIT](LICENSE)

---



## 感谢

欢迎各位学者、专家来提交 Issue 讨论！
