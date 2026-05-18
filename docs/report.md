## FawnCortex技术报告

## 1. 摘要

### 1.1 研究背景

随着25年强化学习框架成熟化和社区化后，强化学习成为了大模型后训练的通用解法之一，同时25年下半年越来越多围绕多模态、语音模态的端到端且全双工交互的模型也被各大研究学者开源推出。然而，作为工业界中，半双工仍然是工程化目前较佳的实现方案之一，其可控性和可解释性仍然是工程设计上首要注重的目标。

目前基于LLM的对话助手产品已伴随着智能体框架的完善，社区和解决方案已经迈向非常成熟的地步，然而，这些对话助手产品往往缺少讨论低延迟、高互动性的设计，这些产品成型后往往会有较大的响应延迟，例如深度思考、或ReAct、Plan-and-Execute架构所需的计算延迟与实时对话所要求的低响应延迟之间的冲突。对于工程领域，半双工的级联对话系统仍然是一个较少工程架构涉及的领域。

### 1.2 核心问题

本研究针对以下技术问题展开：

1. **延迟与质量的权衡**：如何在保证深度推理质量的前提下，将对话响应延迟控制在用户可接受的范围（如< 2s）内。
2. **上下文一致性**：当多个智能体并行/异步执行时，如何维护统一的对话上下文。
3. **在对话流程中反思**：设计一种机制，使得对话内容能够在事后反思。

### 1.3 研究范围

本项目的研究范围聚焦于**对话交互场景下的上下文管理管线设计**。具体包括：

- 多智能体角色分工与协作模式（ChatAgent、BrainAgent、EmotionAgent、ReflectionAgent）
- 前后台解耦的异步推理管线
- 大脑智能体的增量式思考过程注入对话机制
- 表情、对话输出调度

*备注：项目不讨论端到端的语音交互建模（ASR+VAD+TTS 的全链路联合优化），也不涉及记忆存储的底层内核设计（记忆智能体的实现方式是通过LLM推理召回关键词、以及每轮对话总结应该有哪些记忆被存储）*

### 1.4 主要贡献

本项目的主要技术贡献包括：

1. **"快前台 + 深后台"的异步线程**：提出快速响应（System 1）与深度推理（System 2）解耦的并行流水线设计，其可使得对话响应不阻塞后台模型思考，从根本上消除级联延迟累积。

2. **增量思维截断与注入机制**：通过流式输出截取和增量上下文同步的设计，实现将后台流式推理过程的"中间思考"提前加入到对话上下文中，使得前台对话响应可以伴随着思考过程不停追答回复，使用户**在对话交互中逐渐获得**有价值的回复。

3. **反思控制**：引入 ReflectionAgent 为每一轮追答和总结都进行质量过滤，避免因为禁用深思考的前台模型因为产出低质量信息而干扰对话流畅性。

4. **可观测的上下文管线**：提供基础的 Web UI 上下文监控面板，将 Agent 内部状态（ReAct 轮次、思考快照、工具调用）实时可视化，为对话系统的调试和优化提供基础设施。

---

## 2. 项目架构设计

### 2.1 架构管线设计

FawnCortex 采用**分层解耦的管道-过滤器（Pipe-and-Filter）架构**，整体信息流可概括为：

```
用户输入 → EventBus → [前台并行轨道] + [后台推理轨道] → OutputScheduler → TTS/VTS输出
                        ↓                ↓
                   ChatAgent +      BrainAgent (ReAct)
                   EmotionAgent          ↓
                        ↓           midway_watcher
                     快速响应        (增量思维注入)
                        ↓                ↓
                   OutputScheduler ←── brain_summary
```

#### 2.1.1 智能体角色定义

项目依赖于基座模型的推理范式，必须是受thinking推理范式的模型才适配本项目框架。

系统包含四个核心智能体，各司其职：

| 智能体 | 职责能力 | LLM 配置策略 | 推理模式 |
|--------|------|-------------|---------|
| **ChatAgent** | 前台对话生成，接收到用户消息后直接<u>基于上下文</u>和<u>模型内部权重</u>直接生成语言响应 | 轻量模型，thinking 关闭 | 单步调用 |
| **EmotionAgent** | 生成表情动作指令与口吻控制，用于驱动VTS动画或TTS模型的提示词参考 | 轻量模型，thinking 关闭 | 单步调用 |
| **BrainAgent** | 后台慢思考推理，工具调用，记忆检索&存储 | 强模型，thinking 开启 | ReAct 循环 |
| **ReflectionAgent** | 响应质量评估，对话干预决策 | 轻量模型，thinking 关闭 | 分类器 |

关于ChatAgent如何获取大脑智能体的上下文：

- 可以通俗为拷贝模式，一旦大脑智能体有流式生成的中间输出，就将粘贴到对话智能体的上下文中
- 即对话智能体一直受大脑智能体影响着，故达到对话生成指导的效果



#### 2.1.2 前台并行管道（FrontStagePipeline）

前台并行管道将由 ChatAgent 与 EmotionAgent 这两位智能体进行**并行**处理：

- 两者同时启动，互不阻塞
- 理论上是多个并行线程LLM推理
- 两者均完成后，通过 `OutputScheduler.schedule()` 统一调度，确保文本与表情/语气的匹配

这一设计将前台响应严格约束为**两个独立的 LLM 调用**，不引入任何 AgentLoop（由本项目封装AgentScope的Agent类实现单步调用），从而保证前台响应的较低延迟性（通常在 1-3 秒内）。



#### 2.1.3 后台推理轨道（BackgroundBrainAgent）

BrainAgent 被包装为后台常驻任务（`BackgroundBrainAgent`），通过 EventBus 异步接收用户输入：

- 每轮用户输入触发一个新的 `asyncio.Task` 执行 `brain.think()`
- 新输入到达时自动取消上一轮未完成的思考（cancel + 清理），但不会清空BrainAgent的短期记忆
- 思考结果通过 `asyncio.Queue` 输出，供主循环消费

BrainAgent 内部基于 AgentScope 的 `ReActAgent` 实现，支持多轮 reasoning-acting 循环，可调用记忆检索、网络搜索、论文检索等工具。关键的**流式截取机制**通过 monkey-patch `ReActAgent._reasoning` 和 `print()` 方法实现，在不修改底层AgentScope框架的前提下捕获增量推理文本。此处建议学习 [AgentScope 智能体钩子函数](https://doc.agentscope.io/zh_CN/tutorial/task_hook.html#)来进行理解。



#### 2.1.4 中期汇报机制（Midway Watcher）

这是连接前后台的核心创新点。`midway_watcher` 在 BrainAgent 思考期间以 1 秒为周期轮询：

**触发中间汇报的条件**（以下条件必须同时满足）：

1. 如果 BrainAgent 状态为 `thinking`
2. 已超出动态阈值（默认基于 `BRAIN_CUT_TIME_DURATION`，可配置）
3. BrainAgent 已流式生成出足够推理所使用的内容（> 100 字符）
4. 最大介入中间汇报的次数未达上限（`MAX_MIDWAY_INTERVENTIONS = 12`）

**触发后动作**：

1. 获取 BrainAgent 自上次同步以来的增量推理内容（`get_stream_buffer_delta()` / `get_new_reasonings_since_last_sync()`）
2. 以 `[系统提示]` + `[系统思考]` 的标记格式注入 ChatAgent 的短期记忆
3. 异步管道建立 ChatAgent 的对话任务（并加入队列），触发 ChatAgent 来生成中间回复
4. ReflectionAgent 对中间回复进行质量判决（`clarify` / `ignore`）
5. 只有ReflectionAgent判断通过的回复才进入 `OutputScheduler` 后进行TTS播报

#### 2.1.5 总结补充机制（Brain Summary）

当 BrainAgent 完成思考（或超时）后，触发 `brain_summary`：

- 采集 midway 截断点之后的增量 thinking 内容
  - 即只采集`<think>...</think>`的内容，`<asnwer>...</answer>`的内容丢弃
  - 受限于LLM模型推理范式，若模型在预训练或后训练无此推理范式，则本项目pipeline可能会失效
  - 设计思路：因为思考过程即大模型一步一步解决的过程，且ReAct过程就是一个将复杂问题拆解为简单问题的过程（且目前RL也非常认可），因此LLM就天然具备拆解信息的能力，故只使用thinking思考过程就可解决对话任务，而无需最后的总结。

- 每一轮 midway 截断后的增量 thinking 内容注入 ChatAgent 的短期上下文
- 由 ChatAgent 将基于上下文的已有token，从而组织为新的一轮追答文本
- ReflectionAgent 再次质量过滤后调度输出

设计的核心理念是：**让 ChatAgent 始终作为唯一声音出口**，BrainAgent 只提供"辅助推理材料生成"，不直接对话，但对话智能体的上下文是一直依赖大脑智能体的，从而保证上下文一致性。



### 2.2 上下文管理

#### 2.2.1 短期记忆：滑动窗口 + 标记过滤

`ShortTermMemory` 实现了带容量上限的短期记忆：

- 固定容量（默认 30 条），超过时自动移除最旧消息
- 可按 `msg_id` 精确删除（用于 ReflectionAgent 否决后的上下文回溯）

#### 2.2.2 长期记忆：mem0 框架

项目采用 mem0 框架实现长期记忆：

- **向量存储**：ChromaDB，本地持久化
- **记忆检索**：BrainAgent 在 ReAct 循环中自动调用 `retrieve_from_memory`
- **记忆写入**：对话内容根据 BrainAgent 的任务决策，使用工具 `record_to_memory` 按有价值的信息存入记忆库

#### 2.2.3 特殊标记

上下文对话以 ChatAgent 为基准，因此上下文引入了两种字符级标记来区分指令来源：

- `[系统提示]`：由系统发出的工作指令，触发 ChatAgent 的行为转换
- `[系统思考]`：由 BrainAgent 产生的推理内容，作为 ChatAgent 的区分角色的知识参考



#### 2.2.4 增量同步与状态指针

BrainAgent 维护多组同步指针，实现精细化上下文管理：

- `_last_midway_sync_iter`：记录已同步到 ChatAgent 的 ReAct 轮次
- `_last_stream_sync_len`：记录流式缓冲区已同步的字符位置
- `_sub_status`：实时子状态机（`idle` / `reasoning` / `acting`）

这组指针确保 midway 和 summary 只传输**增量内容**，避免重复和冗余。



### 2.3 异步编排调度

#### 2.3.1 EventBus：Publish-Subscribe 总线

`EventBus` 是本项目的异步事件基础设施：

- 每个 Agent 拥有专属 `asyncio.Queue`
- 基于 topic 的发布-订阅路由（topic可理解为信息来源 / 角色）
- 不采用 AgentScope 的现有组件 `MsgHub` ，**MsgHub仅能做到信息发布/信息广播，并不支持信息订阅**，没办法做到事件状态一被更新后马上触发下游指令。因此本项目搭建`EventBus`目的是希望通过订阅事件来达成异步线程下的再触发操作。



#### 2.3.2 OutputScheduler：优先级输出队列

项目中所有文本输出和音频输出都由 `OutputScheduler` 统一管理输出：

- 基于 `asyncio.PriorityQueue`，支持 `NORMAL`（普通）和 `HIGH`（插队）优先级
- 同时调度 TTS 语音合成和 VTS 表情动画
- 支持 `interrupt()` 打断：清空正在执行的任务队列 （包括正在播放的tts任务），用于用户新输入到达时的即时响应



#### 2.3.3 异步安全与超时机制

项目在系统中存在在多处实现了超时的逻辑设计：

- 所有任务队列都有 `Task.cancel()` 操作，配合避免无限等待
- BrainAgent 设置 `BRAIN_TIMEOUT = 300s` 的总思考上限，代表大脑智能体总共可以是靠多少时间
- 中期汇报时间监控器/订阅器`midway_watcher` 和对话轮数最后一轮的大脑触发总结 `brain_summary`  均在独立 Task 中执行，可被外部事件中断
- TTS 的线程任务封装在（`run_in_executor`）中，不阻塞其它事件循环

### 

---

## 3. 工程化部署和迭代更新/运维方式

### 3.1 部署架构

FawnCortex 支持两种运行模式：

#### 3.1.1 CLI 文本模式（推荐用于学习/调试）

```bash
python chat_cli.py
```

- 直接在终端进行文本对话
- 完整展示信息管线：前台响应 → 后台推理 → 中途汇报 → 总结补充
- 支持本地 TTS 播放和 VTS 连接（可选）

#### 3.1.2 Web 服务模式（生产/交互使用）

```bash
python server.py
```

- 基于 **FastAPI + WebSocket** 构建
- 提供两个页面：
  - `/`：监控面板（上下文可视化、Agent 状态、延迟统计）
  - `/live`：直播交互页面（支持文本/语音输入、TTS 音频流式播放）
- 前端集成 Silero VAD 实现语音活动检测
- TTS 音频通过 WebSocket 以 base64 PCM 流式传输到前端播放

### 3.2 配置管理

所有配置集中在 `config.py`，支持环境变量覆盖：

- **角色级 LLM 映射**：不同 Agent 可配置不同的模型/API/生成参数（如 ChatAgent 禁用 thinking，BrainAgent 启用 thinking）
- **超时参数**：`BRAIN_TIMEOUT`（最大思考时间）、`BRAIN_CUT_TIME_DURATION`（中途汇报阈值）
- **外部服务**：ASR/TTS API、VTube Studio 端口、Embedding 模型

### 3.3 可观测性

#### 3.3.1 延迟分析（LatencyTracker）

`LatencyTracker` 记录三类指标：

1. **各智能体独立耗时**：ChatAgent、EmotionAgent、BrainAgent、ReflectionAgent 的 LLM 调用时间
2. **端到端时间**：前台轨道总耗时、后台轨道总耗时
3. **用户感知延迟**：输入 → 首次听到语音的时间

#### 3.3.2 日志

- 文件日志：`enable_file_logging()` 将所有输出重定向到 `./logs/run_YYYYMMDD_HHMMSS.log`
- LLM 调试日志：每个 Agent 的 `print_llm_prompt()` / `print_llm_response()` 完整打印输入输出
- Token 统计：每轮记录各 Agent 的 input/output tokens 和 LLM 调用次数

#### 3.3.3 Web UI 实时监控

Web 模式提供多种事件类型的实时可视化，可供性能评测或后续扩展使用：

- `brain_snapshot`：ReAct 循环状态、工具调用、流式缓冲区内容
- `midway_message` / `brain_summary`：中间汇报和总结事件
- `reflection_judgment`：ReflectionAgent 的判决结果
- `chat_context`：ChatAgent 当前短期记忆的完整内容



---

## 4. 趋势和挑战

### 4.1 端到端对话系统展望

当前 FawnCortex 仍基于**半双工、级联式**的对话架构，前沿研究正朝着端到端模型演进：

- **全双工语音模型**（如 GPT-4o Realtime、MiniCPM-o）：将 ASR、LLM、TTS 压缩为单一模型，实现真正的流式对话
- **Turn-taking 建模**：模型自身学习何时应该倾听、何时应该打断、何时应该停顿，目前该研究已非常成熟。

因此，本项目所采用的级联半双工的对话交互方式，我相信未来有一天会被端到端模型所容纳，故讨论如何完全控制对话策略、在对话中博弈网络、多模态人机交互会是未来最有价值讨论的课题。

**挑战**：



### 4.2 反思反馈瓶颈

目前反思智能体是一个轻量化的分类器设计，仅能作为是否输出该句回答给用户进行判断，此设计是非常容易被攻击的：

- 反思器完全不参与对话链路，不参与任何信息流的循环，性能没办法改善或动态调整。
- 反思器无经验参考、未涉及性能优化设计都使得这是一个极具危险和难以实验调整的设计，需要任务定义。
- 反思器回溯对话智能体的上下文后，并没有讨论回溯后怎么触发对话，会导致追答时遗漏某一环节。
- 未加入动态截取思考过程的设计，会出现上下文信息分布位置有的非常庞大、有的非常少
- 未加入动态思考时间阈值的设计，即TTS总时长的时间也是一个可加入反思机制的设计，以此让智能体了解到什么时候应该开始追答思考
- 表情智能体并未加入到反思智能体的信息流中，即实际智能体并不知道什么时候自己做过了表情

简而言之，若强调高交互性且完全可控的对话助手，反思这一个环节是一个有挑战性的研究课题，一方面他会被AgentLoop / 信息循环的响应延时所影响，另一方面将反思机制做太轻会使得对话流程机械化。



### 4.3 性能评测

[TODO:]

---



## 5. 结论

FawnCortex 提出并实践了一种**面向低延迟对话交互的多智能体协作架构**。其核心方法论可归纳为：

1. **推理解耦**：将"快速响应"与"深度推理"分配到独立的执行轨道，从根本上消除级联延迟
2. **增量注入**：通过流式截取和动态阈值机制，使推理过程的中间产物能够实时补充到对话中

该项目更多讨论着一个**对话系统上下文管线的参考实现**，其管道设计、异步编排模式希望能为从业者或研究人员提供可讨论分享的机会。

