# 技术报告：BrainAgent × ReflectionAgent 深度联动架构设计

> 报告日期：2026-05-07  
> 关联模块：`deerberry/agent/brain_agent.py`、`deerberry/agent/reflection_agent.py`、`deerberry/pipeline/chatroom_controller.py`、`main7_chatroom.py`  
> 前置文档：`docs/react_intervention_design_report.md`

---

## 一、执行摘要

本报告在 `react_intervention_design_report.md` 的基础上，针对用户提出的 **4 大功能需求**（状态同步、阈值反思限制、上下文隔离、流式截取）进行深度技术方案设计。核心目标是在不破坏 ReActAgent 核心循环的前提下，实现 ReflectionAgent 对 BrainAgent 中间过程的感知与介入。

**关键设计原则**：
1. **旁路介入**：BrainAgent 的 ReAct 循环始终独立运行，中间汇报是旁路，不打断推理
2. **状态透明**：BrainAgent 通过 Hook 暴露细粒度子状态（reasoning / acting），供 ReflectionAgent 实时查询
3. **防御性设计**：定时器判断默认"不汇报"，宁可错过也不打扰用户
4. **上下文隔离**：中间汇报使用临时上下文，汇报结束后销毁，不污染 ChatAgent 长期记忆

---

## 二、需求映射与技术可行性分析

### 2.1 需求总览

| 需求 | 关键问题 | 当前可行性 | 推荐方案 |
|------|---------|-----------|---------|
| **状态同步**（总线 thinking + 子状态 reasoning/acting） | ReActAgent 未暴露子状态 | ✅ 高 | Hook 保存子状态 |
| **阈值反思限制**（动态阈值 + 中间介入） | 如何判断"是否需要介入" | ✅ 高 | 定时器 + 规则驱动 |
| **上下文隔离**（临时上下文 + observe 回 brain） | observe 是否破坏 AgentLoop | ✅ 高 |  system 消息注入 |
| **流式截取**（获取正在生成的内容） | stream 中如何截取中间文本 | ⚠️ 中 | `post_reasoning` 快照 |

### 2.2 AgentScope 框架约束

**已确认的关键事实**（源码级）：

1. **`ReActAgent._reasoning()` 是阻塞调用**：内部的 `async for` 循环不会 yield 控制权给外部任务（除非 `await self.print()` 内部有 sleep，但通常只有 1-3ms）
2. **`post_reasoning` 的 `output` 已是完整内容**：即使是 `stream=True`，`content_chunk.content` 是**完整文本快照**（如 `'好'` → `'好开心'`），`post_reasoning` 触发时 `output` 包含本轮全部生成内容
3. **`_acting()` 也是阻塞调用**：工具执行期间，外部无法获取中间状态
4. **`memory.add()` 是异步的**：在 ReAct 循环中，每轮结束后会 `await self.memory.add(msg)`，将结果存入 memory
5. **`observe()` 本质上就是 `memory.add()`**：`AgentBase.observe()` 的实现与 `memory.add()` 等价

**核心结论**：
- "实时截取正在生成的内容"在框架层面**不可行**（除非 Monkey-patch `_reasoning()`）
- "截取最近一轮已完成的 reasoning/acting 结果"**完全可行**（通过 Hook）
- 这是**务实且足够好**的方案：reasoning 通常 1-3 秒完成，等待 `post_reasoning` 的延迟可接受

---

## 三、状态同步设计（总线 + 子状态）

### 3.1 状态层级定义

```
BrainAgent 状态机（两层）
┌─────────────────────────────────────────────────────────────┐
│  总线状态 (Bus-Level)                                        │
│  ├─ idle      : 未在思考                                     │
│  ├─ thinking  : 正在 ReAct 循环中（由 BackgroundBrainAgent） │
│  └─ completed : 思考完成                                     │
├─────────────────────────────────────────────────────────────┤
│  子状态 (Hook-Level)                                         │
│  ├─ idle      : 未在 reasoning/acting                        │
│  ├─ reasoning : LLM 正在生成思考内容                         │
│  └─ acting    : 正在执行工具调用                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 子状态流转时序

```
user.input → BackgroundBrainAgent._think()
                │
                ▼
            status = "thinking"
                │
                ├── iter 1 ─────────────────────────────┐
                │   │                                   │
                │   ▼                                   │
                │   sub_status = "reasoning"            │
                │   await _reasoning()                  │
                │   │                                   │
                │   ├── post_reasoning hook             │
                │   │   → sub_status = "acting"         │
                │   │   （如果有 tool_use）             │
                │   │                                   │
                │   ├── _acting() 执行中                │
                │   │   sub_status = "acting"           │
                │   │                                   │
                │   ├── post_acting hook                │
                │   │   → sub_status = "idle"           │
                │   │   （回到 iter 顶部）              │
                │   │                                   │
                │   └── 无 tool_use → break             │
                │                                       │
                ├── iter 2 ...                          │
                │                                       │
                └── iter N → break                      │
                            │                           │
                            ▼                           │
                        status = "completed"            │
                        sub_status = "idle"             │
                            │                           │
                            └──→ judge_after_brain()    │
                                                        │
【定时器旁路】──────────────────────────────────────────┘
    定时器每隔 1s 检查：
    - 如果 elapsed > threshold 且 sub_status == "reasoning"
      → 获取 latest_reasoning_text → 旁路汇报
    - 如果 elapsed > threshold 且 sub_status == "acting"
      → 获取 latest_reasoning_text + 工具状态 → 旁路汇报
```

### 3.3 代码实现（BrainAgent Hook 增强）

```python
class BrainAgent:
    def __init__(self, ...):
        ...
        # ── 状态机 ──
        self._sub_status: str = "idle"          # idle | reasoning | acting
        self._latest_reasoning_text: str = ""   # 最新 reasoning 文本快照
        self._latest_tool_name: str = ""        # 最新调用的工具名
        self._has_used_tools: bool = False      # 本轮是否调用过工具
        self._think_start_ts: float = 0.0       # 本轮 think() 开始时间

        async def _hook_post_reasoning(react_self, kwargs, output):
            self._sub_status = "reasoning"  # 实际上 reasoning 已完成，但下一轮前保持
            self._latest_reasoning_text = output.get_text_content() or ""
            tool_uses = output.get_content_blocks("tool_use") if hasattr(output, "get_content_blocks") else []
            if tool_uses:
                self._has_used_tools = True
                self._sub_status = "acting"  # 有 tool_use，即将进入 acting
            else:
                self._sub_status = "idle"    # 无 tool_use，本轮结束
            ...

        async def _hook_post_acting(react_self, kwargs, output):
            self._sub_status = "idle"  # acting 完成，回到 idle（等待下一轮 reasoning）
            tool_call = kwargs.get("tool_call", {})
            self._latest_tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
            ...
```

### 3.4 状态信息作为对话补充

用户需求："子状态信息可以作为对话信息来补充，介入目前这个对话信息我认为是固定一个字符串追加到大脑智能体的思考尾部较佳"

**实现方式**：在 `BrainAgent.think()` 返回的 `insight` 尾部追加状态摘要字符串。

```python
def _build_status_suffix(self) -> str:
    """构建状态摘要字符串，追加到 insight 尾部。"""
    if self._sub_status == "reasoning":
        return f"\n\n[系统状态] 当前正在进行第 {self._current_iter} 轮思考生成中..."
    elif self._sub_status == "acting":
        return f"\n\n[系统状态] 正在调用工具 '{self._latest_tool_name}'，请稍候..."
    return ""

async def think(self, user_msg) -> dict:
    ...
    result = await self.agent.reply(user_msg)
    text = result.get_text_content()
    
    # 追加状态信息
    text += self._build_status_suffix()
    ...
```

**用途**：当 ReflectionAgent 或 ChatAgent 获取 brain 的中间结果时，可以基于状态信息生成更自然的回复（如"正在查论文，请稍候"）。

---

## 四、阈值反思限制设计

### 4.1 介入条件矩阵

| 条件组合 | 介入策略 | 说明 |
|---------|---------|------|
| 无工具调用 + 时间未超 | ❌ 不介入 | 纯 reasoning，等 completed 后判断 |
| 无工具调用 + 时间已超 | ❌ 不介入 | 仍是纯 reasoning，最终判断即可 |
| 有工具调用 + 时间未超 | ❌ 不介入 | 工具执行中，等 acting 完成 |
| 有工具调用 + 时间已超 | ✅ **中间介入** | 工具可能卡住或太慢，需要旁路汇报 |
| 有工具调用 + 已完成 | ✅ 最终判断 | 正常的 completed 判断 |

**核心规则**：
- **无工具调用**：无论时间多长，只在 `completed` 后判断（现有逻辑）
- **有工具调用**：当 `elapsed > threshold` 时，触发中间介入

### 4.2 动态阈值算法

```python
def compute_dynamic_threshold(chat_result: Msg) -> float:
    """根据前台对话长度计算动态阈值。
    
    逻辑：
    - 前台回复越短 → 用户问题越简单 → 容忍时间越短
    - 前台回复越长 → 用户问题越复杂 → 容忍时间越长
    """
    BASE = 5.0          # 基础阈值 5 秒
    chat_text = chat_result.get_text_content() or ""
    token_count = len(chat_text)  # 简化为字符数，后续可替换为真实 token 数
    
    # 每 10 个字符增加 1 秒，上限 30 秒
    extra = token_count / 10.0
    threshold = BASE + extra
    threshold = min(threshold, 30.0)
    
    return threshold
```

**示例**：

| 前台回复长度 | 动态阈值 |
|------------|---------|
| "请稍等"（6 字） | 5.6 秒 |
| "这个请求有点复杂，我先查查看"（15 字） | 6.5 秒 |
| "这个问题涉及多个方面，我需要从论文检索、作者追踪和引用分析三个维度来回答"（40 字） | 9.0 秒 |

### 4.3 计时器设计与中间介入流程

**计时器启动时机**：`front_stage.respond()` 完成后（即 ChatAgent 已回复用户）

```python
# main7_chatroom.py 中的计时器逻辑
async def _midway_watcher(
    brain_bg: BackgroundBrainAgent,
    chat_agent: ChatAgent,
    reflection: ReflectionAgent,
    threshold: float,
    chat_result: Msg,
):
    """中间过程监听器：每隔 1 秒检查 brain 状态。"""
    start_ts = time.perf_counter()
    already_intervened = False  # 冷却期标记
    
    while brain_bg.status == "thinking" and not already_intervened:
        await asyncio.sleep(1.0)
        elapsed = time.perf_counter() - start_ts
        
        if elapsed > threshold:
            # 检查 brain 是否使用了工具
            if not brain_bg.brain._has_used_tools:
                break  # 无工具调用，不介入
            
            # 获取 brain 当前快照
            snapshot = brain_bg.brain.get_react_snapshot()
            latest_reasoning = brain_bg.brain._latest_reasoning_text
            sub_status = brain_bg.brain._sub_status
            
            # 构建中间汇报内容
            if sub_status == "reasoning":
                midway_content = f"正在分析中... {latest_reasoning[:200]}"
            elif sub_status == "acting":
                tool_name = brain_bg.brain._latest_tool_name
                midway_content = f"正在调用 {tool_name} 查询中，请稍候..."
            else:
                midway_content = f"正在处理中，已完成 {snapshot['total_iters']} 轮思考..."
            
            # 【关键】不调用 ReflectionAgent 判断，直接让 ChatAgent 响应
            # 目的是保持用户感知 AI 活跃
            temp_msg = Msg(
                name="user",
                content=f"[系统提示] 请向用户简要说明当前进展：{midway_content}",
                role="user",
            )
            follow_up = await chat_agent.reply(temp_msg)
            
            # 【关键】将 ChatAgent 的中间回复 observe 到 BrainAgent
            # 让 brain 知道自己已经说了什么
            observe_msg = Msg(
                name="system",
                content=f"[已回复用户] {follow_up.get_text_content() or ''}",
                role="system",
            )
            await brain_bg.brain.agent.observe(observe_msg)
            
            already_intervened = True
            break
```

### 4.4 中间介入与最终判断的协作

```
用户提问
    │
    ├──→ ChatAgent 快速回复"稍等"
    │       │
    │       └──→ 启动 midway_watcher（阈值 = 动态计算）
    │
    ├──→ BrainAgent 开始 ReAct 循环 ──────────────────────┐
    │       │                                              │
    │       ├── iter 1: reasoning → search_papers         │
    │       │   │                     │                    │
    │       │   │ post_reasoning      │ acting             │
    │       │   │   ↓                 │   ↓                │
    │       │   │ 更新子状态          │ post_acting        │
    │       │   │                     │   ↓                │
    │       │   │                     │ 更新子状态         │
    │       │   ▼                     ▼                    │
    │       ├── iter 2: reasoning ...                     │
    │       │                                              │
    │       │  【midway_watcher 检查】─────────────────────┤
    │       │   elapsed > threshold?                       │
    │       │   ├── 否 → 继续等待                          │
    │       │   └── 是 → 有工具调用？                      │
    │       │           ├── 否 → 不介入                    │
    │       │           └── 是 → 旁路汇报                  │
    │       │                   ├──→ ChatAgent 中间回复    │
    │       │                   └──→ brain.observe(已回复) │
    │       │                                              │
    │       └── iter N: reasoning（完成，无 tool_use）     │
    │               │                                      │
    │               ▼                                      │
    │           status = "completed"                       │
    │               │                                      │
    │               └──→ judge_after_brain() 最终判断      │
    │                       ├── summarize → ChatAgent 追答 │
    │                       ├── clarify   → ChatAgent 追问 │
    │                       └── ignore    → 静默           │
    │                                                      │
    └──→ ReflectionAgent（只负责最终判断，不负责中间判断）  │
                                                           │
【midway_watcher 独立 Task】───────────────────────────────┘
```

---

## 五、上下文隔离设计

### 5.1 上下文边界定义

| 智能体 | 可见内容 | 不可见内容 |
|-------|---------|-----------|
| **BrainAgent** | 自己的 ReAct 过程 + 工具结果 + 已汇报内容（system 通知） | ChatAgent 的完整闲聊历史 |
| **ChatAgent** | 用户对话历史 + 最终 brain 洞察 + 临时中间上下文 | BrainAgent 的完整工具调用细节 |
| **ReflectionAgent** | 用户对话历史 + ChatAgent 回复 + BrainAgent 快照 | 具体工具参数/结果 |

### 5.2 临时上下文设计（中间汇报时）

**问题**：如何将 BrainAgent 的中间思考过程安全地传递给 ChatAgent，而不污染 ChatAgent 的长期记忆？

**方案**：

```python
async def _inject_midway_context(
    chat_agent: ChatAgent,
    brain_snapshot: dict,
    user_first_msg: str,  # 用户首轮输入（避免重复）
) -> Msg:
    """构建临时上下文消息，注入 ChatAgent。
    
    设计要点：
    1. 只包含 brain 的最新 reasoning 文本（去重：不包含用户首轮输入）
    2. 使用 system 角色，避免破坏 user/assistant 交替
    3. 明确标记为 [中间思考进展]，让 ChatAgent 知道这是临时信息
    """
    
    latest_reasoning = brain_snapshot.get("iterations", [{}])[-1].get("reasoning_text", "")
    iter_count = brain_snapshot.get("total_iters", 0)
    
    context_text = f"""[中间思考进展]
后台大脑智能体正在进行第 {iter_count} 轮思考，当前进展：
{latest_reasoning[:500]}

请基于以上进展，向用户简要说明当前状态（保持自然、口语化，不要透露技术细节）。
"""
    
    return Msg(name="system", content=context_text, role="system")
```

**注入方式**：

```python
# 方式 1：直接调用 chat_agent.reply()（推荐）
temp_msg = Msg(
    name="user",
    content="[系统提示] 请向用户简要说明当前进展：" + latest_reasoning[:200],
    role="user",
)
midway_reply = await chat_agent.reply(temp_msg)

# 方式 2：将临时上下文作为 system 消息加入 ChatAgent memory
# 但 system 消息不会触发 ChatAgent 回复，需要额外调用 reply()
```

**推荐方式 1**：因为 `reply()` 会自动将 user 消息加入 memory 并生成回复，流程最自然。

### 5.3 临时上下文销毁机制

```python
# 中间汇报完成后，临时上下文不应该保留在 ChatAgent memory 中
# 但 ChatAgent 的 reply() 会自动将 user 和 assistant 消息都加入 memory
# 因此需要：

# 1. 中间汇报的 user 消息（"[系统提示] 请向用户..."）不需要保留
#    → 可以设置 chat_agent.save_to_memory = False，但这会影响正常对话
#    → 更好的方案：在中间汇报前临时关闭 save_to_memory，汇报后恢复

# 2. 或者：保留中间汇报内容，因为它确实是对话的一部分
#    → 这是更简单的方案，且语义上合理
```

**最终决策**：**保留中间汇报内容**。因为：
- 中间汇报确实是对话的一部分（ChatAgent 确实向用户说了这些话）
- 保留它可以让后续对话有完整的上下文
- 但需要注意：如果中间汇报次数多，memory 会膨胀 → 后续可通过 memory 压缩解决

### 5.4 observe() 回灌 BrainAgent 的设计

**需求**："对话智能体的中期回复内容 Msg() 应该能够让大脑智能体 observe() 看到"

**实现**：

```python
# 中间汇报完成后
midway_reply_text = midway_reply.get_text_content() or ""

# 将 ChatAgent 的回复 observe 到 BrainAgent
observe_msg = Msg(
    name="system",
    content=f"[已回复用户] {midway_reply_text}",
    role="system",
)
await brain_bg.brain.agent.observe(observe_msg)
```

**风险分析**：observe 回灌是否会影响 ReAct 循环？

| 场景 | 影响 | 结论 |
|------|------|------|
| observe 在 reasoning 过程中回灌 | 消息进入 memory，但当前轮次的 prompt 已经构造完成 | **不影响当前轮** |
| observe 在 reasoning 完成后回灌 | 下一轮 reasoning 的 prompt 会包含 observe 内容 | **正确行为**，brain 需要知道已汇报内容 |
| observe 内容过长 | 可能导致 token 超限 | 需要限制长度（<200 字） |
| observe 角色为 system | 被 formatter 转为 system 消息，不影响 user/assistant 交替 | **安全** |

**结论**：observe 回灌是**安全的**，不会影响 ReAct 循环。

### 5.5 重复筛选（去重）

**问题**：临时上下文中不应重复包含用户首轮输入（因为 ChatAgent 的 memory 中已有）。

**方案**：临时上下文只包含 brain 的思考进展，不包含用户输入。

```python
# 构建临时上下文时，只包含 brain 内容
temp_context = f"[中间思考进展]\n{latest_reasoning[:500]}"

# 不重复包含 user_first_msg
# 因为 ChatAgent 的 memory 中已经有用户输入了
```

---

## 六、流式截取设计（更新版）

### 6.1 核心发现：`print()` 方法可被 Patch

通过对 AgentScope `ReActAgent._reasoning()` 和 `print()` 源码的深度分析，我们发现了一个**优雅的实时截取方案**：

```python
# _reasoning() stream 循环
async for content_chunk in res:
    msg.content = content_chunk.content  # 完整文本快照
    await self.print(msg, False)         # ← 每次 token 都会调用

# print() 方法内部
async def print(self, msg, last=True, speech=None):
    await self.msg_queue.put((deepcopy(msg), last, speech))
    await asyncio.sleep(0)  # ← 关键！yield 控制权给事件循环
    # ... 打印逻辑
```

**关键事实**：
1. `print()` 接收的 `msg` 包含当前已生成的**完整文本快照**
2. `print()` 内部有 `await asyncio.sleep(0)`，**会 yield 控制权给事件循环**
3. 这意味着：定时器任务可以在 stream 过程中运行，读取最新的 buffer

**结论**：通过 Monkey-patch `print()` 方法，可以在**不修改 `_reasoning()` 核心逻辑**的前提下，实现真正的流式截取。

### 6.2 流式截取实现方案

```python
class BrainAgent:
    def __init__(self, ...):
        ...
        # ── 流式截取缓冲区 ──
        self._stream_buffer: str = ""
        self._is_streaming_reasoning: bool = False
        
        # 保存原始 print 方法
        original_print = self.agent.print
        
        async def patched_print(msg, last=True, speech=None):
            """Patch print 方法，在 stream 过程中实时捕获 reasoning 文本。"""
            if self._is_streaming_reasoning:
                # 提取当前完整文本
                text = msg.get_text_content() or ""
                self._stream_buffer = text
            
            # 调用原始 print（保持原有行为）
            await original_print(msg, last, speech)
        
        self.agent.print = patched_print
        
        # ── Hook 配合：标记 stream 起止 ──
        async def _hook_pre_reasoning(react_self, kwargs):
            self._is_streaming_reasoning = True
            self._stream_buffer = ""
            return kwargs
        
        async def _hook_post_reasoning(react_self, kwargs, output):
            self._is_streaming_reasoning = False
            self._latest_reasoning_text = output.get_text_content() or ""
            ...
        
        self.agent.register_instance_hook("pre_reasoning", "brain_stream_start", _hook_pre_reasoning)
        self.agent.register_instance_hook("post_reasoning", "brain_stream_end", _hook_post_reasoning)
    
    def get_stream_buffer(self) -> str:
        """获取当前流式生成中的最新文本（供定时器调用）。"""
        return self._stream_buffer
```

### 6.3 流式截取的数据流

```
_reasoning() stream 循环
    │
    ├── token 1 → msg.content = "我" 
    │   → await self.print(msg, False)
    │       → patched_print()
    │           → _stream_buffer = "我"
    │           → await asyncio.sleep(0)  ← 定时器获得执行权
    │               → midway_watcher 读取 _stream_buffer → "我"
    │
    ├── token 2 → msg.content = "我在"
    │   → await self.print(msg, False)
    │       → patched_print()
    │           → _stream_buffer = "我在"
    │           → await asyncio.sleep(0)  ← 定时器获得执行权
    │               → midway_watcher 读取 _stream_buffer → "我在"
    │
    ├── token 3 → msg.content = "我在查"
    │   → ...
    │
    └── stream 完成
        → post_reasoning hook
        → _stream_buffer 不再更新
```

### 6.4 延迟分析

| 模型 | 平均每 token 耗时 | `print()` 触发频率 | 定时器读取延迟 | 可接受？ |
|------|------------------|-------------------|--------------|---------|
| qwen3.6-27b | 50-100ms | 每 token 一次 | <100ms | ✅ 优秀 |
| GPT-4 | 30-80ms | 每 token 一次 | <80ms | ✅ 优秀 |
| DeepSeek-R1 | 100-300ms | 每 token 一次 | <300ms | ✅ 可接受 |

**结论**：`print()` Patch 方案的延迟极低（<300ms），完全满足"实时截取"需求。

### 6.5 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `print()` 被 `_acting` 调用时误捕获 | `_stream_buffer` 被 tool_result 污染 | 通过 `_is_streaming_reasoning` 标志区分，只有 reasoning stream 期间才捕获 |
| `print()` Patch 导致 AgentScope 升级不兼容 | 未来升级后行为异常 | Patch 代码集中在 BrainAgent 初始化中，升级时容易替换；同时添加版本检查 |
| 高频更新 `_stream_buffer` 导致竞争条件 | 定时器读取到不完整内容 | 使用 Python GIL 保护（单线程模型下无需锁）；`str` 赋值是原子操作 |

---

## 七、数据流与架构总图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          main7_chatroom.py                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   ChatAgent  │  │ EmotionAgent │  │ BrainAgent   │  │ Reflection  │ │
│  │  (前台对话)   │  │  (前台表情)   │  │  (后台思考)   │  │  (元认知)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                  │                  │                  │        │
│         └──────────────────┴──────────────────┘                  │        │
│                          │                                       │        │
│              FrontStagePipeline                                  │        │
│                          │                                       │        │
│         ┌────────────────┴────────────────┐                      │        │
│         │         EventBus                 │                      │        │
│         │   (user.input / brain.thought)   │                      │        │
│         └────────────────┬────────────────┘                      │        │
│                          │                                       │        │
│              ┌───────────┴───────────┐                          │        │
│              │ BackgroundBrainAgent  │                          │        │
│              │   (后台常驻包装器)     │                          │        │
│              │   status: thinking    │                          │        │
│              │   sub_status: reason  │                          │        │
│              └───────────┬───────────┘                          │        │
│                          │                                       │        │
│              ┌───────────┴───────────┐                          │        │
│              │   midway_watcher      │◄── 独立 asyncio.Task     │        │
│              │   (中间过程监听器)     │    阈值 = 动态计算         │        │
│              │   每 1s 检查状态      │                          │        │
│              └───────────┬───────────┘                          │        │
│                          │                                       │        │
│              ┌───────────┴───────────────────────────┐          │        │
│              │  触发条件：elapsed > threshold          │          │        │
│              │  + has_used_tools = True               │          │        │
│              └───────────┬───────────────────────────┘          │        │
│                          │                                       │        │
│              ┌───────────┴───────────┐                          │        │
│              │   旁路触发 ChatAgent   │                          │        │
│              │   中间汇报给用户       │                          │        │
│              └───────────┬───────────┘                          │        │
│                          │                                       │        │
│              ┌───────────┴───────────┐                          │        │
│              │   brain.observe(已回复)│                          │        │
│              │   (让 brain 知道已说)  │                          │        │
│              └────────────────────────┘                          │        │
│                                                                  │        │
│  【最终判断流程】─────────────────────────────────────────────────┘        │
│  BrainAgent 完成后 → status = "completed"                                  │
│  → judge_after_brain() 最终判断                                            │
│  → summarize / clarify / ignore                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 八、实现路径（分阶段）

### 阶段 1：状态同步 + Hook 增强 + 流式截取（1-2 天）

**目标**：实现子状态追踪、状态信息追加、流式截取。

**修改文件**：
- `deerberry/agent/brain_agent.py`
  - [ ] 新增 `_sub_status`, `_latest_reasoning_text`, `_latest_tool_name`, `_has_used_tools`, `_think_start_ts`
  - [ ] 在 hook 中更新子状态
  - [ ] 新增 `_build_status_suffix()` 方法
  - [ ] 在 `think()` 中追加状态信息到 insight
  - [ ] **新增**：Patch `print()` 方法实现流式截取
  - [ ] **新增**：`pre_reasoning` hook 标记 stream 起止
  - [ ] 新增 `get_stream_buffer()` 公共接口

**验证点**：
- [ ] `post_reasoning` 触发后 `_sub_status` 正确更新
- [ ] `post_acting` 触发后 `_sub_status` 回到 idle
- [ ] `_has_used_tools` 在有工具调用时标记为 True
- [ ] **新增**：stream 过程中 `_stream_buffer` 实时更新
- [ ] **新增**：非 stream 模式下 `_stream_buffer` 不捕获 acting 内容

### 阶段 2：阈值反思限制 + 中间介入（2-3 天）

**目标**：实现动态阈值和中间旁路汇报。

**修改文件**：
- `deerberry/pipeline/chatroom_controller.py`
  - [ ] `BackgroundBrainAgent` 暴露 `get_midway_snapshot()` 方法
- `deerberry/agent/reflection_agent.py`
  - [ ] 新增 `compute_dynamic_threshold()` 静态方法
- `main7_chatroom.py`
  - [ ] 新增 `midway_watcher()` 独立 Task
  - [ ] 在 `front_stage.respond()` 后启动计时器
  - [ ] 实现旁路触发 ChatAgent 中间汇报
  - [ ] 实现 `brain.observe()` 回灌

**验证点**：
- [ ] 无工具调用时，不触发中间介入
- [ ] 有工具调用 + 超时后，正确触发中间汇报
- [ ] 动态阈值随 chat 长度变化
- [ ] 中间汇报后 brain 能 observe 到已回复内容

### 阶段 3：上下文隔离优化（1 天）

**目标**：优化临时上下文和去重逻辑。

**修改文件**：
- `deerberry/agent/reflection_agent.py`
  - [ ] 新增 `_build_midway_context()` 方法
- `main7_chatroom.py`
  - [ ] 优化临时上下文注入方式
  - [ ] 添加冷却期（每轮最多 1 次中间介入）

**验证点**：
- [ ] 临时上下文不包含用户首轮输入（去重）
- [ ] 同一轮不重复触发中间介入

### 阶段 4：集成测试（1-2 天）

**测试场景**：
1. 简单问题（无工具调用）：只在 completed 后判断
2. 复杂问题（有工具调用）：中间介入 + 最终判断
3. 工具调用缓慢：中间介入显示"正在查询..."
4. 多轮对话：brain 能感知已汇报内容，避免重复
5. **新增**：流式截取验证（`stream=True` 模式下 buffer 实时更新）
6. **新增**：慢模型模拟（人为增加 reasoning 延迟，验证流式截取效果）

---

## 九、风险与待确认事项

### 9.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **Hook 异常阻塞 ReAct 循环** | BrainAgent 卡住 | Hook 内包 try/except，异常时打印日志并返回原始 output |
| **定时器任务与 BrainAgent 并发冲突** | 状态竞争 | 使用 asyncio 单线程模型，Hook 和定时器在同一线程串行执行 |
| **中间汇报污染 ChatAgent 上下文** | ChatAgent 后续回复混乱 | 中间汇报内容明确标记为 `[系统提示]`，且保留在 memory 中 |
| **brain.observe() 回灌导致 token 超限** | LLM 报错 | observe 消息限制长度（<200 字），必要时做摘要 |
| **多次中间汇报打断用户** | 用户体验差 | 每轮最多 1 次中间介入，冷却期机制 |
| **动态阈值公式不当** | 阈值过高/过低 | 初期使用字符数简单计算，后续根据实际表现调参 |

### 9.2 待用户确认的技术点

#### ✅ 确认 1：中间汇报的频率

**用户确认**：无限次介入，但上限 10 轮，设置为可配置变量。

**实现**：`MAX_MIDWAY_INTERVENTIONS = 10`（可配置），每轮通过计数器控制。超过上限后不再介入，等待 completed。

#### ✅ 确认 2：动态阈值的公式

**用户确认**：`threshold = 5.0 + chat_length / 10.0`（上限 30 秒）合适，后续通过日志和实际测试调参。

#### ✅ 确认 3：中间汇报的内容风格

**用户确认**：不需要控制风格。ChatAgent 基于中间思考上下文自主决定如何表达。

**说明**：我提出此问题的初衷是考虑是否在临时上下文中添加"风格指导"的 system prompt。但用户说得对——ChatAgent 有自己的 system prompt，会基于提供的上下文自主生成自然回复，无需额外控制。

#### ✅ 确认 4：状态信息追加位置

**用户问题**："状态信息追加位置是什么意思？"

**解释**：这是指 BrainAgent 的子状态信息（如"正在调用 search_papers 工具中..."）应该放在哪里，以便其他智能体获取。

用户原始需求中提到："这种子状态信息也可以作为对话信息来补充，介入目前这个对话信息我认为是固定一个字符串追加到大脑智能体的思考尾部较佳。"

**实现方式**：在 `BrainAgent.think()` 返回的 `insight` 字符串尾部追加状态摘要。例如：
```python
data = {
    "insight": text.strip() + "\n\n[系统状态] 正在调用 search_papers 工具中...",
    ...
}
```

这样 ReflectionAgent 的 `judge_after_brain()` 和 ChatAgent 都能获取到状态信息。

#### ✅ 确认 5：observe 回灌的角色

**用户确认**：`assistant` 角色。

**实现**：
```python
observe_msg = Msg(
    name="assistant",
    content=f"[已回复用户] {midway_reply_text}",
    role="assistant",
)
await brain_bg.brain.agent.observe(observe_msg)
```

**说明**：使用 `assistant` 角色意味着 BrainAgent 会将这条消息视为"自己之前说过的话"。这在语义上表示"我（通过 ChatAgent）已经向用户汇报过了"，有利于 BrainAgent 后续避免重复。

#### ✅ 确认 6：需要真正的流式截取

**用户确认**：需要真正的流式截取，因为项目后续要通过配置支持多组 LLM（包括可能更慢的模型）。

**更新方案**：详见第六章「流式截取设计（更新版）」。核心思路：通过 Monkey-patch `ReActAgent.print()` 方法，在 stream 过程中实时暴露 `_stream_buffer`。

---

## 十、总结

| 需求 | 推荐方案 | 复杂度 | 优先级 | 依赖 |
|------|---------|--------|--------|------|
| 状态同步（子状态） | Hook 更新 `_sub_status` | 低 | P0 | 无 |
| 状态信息追加 | `think()` 返回 insight 尾部追加 | 低 | P0 | 状态同步 |
| 阈值反思限制 | 动态阈值 + 独立 midway_watcher Task | 中 | P0 | 状态同步 |
| 上下文隔离 | 临时上下文 + observe 回灌 (assistant) | 中 | P0 | 阈值反思 |
| **流式截取** | **Patch `print()` 方法** | **中** | **P0** | **无** |
| 配置化（中间汇报上限等） | 环境变量 / config.py | 低 | P1 | 阈值反思 |

**下一步行动**：
1. 用户确认上述 6 个待确认技术点
2. 按阶段 1 → 阶段 2 → 阶段 3 → 阶段 4 顺序实现
3. 每阶段完成后进行集成测试
