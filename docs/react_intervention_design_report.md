# 技术实现报告：ReActAgent 中间思考过程截断与反思智能体介入机制

> 报告日期：2026-05-06  
> 关联模块：`deerberry/agent/brain_agent.py`、`deerberry/agent/reflection_agent.py`、`deerberry/pipeline/chatroom_controller.py`  
> 核心依赖：AgentScope `ReActAgent` Hook 机制

---

## 一、项目背景与目标

### 1.1 当前痛点

当前架构中，BrainAgent（ReActAgent）在后台完成全部 `max_iters` 轮推理-行动循环后，ReflectionAgent 才在 `judge_after_brain()` 中做一次最终判断。这意味着：

- 用户提出问题 → ChatAgent 快速回复"稍等" → **用户进入漫长等待**（数秒~数十秒）→ BrainAgent 终于完成 → ReflectionAgent 判断是否追答
- 用户在等待期间没有任何信息反馈，体验差
- BrainAgent 的中间思考过程（如已搜索到论文标题、已读取到论文摘要）被浪费，直到最后才一次性汇总

### 1.2 目标

在 BrainAgent 的 ReAct 循环**中间**（非最终完成后），由 ReflectionAgent 判断：
- 当前已产出的思考结果/工具结果是否**足够有价值**立即汇报给用户
- 如果足够有价值 → **截断当前思考**，触发 ChatAgent 做中间汇报
- BrainAgent **不受影响**，继续在后台完成剩余轮次
- BrainAgent 能**观察到**自己已被汇报的内容，避免后续重复

---

## 二、五个功能需求逐条深入分析

### 2.1 时间硬规则：中间截断汇报

**需求描述**：
> 设置响应时间阈值，当前台已响应且 BrainAgent 后台思考一段时间后仍未给出总结回复，ReflectionAgent 硬性获取 BrainAgent 的上下文（含正在流式生成的句子），截取后作为一次正常的大脑思考总结发送到前台智能体。

**核心难点**：

| 难点 | 分析 | 当前可行性 |
|------|------|-----------|
| BrainAgent `reply()` 是原子性异步调用 | `await self.agent.reply(user_msg)` 会阻塞直到全部 `max_iters` 轮完成 | ⚠️ 需要并发架构 |
| 外部无法"打断" ReAct 循环取中间状态 | `for _ in range(self.max_iters)` 在 `ReActAgent.reply()` 内部 | ⚠️ 需要 Hook 介入 |
| "正在流式生成的句子"如何获取 | `_reasoning()` 内部 stream 循环是局部变量 | ❌ 需要特殊设计（详见 2.5） |

**可行的截断时机**：

实际上，ReAct 循环中只有两种"可截断点"：

1. **Reasoning 完成后**（`post_reasoning` hook）：LLM 已输出完整思考内容，但可能还没执行工具
2. **Acting 完成后**（`post_acting` hook）：工具已执行完毕，有实际数据

**不可截断点**：
- `pre_reasoning`：还没生成内容，无信息可汇报
- `pre_acting`：工具还没执行，无结果可汇报
- Reasoning 的 stream 过程中：框架层面难以实时读取（详见 2.5）

**推荐策略**：

```
用户提问
    │
    ▼
ChatAgent 回复"稍等"  ─────────────────────────────┐
    │                                              │
    ▼                                              │
BrainAgent 开始 ReAct 循环                         │
    │                                              │
    ├── iter 1: reasoning ──→ search_papers        │
    │   │                     │                    │
    │   │ post_reasoning      │ acting             │
    │   │   ↓                 │   ↓                │
    │   │ 记录本轮思考         │ post_acting        │
    │   │                     │   ↓                │
    │   │                     │ 记录工具结果        │
    │   ▼                     ▼                    │
    ├── iter 2: reasoning ──→ read_paper           │
    │   │                     │                    │
    │   │                     │                    │
    │   ▼                     ▼                    │
    │  【定时器触发】──────────┼────────────────────┤
    │   │                     │                    │
    │   ▼                     │                    │
    │  ReflectionAgent 检查    │                    │
    │   "iter 1 的 search_papers 结果是否有价值？"  │
    │   → 是 → 截取 iter 1 结果                    │
    │          → 发送给 ChatAgent 汇报              │
    │          → BrainAgent 继续 iter 2（不受影响） │
    │                                             │
    └── iter 3: reasoning（最终总结）              │
        │                                         │
        ▼                                         ▼
    judge_after_brain() 判断是否需要最终追答
```

**关键结论**：
- "时间硬规则"不是"打断 BrainAgent"，而是"**旁路触发 ChatAgent 做中间汇报**
- BrainAgent 的 `reply()` 继续在后台运行，不受影响
- 截取的时机是 `post_acting` 或 `post_reasoning`，而非"任意时刻"

---

### 2.2 大脑智能体必须能够观察到反思结果

**需求描述**：
> 反思智能体将大脑智能体部分思考过程总结为答复告诉用户后，需要将本次截取内容让大脑智能体 `observe()` 到，让大脑智能体知道自己已经总结了哪些内容。

**为什么需要这个机制**：

假设场景：
```
iter 1: BrainAgent 搜索到 5 篇 Transformer 论文
        ReflectionAgent 截取 → ChatAgent 汇报："查到了5篇论文，其中《Attention Is All You Need》最经典"
        
iter 2: BrainAgent 继续思考...
        ❌ 如果没有 observe 机制，BrainAgent 不知道"5篇论文"已经告诉用户了
        ❌ BrainAgent 可能又在 iter 2 中重新搜索或重复汇报
        
iter 2: BrainAgent 看到 observe 消息："已汇报：查到了5篇论文..."
        ✅ BrainAgent 知道 iter 1 的结果已消费
        ✅ iter 2 可以专注于"深入阅读最经典的论文"
```

**实现方式**：

```python
# 在 main7_chatroom.py 中，中间汇报后
intervention_msg = Msg(
    name="system",
    content=f"[中间汇报记录] 以下内容已作为阶段性结果告知用户：\n{summary_text}",
    role="system",
)
await brain_agent.observe(intervention_msg)  # 或 brain_agent.agent.observe(...)
```

**注入位置的选择**：

| 注入方式 | 效果 | 风险 |
|---------|------|------|
| `brain_agent.observe(msg)`（调用 AgentBase.observe） | 消息进入 BrainAgent 的 short-term memory | ✅ 推荐 |
| 直接操作 `brain_agent.agent.memory.add(msg)` | 效果同上 | ⚠️ 侵入 memory 内部 |
| 修改 BrainAgent 的 sys_prompt | 全局影响 | ❌ 不灵活 |

**最佳实践**：
- 使用 `brain_agent.agent.observe(intervention_msg)`（`ReActAgent` 继承自 `AgentBase`，`observe()` 已实现）
- 消息角色设为 `"system"`，让 LLM 明确知道这是"系统通知"而非用户输入
- 消息内容包含明确的标记：`[已汇报给用户] xxx`，避免 LLM 困惑

---

### 2.3 上下文隔离

**需求描述**：
> 大脑智能体只能看到"自己的 ReAct 思考过程 + 工具调用和结果 + 受反思智能体控制的已对话的对话智能体内容"；反思智能体的上下文是"本场与用户的对话历史 + 对话智能体回复内容 + 截取大脑智能体部分思考过程"。

**当前架构的问题**：

当前 `main7_chatroom.py` 中，BrainAgent 的输入只有用户原始消息（通过 EventBus 投递）。BrainAgent 的 memory 是独立的 `InMemoryMemory()`，不包含 ChatAgent 的对话历史。

但 `ReActAgent._reasoning()` 中的 prompt 构造会包含 `self.memory` 中的所有消息。如果 `observe()` 被用来注入中间汇报记录，这些记录会进入 BrainAgent 的 memory。

**需要明确隔离边界**：

```
┌─────────────────────────────────────────────────────────────┐
│                    BrainAgent 上下文                         │
├─────────────────────────────────────────────────────────────┤
│  ✅ 自己的 ReAct 思考过程                                    │
│     - iter 1 reasoning 输出                                  │
│     - iter 1 acting 结果                                     │
│     - iter 2 reasoning 输出                                  │
│     - iter 2 acting 结果                                     │
│                                                              │
│  ✅ 工具调用和工具结果                                       │
│     - tool_use: search_papers(...)                           │
│     - tool_result: 5篇论文                                   │
│                                                              │
│  ✅ 受 ReflectionAgent 控制的 ChatAgent 中间汇报             │
│     - system: [已汇报给用户] 查到了5篇论文...                │
│                                                              │
│  ❌ 用户的原始输入（不应该重复出现）                         │
│     - 当前已通过 EventBus 投递，不应再在 memory 中出现     │
│                                                              │
│  ❌ ChatAgent 的完整对话历史                                │
│     - "哈喽欢迎回来"等闲聊内容不应污染 BrainAgent           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 ReflectionAgent 上下文                       │
├─────────────────────────────────────────────────────────────┤
│  ✅ 本场与用户的对话历史（通过 observe 积累）                │
│     - user: "帮我查 Transformer 论文"                        │
│     - user: "你刚刚查了什么"                                 │
│                                                              │
│  ✅ 对话智能体回复内容（通过 observe 积累）                  │
│     - assistant: "哇这题超难，我先查查"                      │
│                                                              │
│  ✅ 截取大脑智能体部分思考过程                               │
│     - iter 1 reasoning 输出                                  │
│     - iter 1 acting 结果                                     │
│                                                              │
│  ❌ BrainAgent 的完整工具调用细节（不需要）                  │
│     - 如 tool 的原始 JSON 参数等                            │
└─────────────────────────────────────────────────────────────┘
```

**实现建议**：

1. **BrainAgent 侧**：
   - 保持当前设计：用户输入通过 EventBus 投递，不通过 memory
   - `observe()` 只注入"系统通知"（中间汇报记录）
   - 不注入 ChatAgent 的闲聊历史

2. **ReflectionAgent 侧**：
   - `observe()` 积累用户输入 + ChatAgent 回复（已在 main7 中实现）
   - 从 BrainAgent Hook 中获取 iter 结果（通过 Hook 保存）

---

### 2.4 大脑智能体的思考过程轮数

**需求描述**：
> 【reasoning → tool_call(若有)】为一轮 loop / iter。明确澄清：【reasoning → reasoning】是 2 轮思考过程，【reasoning → tool_call → reasoning → reasoning】是 3 轮 loop。

**ReActAgent 源码中的循环结构**：

```python
# ReActAgent.reply() 核心循环
for _ in range(self.max_iters):
    msg_reasoning = await self._reasoning(tool_choice)  # ← iter N 开始：reasoning
    
    futures = [
        self._acting(tool_call)
        for tool_call in msg_reasoning.get_content_blocks("tool_use")
    ]
    # ... 执行 acting ...
    
    if not msg_reasoning.has_content_blocks("tool_use"):
        break  # ← iter N 结束：无 tool_use，退出
```

**轮数定义映射**：

| 用户定义 | ReActAgent 源码对应 | 说明 |
|---------|-------------------|------|
| 1 轮 loop = 【reasoning + tool_call(若有)】 | `for` 循环的一次完整迭代 | 包含 `_reasoning()` + 可选的 `_acting()` |
| 【reasoning → reasoning】= 2 轮 | 第一次 iteration 只有 reasoning（无 tool_use）→ break；第二次 `reply()` 调用时新的 iteration | 这里用户可能指的是两个独立的 reasoning 调用 |
| 【reasoning → tool_call → reasoning → reasoning】= 3 轮 | iter 1: reasoning + acting；iter 2: reasoning（无 tool_use，break）；iter 3: 不存在 | 需要澄清 |

**澄清建议**：

在 ReActAgent 的 `for` 循环中：
- **iter 1** = `_reasoning()` → 如果有 tool_use → `_acting()` → 回到 loop 顶部
- **iter 2** = `_reasoning()` → 如果无 tool_use → `break`

所以 "【reasoning → reasoning】" 实际上是在描述两次 `_reasoning()` 调用，而不是一个完整的 iter。

**推荐的标准定义**：

```
轮次(iter) = for 循环的一次完整迭代
  = _reasoning() 调用
  + 可选的 _acting() 调用（如果有 tool_use）

iter 1: reasoning_1 → [acting_1] → (有 tool_use，继续)
iter 2: reasoning_2 → [acting_2] → (有 tool_use，继续)
iter 3: reasoning_3 → (无 tool_use，break，完成)

总轮数 = 3 轮
```

**轮次追踪的实现**：

通过 Hook 可以轻松实现轮次计数：

```python
# 在 BrainAgent.__init__ 中注册 hook
self._current_iter = 0

async def pre_reasoning_hook(self, kwargs):
    self._current_iter += 1
    print(f"[ReAct] 🔄 第 {self._current_iter} 轮开始")
    return kwargs

async def post_acting_hook(self, kwargs, output):
    print(f"[ReAct] ✅ 第 {self._current_iter} 轮完成")
    return output
```

---

### 2.5 大脑智能体流式正在生成的截取方式

**需求描述**：
> ReActAgent 的 `print()` 使用流式打印，需要设计如何截取"正在流式生成的句子"。

**当前 ReActAgent `_reasoning()` 的 stream 实现**：

```python
# _react_agent.py ~585-606
if self.model.stream:
    async for content_chunk in res:
        msg.invocation_id = content_chunk.id
        msg.content = content_chunk.content  # ← 完整文本快照
        
        await self.print(msg, False, speech=speech)  # ← 流式打印
else:
    msg.invocation_id = res.id
    msg.content = list(res.content)
```

**关键事实**：
- `content_chunk.content` 是**完整文本快照**（如 `'好'` → `'好开心'` → `'好开心呀'`），不是增量 delta
- `msg.content` 每次循环都被**覆盖**为当前完整内容
- `await self.print(msg, False)` 每次都会打印完整内容（AgentScope 内部做了增量处理）

**这意味着**：在 `post_reasoning` hook 中，`output` 参数已经包含了**本轮 reasoning 的完整生成内容**。不需要在 stream 过程中实时截取。

**但用户的真实需求可能是**：

当"时间硬规则"触发时（超时），如果当前正处于 `_reasoning()` 的 stream 循环中（即 `post_reasoning` 还没触发），如何获取"已经生成了什么"？

**三种场景分析**：

| 场景 | BrainAgent 当前状态 | 可截取的内容 | 获取方式 |
|------|-------------------|------------|---------|
| A | 正在 `_reasoning()` stream 中 | 已生成的部分文本 | ❌ 难以获取（局部变量） |
| B | `_reasoning()` 已完成，`post_reasoning` 已触发 | 本轮完整 reasoning 输出 | ✅ 通过 Hook 保存 |
| C | 正在 `_acting()` 中 | 上一轮 reasoning 输出 | ✅ 通过 Hook 保存 |
| D | `_acting()` 已完成，`post_acting` 已触发 | 本轮 tool_result | ✅ 通过 Hook 保存 |

**方案对比**：

| 方案 | 描述 | 侵入性 | 可行性 |
|------|------|--------|--------|
| **方案 1：Hook 保存（推荐）** | 在 `post_reasoning` / `post_acting` 中保存每轮结果 | 低 | ✅ 高 |
| **方案 2：Monkey-patch stream 循环** | 在 `_reasoning()` 的 `async for` 中暴露回调 | 中 | ⚠️ 中 |
| **方案 3：继承 ReActAgent 重写 `_reasoning()`** | 自定义子类，加入 `_stream_buffer` 属性 | 中 | ⚠️ 中 |
| **方案 4：外部轮询 `self.memory`** | 定时器检查 BrainAgent memory 最新内容 | 低 | ⚠️ 中 |

**推荐方案 1 的详细设计**：

```python
class BrainAgent:
    def __init__(self, ...):
        ...
        # 注册 hook 保存每轮结果
        self.agent.register_instance_hook("post_reasoning", self._save_reasoning)
        self.agent.register_instance_hook("post_acting", self._save_acting)
        
        # 轮次结果缓存
        self._iter_results: list[dict] = []  # [{"iter": 1, "reasoning": "...", "acting": "..."}, ...]
        self._current_iter = 0
    
    async def _save_reasoning(self, kwargs, output):
        """post_reasoning hook：保存本轮 reasoning 输出"""
        self._current_iter += 1
        text = output.get_text_content() or ""
        tool_uses = output.get_content_blocks("tool_use")
        
        self._iter_results.append({
            "iter": self._current_iter,
            "reasoning_text": text,
            "tool_calls": [t["name"] for t in tool_uses],
            "timestamp": time.time(),
        })
        return output
    
    async def _save_acting(self, kwargs, output):
        """post_acting hook：保存本轮 acting 结果"""
        tool_name = kwargs["tool_call"]["name"]
        
        # 从 memory 读取 tool_result（因为 post_acting 的 output 是 None）
        memory = await self.agent.memory.get_memory()
        last_result = None
        for msg in reversed(memory):
            if msg.role == "system":
                blocks = msg.get_content_blocks("tool_result")
                if blocks and blocks[0].get("name") == tool_name:
                    last_result = blocks[0].get("output", "")
                    break
        
        if self._iter_results:
            self._iter_results[-1]["acting_result"] = {
                "tool_name": tool_name,
                "result_summary": str(last_result)[:500] if last_result else "",
            }
        return output
    
    def get_latest_iter_result(self) -> dict | None:
        """供 ReflectionAgent 调用，获取最新一轮的结果"""
        return self._iter_results[-1] if self._iter_results else None
```

**关于"正在流式生成"的现实**：

如果定时器触发时，`post_reasoning` 尚未触发（即还在 stream 中），实际上**没有好办法**获取中间状态。因为：
- `_reasoning()` 是一个阻塞的异步函数
- 其内部的 `async for` 循环不会 yield 控制权给外部任务（除非 `await self.print()` 内部有 sleep）

**务实的解决方案**：

1. **短期**：只截取 `post_reasoning` / `post_acting` 已保存的完整结果
   - 如果定时器触发时还在 reasoning 中 → 等待 `post_reasoning` 触发后再截取（通常只有几百毫秒延迟）
   
2. **长期**：如果真的需要在 stream 过程中截取，需要：
   - 继承 `ReActAgent` 创建 `StreamingBrainAgent`
   - 重写 `_reasoning()`，在 `async for` 循环中暴露 `self._current_streaming_text`
   - 或 Monkey-patch：
     ```python
     original_reasoning = self.agent._reasoning
     async def patched_reasoning(*args, **kwargs):
         # 注入回调...
         return await original_reasoning(*args, **kwargs)
     self.agent._reasoning = patched_reasoning
     ```

---

## 三、架构设计总览

### 3.1 数据流

```
用户输入
    │
    ├──→ ChatAgent ──→ 前台快速回复"稍等"
    │
    ├──→ BrainAgent ──→ ReAct Loop（后台运行）
    │       │
    │       ├── iter 1: reasoning → acting
    │       │   │
    │       │   ├── post_reasoning hook ──→ 保存 reasoning_1
    │       │   ├── post_acting hook ──→ 保存 acting_1
    │       │   │
    │       │   └──→ 定时器检查（独立任务）
    │       │           │
    │       │           ├── 未超时 → 继续等待
    │       │           └── 超时 ──→ ReflectionAgent 判断
    │       │                   │
    │       │                   ├── 有价值？──→ 截取 iter 1 结果
    │       │                   │               ├──→ ChatAgent 中间汇报
    │       │                   │               └──→ brain_agent.observe(已汇报)
    │       │                   │
    │       │                   └── 无价值？──→ 继续等待
    │       │
    │       ├── iter 2: reasoning → acting
    │       │   │
    │       │   └── ...
    │       │
    │       └── iter 3: reasoning（完成，无 tool_use）
    │           │
    │           └──→ post_reply（或 think() 返回）
    │               │
    │               └──→ judge_after_brain() 最终判断
    │                   │
    │                   ├── summarize/ignore/clarify
    │                   └──→ ChatAgent 最终追答（如果需要）
    │
    └──→ ReflectionAgent
            │
            ├── observe(user_msg)     # 积累用户输入
            ├── observe(chat_result)  # 积累 ChatAgent 回复
            │
            └── 定时器任务（独立 asyncio.Task）
                    │
                    ├── 每 500ms 检查一次
                    ├── 读取 BrainAgent._iter_results
                    ├── 调用 judge_midway() 判断
                    └── 触发中间汇报
```

### 3.2 Hook 触发时机映射

| Hook | 触发时 BrainAgent 状态 | Reflection 可获取信息 | 是否适合触发中间汇报 |
|------|----------------------|---------------------|-------------------|
| `pre_reasoning` | 即将调用 LLM | 无（还没生成） | ❌ |
| `post_reasoning` | LLM 已输出思考+tool_use | 本轮思考文本、决定调用的工具 | ⚠️（如果有 tool_use，建议等 acting 完成） |
| `pre_acting` | 即将执行工具 | 工具名、参数 | ❌ |
| `post_acting` | 工具已执行完毕 | 工具返回结果 | ✅ **最适合** |

**最佳实践**：
- 轻量级工具（如 memory 检索）：`post_acting` 后立即判断
- 重量级工具（如论文搜索、PDF 读取）：`post_acting` 后检查返回结果长度，若内容丰富则触发汇报

---

## 四、关键技术方案

### 4.1 定时器 + Hook 状态机

```python
class BrainAgent:
    def __init__(self, ...):
        ...
        # Hook 注册
        self.agent.register_instance_hook("post_reasoning", self._on_post_reasoning)
        self.agent.register_instance_hook("post_acting", self._on_post_acting)
        
        # 状态机
        self._iter_results: list[dict] = []
        self._current_iter = 0
        self._is_running = False
        self._latest_reasoning = ""
        self._latest_acting = None
    
    async def _on_post_reasoning(self, kwargs, output):
        self._current_iter += 1
        self._latest_reasoning = output.get_text_content() or ""
        self._iter_results.append({
            "iter": self._current_iter,
            "reasoning": self._latest_reasoning,
            "acting": None,
        })
        return output
    
    async def _on_post_acting(self, kwargs, output):
        tool_name = kwargs["tool_call"]["name"]
        self._latest_acting = {"tool": tool_name, "result": "..."}
        if self._iter_results:
            self._iter_results[-1]["acting"] = self._latest_acting
        return output
    
    def get_current_snapshot(self) -> dict:
        """供 ReflectionAgent 定时器调用"""
        return {
            "iter_count": self._current_iter,
            "latest_reasoning": self._latest_reasoning,
            "latest_acting": self._latest_acting,
            "all_results": self._iter_results,
        }
```

### 4.2 上下文注入（observe）

```python
# 在 main7_chatroom.py 中，中间汇报后
async def trigger_midway_report(brain_agent, reflection, chat_agent, iter_result):
    # 1. ReflectionAgent 判断是否有价值
    # （此处可复用 judge_after_brain 的逻辑，或新增 judge_midway 方法）
    
    # 2. 生成汇报内容
    summary = iter_result["reasoning"][:200]  # 截取前 200 字
    
    # 3. ChatAgent 汇报给用户
    follow_up = await chat_agent.reply(
        Msg(name="user", content=f"请向用户汇报以下信息：{summary}", role="user")
    )
    
    # 4. BrainAgent 观察到自己已汇报的内容
    observe_msg = Msg(
        name="system",
        content=f"[系统通知] 以下内容已作为阶段性结果告知用户，请后续思考避免重复：\n{summary}",
        role="system",
    )
    await brain_agent.observe(observe_msg)
```

### 4.3 流式生成的现实处理

| 情况 | 处理方式 |
|------|---------|
| 模型 stream=False | `post_reasoning` 的 `output` 直接包含完整内容，无问题 |
| 模型 stream=True，但 reasoning 很快（<1s） | `post_reasoning` 几乎立即触发，无问题 |
| 模型 stream=True，reasoning 很慢（>3s） | 定时器触发时可能还在 stream 中 → 建议等待 `post_reasoning` 完成后再截取 |
| 确实需要"逐字截取" | 需 Monkey-patch 或继承 ReActAgent 重写 `_reasoning()` |

**务实的建议**：

对于当前项目（`qwen3.6-27b` 模型，reasoning 通常在 1-3 秒内完成），`post_reasoning` hook 的延迟是可接受的。"正在生成的句子"这一需求可以降级为"最近一轮已完成的 reasoning 结果"。

如果未来切换到更慢的模型（如 deep reasoning 模型），再考虑 Monkey-patch 方案。

---

## 五、推荐实现路径（分阶段）

### 阶段 1：Hook 保存 + 最终汇报增强（1-2 天）

1. 在 `BrainAgent` 中注册 `post_reasoning` / `post_acting` hook
2. 保存每轮 iter 结果到 `_iter_results`
3. 在 `judge_after_brain()` 中利用 iter 历史做更精细的判断
   - 例如：如果 iter 1 已搜索到论文，iter 2 只是记录记忆，则判断为"已有足够信息"

### 阶段 2：定时器 + 中间汇报（2-3 天）

1. 在 `main7_chatroom.py` 中创建独立的定时器任务
2. 定时器读取 `BrainAgent._iter_results`
3. 调用 `ReflectionAgent.judge_midway()` 判断
4. 触发 ChatAgent 中间汇报
5. 通过 `brain_agent.observe()` 注入"已汇报"通知

### 阶段 3：流式截取优化（可选，视需求）

1. 如果 stage 2 的延迟不可接受，再考虑 Monkey-patch `_reasoning()`
2. 在 stream 循环中暴露 `self._current_streaming_text`
3. 定时器直接读取 streaming buffer

---

## 六、风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **Hook 执行异常会阻塞 ReAct 循环** | BrainAgent 卡住 | Hook 中包 try/except，异常时打印日志并返回原始 output |
| **定时器任务与 BrainAgent 并发冲突** | 状态竞争 | 使用 asyncio 单线程模型，Hook 和定时器在同一线程中串行执行 |
| **中间汇报污染 ChatAgent 上下文** | ChatAgent 后续回复混乱 | 中间汇报以 `user` 角色 trigger，汇报结果作为 `assistant` 存入 memory，保持对话连贯 |
| **BrainAgent observe 注入导致上下文过长** | LLM token 超限 | observe 消息保持简短（<100 字），必要时对 iter 结果做摘要 |
| **多次中间汇报导致用户被打断** | 用户体验差 | ReflectionAgent 的 `judge_midway()` 增加冷却期（同一轮不重复汇报） |
| **流式截取 Monkey-patch 导致 AgentScope 升级不兼容** | 未来升级困难 | Monkey-patch 代码集中在一个文件中，升级时容易替换 |

---

## 七、总结

| 需求 | 推荐方案 | 复杂度 | 优先级 |
|------|---------|--------|--------|
| 时间硬规则 | Hook 保存 + 独立定时器任务 | 中 | P0 |
| BrainAgent 观察反思结果 | `brain_agent.observe()` 注入系统通知 | 低 | P0 |
| 上下文隔离 | 明确 BrainAgent/ReflectionAgent 各自的 observe 范围 | 低 | P1 |
| 思考过程轮数 | `post_reasoning` hook 计数器 | 低 | P1 |
| 流式生成截取 | 短期用 `post_reasoning` 结果；长期考虑 Monkey-patch | 高 | P2 |

**核心设计原则**：
1. **不阻断**：BrainAgent 的 ReAct 循环始终独立运行，中间汇报是"旁路"
2. **可观察**：BrainAgent 通过 `observe()` 知道自己已汇报的内容
3. **轻量级**：Hook 中只做保存，不做复杂计算，避免拖慢推理
4. **防御性**：定时器判断失败时默认"不汇报"，宁可错过也不打扰用户
