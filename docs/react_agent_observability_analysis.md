# ReActAgent 可观测性分析（推理过程打印位置）

> 分析时间：2026-05-06  
> 关联文件：`deerberry/agent/brain_agent.py`、AgentScope `ReActAgent` 源码

---

## 一、ReAct 循环结构

`BrainAgent.think()` 调用 `ReActAgent.reply()`，其内部逻辑如下：

```
reply(msg) 入口
    │
    ├── 检索长期记忆 _retrieve_from_long_term_memory()
    ├── 检索知识库 _retrieve_from_knowledge()
    │
    └── for iter in range(max_iters=10):   ← 【循环节点】
            │
            ├── _compress_memory_if_needed()
            │
            ├── _reasoning(tool_choice)    ← 【推理节点】
            │       └── 调用 LLM → 返回 msg（含 text + tool_use）
            │
            ├── _acting(tool_call) × N     ← 【行动节点】
            │       └── 并行/串行执行工具 → 返回 tool_result
            │
            └── 检查退出条件               ← 【决策节点】
                    ├── 无 tool_use → break（完成任务）
                    └── 有 tool_use → 继续下一轮
    │
    └── _summarizing()                     ← 【兜底节点】
            └── max_iters 耗尽时的总结回复
```

---

## 二、可观测性分层

### 🔴 第一层：应用层（BrainAgent 侧）

位置在 `brain_agent.py` 中，**不涉及第三方库源码**，稳定性最高。

| 位置 | 代码行 | 适合打印的内容 | 当前状态 |
|------|--------|---------------|----------|
| `think()` 调用前 | ~120 | 每轮思考的触发信号、用户输入摘要、当前上下文长度 | ❌ 无 |
| `think()` 调用后 | ~123-129 | **已有**：最终洞察文本（`[LLM OUTPUT] Agent: brain_center`） | ✅ 有 |
| `think()` 返回前 | ~136 | 本轮检索到的记忆列表、洞察生成耗时 | ❌ 无 |

**痛点**：当前 `think()` 中只有一个"最终输出"打印，中间 1~10 轮的推理-行动循环完全黑盒。

### 🟡 第二层：框架层（ReActAgent Hook 侧）

AgentScope `ReActAgentBase` 原生支持 **10 种 Hook**，可在 `BrainAgent.__init__` 中通过 `register_instance_hook()` 注册。

| Hook 类型 | 触发时机 | 适合打印的内容 |
|-----------|---------|---------------|
| `pre_reply` | `reply()` 开始时 | BrainAgent 开始深度思考的信号、用户 query |
| `post_reply` | `reply()` 结束时 | 总耗时、迭代次数、最终答案预览 |
| `pre_reasoning` | 每轮 `_reasoning()` 调用 LLM 前 | **第 X 轮推理开始**、当前 memory 中的工具历史摘要 |
| `post_reasoning` | LLM 返回后 | LLM 的 text 输出（思考内容）、**决定调用的工具列表及参数** |
| `pre_acting` | 每个工具执行前 | **正在执行的工具名、传入参数** |
| `post_acting` | 每个工具返回后 | **工具返回结果摘要**（长度/状态） |
| `pre_print` | `self.print()` 调用前 | （一般无需额外打印） |
| `post_print` | `self.print()` 调用后 | （一般无需额外打印） |

**核心观察点**：
- `pre_reasoning` + `post_reasoning` = 能看到 LLM 的"内心独白"和决策
- `pre_acting` + `post_acting` = 能看到工具的调用链路

### 🟢 第三层：循环内部（需轻微侵入，不推荐直接改源码）

`ReActAgent.reply()` 的 `for` 循环（源码第 432 行）内部适合打印：
- 当前迭代次数：`iter 3/10`
- 退出原因：是"无 tool_use 正常结束"还是"达到 max_iters 被迫总结"

但此位置在 AgentScope 第三方库中，**不建议直接修改源码**。应通过 `post_reply` hook 在结束后反推迭代次数。

---

## 三、当前打印盲区示例

以一次典型调用为例，**你现在能看到 vs 看不到**的信息：

```
用户输入："帮我找 Transformer 论文"

[你能看到的]          [你看不到的 ─── ReAct 黑盒]
     │                          │
     ▼                          ▼
[think() 调用] ──────→  iter 1: reasoning()
     │                    LLM 思考："用户需要论文，我应该先搜索"
     │                    LLM 决定调用 search_papers(query="Transformer")
     │                          │
     │                          ▼
     │                    acting() 执行 search_papers
     │                    返回 5 篇论文
     │                          │
     │                          ▼
     │                    iter 2: reasoning()
     │                    LLM 思考："找到了 paperId，我应该读取第一篇"
     │                    LLM 决定调用 read_paper("arXiv:1706.03762")
     │                          │
     │                          ▼
     │                    acting() 执行 read_paper
     │                    返回 10 页文本
     │                          │
     │                          ▼
     │                    iter 3: reasoning()
     │                    LLM 思考："已获取内容，我可以生成总结了"
     │                    LLM 输出最终文本（无 tool_use）
     │                          │
     ▼                          │
[最终洞察打印] ◄────────────────┘
"[LLM OUTPUT] Agent: brain_center..."
```

**问题**：中间的 iter 1/2/3 发生了什么、调用了什么工具、工具返回了什么，全部不可见。当 ReAct 循环卡住或异常时，无法定位是哪一步出了问题。

---

## 四、推荐的 Hook 注册方案

### 最小可行方案（3 个 hook）

在 `BrainAgent.__init__` 中注册：

```python
self.agent.register_instance_hook("post_reasoning", self._on_post_reasoning)
self.agent.register_instance_hook("pre_acting", self._on_pre_acting)
self.agent.register_instance_hook("post_acting", self._on_post_acting)
```

预期输出示例：

```
[ReAct iter 2/10] 💭 推理结果
思考：用户提到Transformer，我先搜索相关论文
决定调用：search_papers(query="Transformer architecture", limit=5)

[ReAct iter 2/10] 🔧 执行工具: search_papers
参数: {"query": "Transformer architecture", "limit": 5}

[ReAct iter 2/10] 📥 工具返回: search_papers
状态: 成功 | 结果长度: 3200 字符 | 包含 5 篇论文
```

### 完整方案（5 个 hook）

再加上 `pre_reply` 和 `post_reply`：

```python
self.agent.register_instance_hook("pre_reply", self._on_pre_reply)
self.agent.register_instance_hook("post_reply", self._on_post_reply)
```

---

## 五、Hook 函数签名

```python
# pre_* hook: (self, kwargs: dict) -> kwargs | None
async def pre_reasoning_hook(self, kwargs):
    print(f"[ReAct] 🧠 第 X 轮推理开始...")
    return kwargs

# post_* hook: (self, kwargs: dict, output: Any) -> output | None
async def post_reasoning_hook(self, kwargs, output):
    # output 是 Msg 对象，含 content blocks
    print(f"[ReAct] 💭 推理结果: ...")
    return output

# pre_acting: kwargs 中包含 tool_call（ToolUseBlock）
async def pre_acting_hook(self, kwargs):
    tool_call = kwargs["tool_call"]
    print(f"[ReAct] 🔧 准备调用: {tool_call['name']}")
    return kwargs

# post_acting: output 是工具执行结果（dict | None）
async def post_acting_hook(self, kwargs, output):
    print(f"[ReAct] 📥 工具返回: ...")
    return output
```

---

## 六、总结

| 你想观察什么 | 推荐位置 | 实现方式 |
|-------------|---------|----------|
| 每轮循环次数 | `pre_reasoning` hook | `register_instance_hook` |
| LLM 思考内容 | `post_reasoning` hook | 解析 output Msg 的 text block |
| 调用哪些工具 | `post_reasoning` hook | 解析 output Msg 的 tool_use blocks |
| 工具参数 | `pre_acting` hook | 解析 kwargs["tool_call"] |
| 工具返回 | `post_acting` hook | 解析 kwargs + output |
| 总耗时/轮次 | `post_reply` hook | 在 BrainAgent 中计时 |
| 最终洞察 | `think()` 已有打印 | 已存在 ✅ |

**最优先建议**：注册 `post_reasoning` + `pre_acting` + `post_acting` 三个 hook，即可完整覆盖 ReAct 循环的"决策 → 执行 → 观察"链条。
