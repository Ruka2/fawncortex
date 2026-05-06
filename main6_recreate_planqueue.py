
# 项目路径根目录定位
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 模型配置表
import config

# 核心基础依赖
import asyncio
import time
from typing import Any, Callable
from collections import defaultdict

# AgentScope 基础依赖
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.pipeline import FanoutPipeline, MsgHub

# 自定义智能体依赖
from deerberry.agent.chat_agent import ChatAgent
from deerberry.agent.emotion_agent import EmotionAgent
from deerberry.agent.brain_agent import BrainAgent

# 自定义智能体记忆的实例类
from deerberry.base.memory import create_long_term_memory

# 外部引用工具
from deerberry.tools.search_memory import (
    set_memory_manager,
    retrieve_from_memory,
    record_to_memory,
)
from deerberry.tools.arxiv_search import (
    create_arxiv_client,
    register_arxiv_tools,
    close_arxiv_client,
)

# 外部非智能体执行工具
from deerberry.components.voice.tts import SiliconFlowCosyVoice

# 日志打印代码
from deerberry.logger.latency_tracker import LatencyTracker
from deerberry.logger.logger import enable_file_logging


# 基础用户配置
AGENT_NAME = "Ruka"
USER_NAME = "鹿过"


# =============================================================================
# 事件驱动基础设施（参考 PROJECT_ANALYSIS_AND_ROADMAP.md 3.1.1）
# =============================================================================

class UserInputEvent:
    """用户输入事件：由 MessageBus 广播给所有订阅 Agent。"""
    def __init__(self, msg: Msg) -> None:
        self.topic = "user.input"
        self.msg = msg
        self.timestamp = time.time()


class AgentResponseEvent:
    """Agent 响应事件：Agent 完成处理后发布到总线。"""
    def __init__(self, agent_name: str, result: Any, elapsed: float) -> None:
        self.topic = "agent.response"
        self.agent_name = agent_name
        self.result = result
        self.elapsed = elapsed
        self.timestamp = time.time()


class MessageBus:
    """轻量级消息总线（Pub-Sub）。

    参考 PROJECT_ANALYSIS_AND_ROADMAP.md 3.2.1 设计：
    - 支持通配符订阅（如 "user.*"、"agent.*"）
    - 所有事件通过 asyncio.Queue 异步分发
    - 每个订阅者以独立 Task 执行，互不阻塞
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True

    async def subscribe(self, pattern: str, callback: Callable) -> None:
        """订阅特定 topic 模式。"""
        self._subscribers[pattern].append(callback)

    async def publish(self, event: Any) -> None:
        """发布事件到总线队列。"""
        await self._queue.put(event)

    def _match(self, pattern: str, topic: str) -> bool:
        """简单的通配符匹配：'*' 匹配任意后缀。"""
        if pattern.endswith(".*"):
            return topic.startswith(pattern[:-1])
        return pattern == topic

    async def run(self) -> None:
        """总线主循环：常驻后台，持续分发事件。"""
        while self._running:
            event = await self._queue.get()
            topic = getattr(event, "topic", "")
            for pattern, callbacks in self._subscribers.items():
                if self._match(pattern, topic):
                    for cb in callbacks:
                        asyncio.create_task(cb(event))

    async def stop(self) -> None:
        self._running = False


# =============================================================================
# AgentScope Pipeline 风格的并行执行器
# =============================================================================

class ParallelFanoutPipeline:
    """基于 AgentScope FanoutPipeline 思想的并行执行器。

    与标准 FanoutPipeline(enable_gather=True) 的区别：
    - 标准版使用 asyncio.gather()，等待所有 Agent 完成后统一返回；
    - 本版本使用 asyncio.as_completed()，支持"谁先完成谁先输出"的流式感知。

    参考文档：https://doc.agentscope.io/zh_CN/tutorial/task_pipeline.html
    """

    def __init__(self, agents: dict[str, Any]) -> None:
        """Args:
            agents: 字典，key 为 Agent 标识名，value 为 Agent 实例。
        """
        self.agents = agents

    async def run(self, msg: Msg, bus: MessageBus | None = None) -> None:
        """将同一条消息并行分发给所有 Agent，并通过 as_completed 实现谁先输出谁先打印。

        Args:
            msg: 用户输入消息（AgentScope Msg）。
            bus: 可选的消息总线，Agent 完成后会发布 AgentResponseEvent。
        """

        async def _call_agent(name: str, agent: Any, msg: Msg) -> AgentResponseEvent:
            start_ts = time.perf_counter()
            try:
                # 所有 Agent 统一调用 reply() 接口（AgentScope AgentBase 标准接口）
                result = await agent.reply(msg)
            except Exception as exc:
                # 异常隔离：单个 Agent 失败不影响其他 Agent
                result = Msg(name=name, content=f"[ERROR] {exc}", role="assistant")
            elapsed = time.perf_counter() - start_ts
            return AgentResponseEvent(agent_name=name, result=result, elapsed=elapsed)

        # 1. 创建并发任务：所有 Agent 同时开始处理同一条消息
        tasks = [
            asyncio.create_task(_call_agent(name, agent, msg))
            for name, agent in self.agents.items()
        ]

        # 2. 使用 as_completed 实现"谁先输出谁先打印"
        for coro in asyncio.as_completed(tasks):
            event = await coro
            self._print_response(event)
            if bus is not None:
                await bus.publish(event)

    @staticmethod
    def _print_response(event: AgentResponseEvent) -> None:
        """打印单个 Agent 的响应结果。"""
        name = event.agent_name
        elapsed = event.elapsed
        result = event.result

        # 统一提取文本内容
        if isinstance(result, Msg):
            text = result.get_text_content() or str(result.content)
        else:
            text = str(result)

        # 格式化输出
        bar = "━" * 50
        print(f"\n{bar}")
        print(f"🤖 [{name}] 响应完成  |  耗时: {elapsed:.3f}s")
        print(f"{bar}")
        print(text)
        print(f"{bar}\n")


# =============================================================================
# 主程序
# =============================================================================

async def main() -> None:
    # 启用终端日志收集（所有 print 自动写入 logs/ 目录）
    enable_file_logging()

    # 初始化延迟追踪器
    latency_tracker = LatencyTracker()

    # 初始化 TTS
    tts = SiliconFlowCosyVoice(
        api_key=config.TTS_API_KEY,
        api_url=config.TTS_BASE_URL,
        model=config.TTS_MODEL_NAME,
        voice=config.TTS_VOICE,
    )
    print(f"[init] TTS 已创建: {config.TTS_MODEL_NAME}, {config.TTS_VOICE}")

    # 初始化长期记忆（使用 memory 角色专用配置）
    memory_cfg = config.LLM_ROLE_CONFIG.get("memory", {})
    long_term_memory = create_long_term_memory(
        agent_name=AGENT_NAME,
        user_name=USER_NAME,
        vector_store_path=config.MEM0_VECTOR_STORE_PATH,
        history_db_path=config.MEM0_HISTORY_DB_PATH,
        llm_model_name=memory_cfg.get("model_name") or config.LLM_MODEL_NAME,
        llm_api_key=memory_cfg.get("api_key") or config.LLM_API_KEY,
        llm_base_url=memory_cfg.get("base_url") or config.LLM_BASE_URL,
        llm_generate_kwargs=config.LLM_ROLE_GENERATE_KWARGS.get("memory"),
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        embedding_api_key=config.EMBEDDING_API_KEY,
        embedding_base_url=config.EMBEDDING_BASE_URL,
    )
    set_memory_manager(long_term_memory)
    print(
        f"[init] 长期记忆已初始化: {config.MEM0_HISTORY_DB_PATH}, "
        f"Memory LLM: {memory_cfg.get('model_name') or config.LLM_MODEL_NAME}, "
        f"Embedding Model: {config.EMBEDDING_MODEL_NAME}"
    )

    # 按智能体角色创建专用大模型实例
    def build_model_for_role(role: str, stream: bool = True) -> OpenAIChatModel:
        """根据 config.LLM_ROLE_CONFIG 中的角色映射创建 OpenAIChatModel。"""
        cfg = config.LLM_ROLE_CONFIG.get(role, {})
        model_name = cfg.get("model_name") or config.LLM_MODEL_NAME
        api_key = cfg.get("api_key") or config.LLM_API_KEY
        base_url = cfg.get("base_url") or config.LLM_BASE_URL
        generate_kwargs = config.LLM_ROLE_GENERATE_KWARGS.get(role, {})

        return OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs=generate_kwargs,
        )

    chat_model = build_model_for_role("chat", stream=config.STREAM)
    emotion_model = build_model_for_role("emotion", stream=config.STREAM)
    brain_model = build_model_for_role("brain", stream=config.STREAM)

    # 打印各角色模型映射信息
    print("[init] 多角色 LLM 配置映射:")
    for role, model in [
        ("chat", chat_model),
        ("emotion", emotion_model),
        ("brain", brain_model),
    ]:
        cfg = config.LLM_ROLE_CONFIG.get(role, {})
        used_model = cfg.get("model_name") or config.LLM_MODEL_NAME
        used_base = cfg.get("base_url") or config.LLM_BASE_URL
        print(f"       {role:15s} model={used_model}, base_url={used_base}")

    # 记忆系统配置映射
    memory_used_model = memory_cfg.get("model_name") or config.LLM_MODEL_NAME
    memory_used_base = memory_cfg.get("base_url") or config.LLM_BASE_URL
    print(f"       {'memory':15s} model={memory_used_model}, base_url={memory_used_base}")

    # 组装 BrainAgent 的 Toolkit
    toolkit = Toolkit()
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
    schemas = toolkit.get_json_schemas()
    print(f"[init] Brain Agent Toolkit 已组装，共 {len(schemas)} 个工具:")
    for s in schemas:
        name = s.get("function", {}).get("name", "unknown")
        print(f"       {name}")

    # =============================================================================
    # 初始化核心智能体（Event-Driven 模式：取消 Orchestrator + TaskExecutor）
    # =============================================================================
    chat_agent = ChatAgent(model=chat_model, agent_name=AGENT_NAME)
    emotion_agent = EmotionAgent(model=emotion_model)
    brain_agent = BrainAgent(
        model=brain_model,
        long_term_memory=long_term_memory,
        toolkit=toolkit,
    )

    agents = {
        "ChatAgent": chat_agent,
        "EmotionAgent": emotion_agent,
        "BrainAgent": brain_agent,
    }
    print(f"[init] 核心智能体集群已创建: {list(agents.keys())}")

    # 初始化事件总线（MessageBus）
    bus = MessageBus()
    bus_task = asyncio.create_task(bus.run())
    print("[init] MessageBus 事件总线已启动")

    # 初始化 AgentScope Pipeline 风格的并行扇出执行器
    pipeline = ParallelFanoutPipeline(agents=agents)
    print("[init] ParallelFanoutPipeline 并行扇出管道已创建")

    # =============================================================================
    # 主循环：事件驱动的并行响应
    # =============================================================================
    round_num = 0

    try:
        while True:
            try:
                user_input = (
                    await asyncio.get_event_loop().run_in_executor(None, input, "")
                ).strip()

                if not user_input:
                    continue

                round_num += 1
                msg = Msg(name="user", content=user_input, role="user")
                print(f"\n{'='*60}")
                print(f"🚀 第 {round_num} 轮  |  用户输入: {user_input}")
                print(f"{'='*60}")

                # 开始新一轮计时
                latency_tracker.start_round(round_num, user_input)
                round_start = time.perf_counter()

                # ── 事件驱动并行扇出：1 条消息同时广播给所有 Agent ──
                # 设计说明：
                #   - ChatAgent   : System 1 极速轨道（< 1.5s），负责前台对话
                #   - EmotionAgent: System 1 极速轨道，负责表情驱动
                #   - BrainAgent  : System 2 后台轨道，负责深度思考/策略/记忆
                # 三个 Agent 并发执行，通过 as_completed 实现"谁先完成谁先打印"。
                await pipeline.run(msg=msg, bus=bus)

                round_elapsed = time.perf_counter() - round_start
                print(f"[Round] 第 {round_num} 轮全部完成，总耗时: {round_elapsed:.3f}s")

                # 延迟追踪记录
                latency_tracker.record_agent(
                    agent_name="round_total",
                    node_type="round",
                    start_ts=round_start,
                    end_ts=time.perf_counter(),
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"❌ 第 {round_num} 轮发生未捕获异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"💥 程序主循环发生致命异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Shutdown] 正在关闭事件总线...")
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass
        print("[Shutdown] 事件总线已关闭")
        print("[Shutdown] 正在关闭 MCP 服务...")
        await close_arxiv_client()
        print("[Shutdown] MCP 服务已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Logger] 程序已正常中断")
    finally:
        import os
        from deerberry.logger.logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        os._exit(0)
