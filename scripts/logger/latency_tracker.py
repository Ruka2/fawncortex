"""
延迟追踪器（LatencyTracker）
==========================
解耦式性能监控模块，负责记录：
1. 各智能体独立响应时间
2. 前台/后台端到端时间（智能体角度）
3. 用户角度：输入 → 首次听到语音的延迟

使用方式：
    tracker = LatencyTracker()
    tracker.start_round(1, "你好")
    # ... 智能体执行时自动记录 ...
    tracker.finish_round(clarification_option="ignore")
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentLatencyRecord:
    """单个智能体的耗时记录。"""
    agent_name: str
    node_type: str
    start_ts: float
    end_ts: float
    duration_s: float


@dataclass
class RoundLatencyReport:
    """单轮完整延迟报告。"""
    round_num: int
    user_input: str
    agent_records: list[AgentLatencyRecord] = field(default_factory=list)
    frontend_e2e_s: float = 0.0
    backend_e2e_s: float = 0.0
    user_perceived_s: float = 0.0

    def print_report(self) -> None:
        """打印本轮延迟报告。"""
        print(f"{'='*60}")
        print(f"[Latency Report] 第 {self.round_num} 轮 | 输入: {self.user_input}")

        # 1. 各智能体独立耗时
        print("[各智能体独立耗时]")
        for rec in self.agent_records:
            print(
                f"  {rec.agent_name:14s} ({rec.node_type:14s}): "
                f"{rec.duration_s:10.3f} s"
            )

        # 2. 端到端时间（智能体角度）
        print("[端到端时间]")
        print(f"  前台响应 (Agent视角): {self.frontend_e2e_s:10.3f} s")
        if self.backend_e2e_s > 0:
            print(f"  后台思考 (Agent视角): {self.backend_e2e_s:10.3f} s")

        # 3. 用户角度
        print("[用户角度]")
        print(f"  输入 → 首次听到语音: {self.user_perceived_s:10.3f} s")
        print(f"{'='*60}\n")


class LatencyTracker:
    """延迟追踪器。

    设计为可选依赖：TaskExecutor / OutputScheduler / main 中若传入了则计时，
    未传入则保持零开销。
    """

    def __init__(self) -> None:
        self._reports: list[RoundLatencyReport] = []
        self._current: Optional[RoundLatencyReport] = None
        self._user_input_ts: float = 0.0
        self._first_sound_marked: bool = False

    # -------------------------------------------------------------------------
    # 轮次生命周期
    # -------------------------------------------------------------------------
    def start_round(self, round_num: int, user_input: str) -> None:
        """新一轮用户输入到达时调用。"""
        self._user_input_ts = time.perf_counter()
        self._first_sound_marked = False
        self._current = RoundLatencyReport(
            round_num=round_num,
            user_input=user_input,
        )

    def finish_round(self, clarification_option: str = "ignore") -> None:
        """一轮所有节点执行完毕后调用，计算端到端并打印报告。

        Args:
            clarification_option: brain 输出的 clarification 选项，
                用于决定 deep_think 是否计入前台端到端。
        """
        if not self._current:
            return

        self._calc_e2e(clarification_option)
        self._reports.append(self._current)
        self._current.print_report()
        self._current = None

    # -------------------------------------------------------------------------
    # 事件记录
    # -------------------------------------------------------------------------
    def record_agent(
        self,
        agent_name: str,
        node_type: str,
        start_ts: float,
        end_ts: float,
    ) -> float:
        """记录单个智能体的执行耗时。

        Returns:
            耗时（毫秒）。
        """
        if not self._current:
            return 0.0
        duration = end_ts - start_ts
        self._current.agent_records.append(
            AgentLatencyRecord(
                agent_name=agent_name,
                node_type=node_type,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_s=duration,
            )
        )
        return duration

    def mark_first_sound(self) -> None:
        """标记本轮首次语音开始播放（幂等：每轮只记一次）。

        由 OutputScheduler 在第一个 TTS 任务开始执行时调用。
        """
        if not self._current or self._first_sound_marked or not self._user_input_ts:
            return
        self._first_sound_marked = True
        self._current.user_perceived_s = (
            time.perf_counter() - self._user_input_ts
        )

    # -------------------------------------------------------------------------
    # 内部计算
    # -------------------------------------------------------------------------
    def _calc_e2e(self, clarification_option: str) -> None:
        """根据节点类型和 clarification_option 计算端到端时间。"""
        frontend_start: Optional[float] = None
        frontend_end: float = 0.0
        backend_start: Optional[float] = None
        backend_end: float = 0.0

        for rec in self._current.agent_records:
            if rec.node_type in ("quick_chat", "emotion_action", "summary_chat"):
                if frontend_start is None:
                    frontend_start = rec.start_ts
                frontend_end = max(frontend_end, rec.end_ts)

            elif rec.node_type == "deep_think":
                if backend_start is None:
                    backend_start = rec.start_ts
                backend_end = max(backend_end, rec.end_ts)

                # 只有当 clarification_option != "ignore" 时，
                # deep_think 才计入前台端到端
                if clarification_option != "ignore":
                    if frontend_start is None:
                        frontend_start = rec.start_ts
                    frontend_end = max(frontend_end, rec.end_ts)

        if frontend_start:
            self._current.frontend_e2e_s = (
                frontend_end - frontend_start
            )
        if backend_start:
            self._current.backend_e2e_s = (
                backend_end - backend_start
            )
