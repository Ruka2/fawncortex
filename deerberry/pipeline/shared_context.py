"""
公共信息域（SharedContext）
===========================
多智能体/多异步任务共享的认知资产与任务状态管理中心。
通俗简介就是用于大脑智能体与闲聊智能体异步通信时的信息域共享
"""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SharedContextData:
    """公共信息域的数据结构。"""
    version: int = 0
    # --- 任务编排字段 ---
    clarification_needed: bool = False    # 是否需要澄清修改任务队列节点
    clarification_option: str = ""        # 此处是重点控制逻辑之一，目前先不ignore
    clarification_reason: str = ""        # 此次任务澄清原因
    # --- 用户信息缓存字段 ---
    user_profile: str = ""      # 用户画像要点
    user_emotion: str = ""      # 用户当前情绪
    user_intent: str = ""       # 用户当前意图
    # --- 对话策略字段 ---
    suggested_dialogue_strategy: str = ""  # 建议的对话策略方向
    suggested_emotion: str = ""            # 建议的表情
    # --- 长期记忆 ---
    retrieved_memories: list[str] = field(default_factory=list)  # 大脑检索到的相关长期记忆列表

class SharedContext:
    """大脑Agent生产的洞察 + 任务编排信号，供前台Agent消费。"""

    def __init__(self):
        self._data = SharedContextData()
        self._lock = asyncio.Lock()

    async def update(self, version: int, **kwargs) -> None:
        """大脑Agent完成推理后，将结果写入公共域。

        Args:
            version: 对应轮次。
            **kwargs: 支持的字段：...
        """
        async with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._data, k):
                    setattr(self._data, k, v)
            self._data.version = version

    def peek(self) -> dict:
        """只读查看当前数据（调试用）。"""
        return self._data.__dict__.copy()

    async def clear(self) -> None:
        """清空公共域数据（新一轮开始时调用）。"""
        async with self._lock:
            self._data = SharedContextData()



    # --- 便捷属性访问 ---
    @property
    def suggested_emotion(self) -> str:
        return self._data.suggested_emotion

    @property
    def replan_requested(self) -> bool:
        return self._data.clarification_option == "replan"

    @property
    def ignore_requested(self) -> bool:
        return self._data.clarification_option == "ignore"