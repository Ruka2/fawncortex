"""
任务计划（Task Plan）
=====================
定义任务编排中的节点（TaskNode）与计划（TaskPlan）数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# 可用的Agent集群任务列表
AGENT_LIST_FAST_CHAT = ["chat"]
AGENT_LIST_DEEP_THINK = ["brain"]


class NodeType(Enum):
    """任务节点类型。"""
    QUICK_CHAT = "quick_chat"          # 快速闲聊
    DEEP_THINK = "deep_think"          # 大脑深度思考（阻塞）
    EMOTION_ACTION = "emotion_action"  # 表情动作展示
    SUMMARY_CHAT = "summary_chat"      # 基于大脑洞察的总结回复


class ExecutionMode(Enum):
    """节点内多Agent执行模式。"""
    GATHER = "gather"      # 并发执行（asyncio.gather）
    SEQUENTIAL = "seq"     # 顺序执行


@dataclass
class TaskNode:
    """任务节点：描述一个执行步骤。

    Attributes:
        node_type: 节点类型。
        name: 可读名称。
        agent_names: 参与此节点的Agent名称列表。
        blocking: 是否阻塞后续节点（True=等待完成后再执行下一个）。
        mode: 多Agent执行模式。
        post_condition: 完成后的状态标识（供大脑反思判断）。
        metadata: 额外元数据。
    """
    node_type: NodeType
    name: str
    agent_names: list[str] = field(default_factory=list)
    blocking: bool = True
    mode: ExecutionMode = ExecutionMode.GATHER
    post_condition: Optional[str] = None
    # metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type.value,
            "name": self.name,
            "agent_names": self.agent_names,
            "blocking": self.blocking,
            "mode": self.mode.value,
            "post_condition": self.post_condition,
            # "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d) -> "TaskNode":
        """从简化输入构造 TaskNode。

        支持三种输入形式：
        1. 字符串: "quick_chat"
        2. 简化字典: {"node_type": "quick_chat", "blocking": false}
        3. 完整字典: {"node_type": "...", "name": "...", "agent_names": [...]}

        会根据 node_type 的硬规则自动填充缺失字段（name/agent_names/blocking/mode）。
        """
        # --- 统一输入格式：字符串 → 字典 ---
        if isinstance(d, str):
            d = {"node_type": d}
        elif not isinstance(d, dict):
            print(f"[WARN] TaskNode.from_dict 收到非法输入类型 {type(d)}，回退为 quick_chat")
            d = {"node_type": "quick_chat"}

        # --- NodeType 解析（含非法值兜底） ---
        node_type_str = d.get("node_type", "quick_chat")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            print(f"[WARN] 非法的 NodeType '{node_type_str}'，回退为 quick_chat")
            node_type = NodeType.QUICK_CHAT

        # --- ExecutionMode 解析（含非法值兜底） ---
        mode_str = d.get("mode", "gather")
        try:
            mode = ExecutionMode(mode_str)
        except ValueError:
            print(f"[WARN] 非法的 ExecutionMode '{mode_str}'，回退为 gather")
            mode = ExecutionMode.GATHER

        # --- 根据 node_type 硬规则填充默认值 ---
        defaults = {
            NodeType.QUICK_CHAT: {
                "name": "快速响应",
                "agent_names": ["qucik_chat"],
                "blocking": False,
                "mode": ExecutionMode.GATHER,
            },
            NodeType.DEEP_THINK: {
                "name": "深度思考",
                "agent_names": ["deep_think"],
                "blocking": True,
                "mode": ExecutionMode.GATHER,
            },
            NodeType.EMOTION_ACTION: {
                "name": "表情动作",
                "agent_names": ["emotion"],
                "blocking": False,
                "mode": ExecutionMode.GATHER,
            },
            NodeType.SUMMARY_CHAT: {
                "name": "总结回复",
                "agent_names": ["chat"],
                "blocking": True,
                "mode": ExecutionMode.GATHER,
            }
        }
        rule = defaults.get(node_type, defaults[NodeType.QUICK_CHAT])

        # --- 兼容 agent_name（单数字符串）→ agent_names（列表） ---
        agent_names = d.get("agent_names")
        if agent_names is None:
            agent_name = d.get("agent_name")
            if isinstance(agent_name, str):
                agent_names = [agent_name]
            else:
                agent_names = rule["agent_names"]

        # 最终取值优先级：传入值 > 硬规则默认值
        return cls(
            node_type=node_type,
            name=d.get("name") or rule["name"],
            agent_names=agent_names,
            blocking=d.get("blocking") if "blocking" in d else rule["blocking"],
            mode=mode,
            post_condition=d.get("post_condition"),
        )


@dataclass
class TaskPlan:
    """任务计划：有序的任务节点列表。

    Attributes:
        nodes: 任务节点列表。
        version: 计划版本号（大脑重排后递增）。
        source: 计划来源（orchestrator / brain_replan）。
    """
    nodes: list[TaskNode] = field(default_factory=list)
    version: int = 1
    source: str = "orchestrator"

    def is_done(self, index: int) -> bool:
        """判断给定索引是否已越界。"""
        return index >= len(self.nodes)

    def insert_after(self, index: int, nodes: list[TaskNode]) -> None:
        """在指定索引后插入新节点。"""
        for i, node in enumerate(nodes):
            self.nodes.insert(index + 1 + i, node)
        self.version += 1

    def truncate_from(self, index: int) -> None:
        """从指定索引开始截断（含）。"""
        self.nodes = self.nodes[:index]
        self.version += 1

    def clear(self) -> None:
        """清空所有节点。"""
        self.nodes.clear()
        self.version += 1

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskPlan":
        return cls(
            nodes=[TaskNode.from_dict(n) for n in d.get("nodes", [])],
            version=d.get("version", 1),
            source=d.get("source", "orchestrator"),
        )

    @classmethod
    def from_raw_list(
        cls,
        raw_nodes: list,
        version: int = 1,
        source: str = "orchestrator",
    ) -> "TaskPlan":
        """从 Orchestrator 返回的简化节点列表构造 TaskPlan。

        支持 raw_nodes 为字符串列表（如 ["quick_chat", "deep_think"]）
        或简化字典列表（如 [{"node_type": "quick_chat", "blocking": false}]）。
        """
        return cls(
            nodes=[TaskNode.from_dict(n) for n in raw_nodes],
            version=version,
            source=source,
        )
