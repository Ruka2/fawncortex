"""
arXiv MCP 工具封装
==================
通过 StdIOStatefulClient 连接本地 arxiv-mcp-server，
提供论文搜索、下载、读取等能力供 BrainAgent 使用。

依赖：
    pip install arxiv-mcp-server

已注册工具（共10个）：
    - search_papers      搜索 arXiv 论文（支持日期范围、分类筛选）
    - download_paper     下载指定论文到本地存储
    - list_papers        列出所有已下载的论文
    - read_paper         读取已下载论文的全文内容
    - get_abstract       获取论文摘要和元数据（不下载全文）
    - semantic_search    对已下载论文做语义相似度搜索
    - reindex            重建本地语义索引
    - citation_graph     查看论文引用关系图谱
    - watch_topic        订阅研究主题（持续追踪）
    - check_alerts       检查订阅主题的新发表论文

使用方式（在 main5_planqueue.py 中）：
    from deerberry.tools.arxiv_search import (
        create_arxiv_client,
        register_arxiv_tools,
        close_arxiv_client,
    )

    # 初始化时连接
    await create_arxiv_client(storage_path="./data/arxiv_papers")

    # 注册到 BrainAgent 的 Toolkit
    toolkit = Toolkit()
    toolkit.register_tool_function(retrieve_from_memory)
    toolkit.register_tool_function(record_to_memory)
    await register_arxiv_tools(toolkit)   # <-- 混入 MCP 工具

    brain_agent = BrainAgent(model=..., long_term_memory=..., toolkit=toolkit)

    # 程序退出时关闭（注意：多个 StatefulClient 必须按 LIFO 顺序关闭）
    await close_arxiv_client()
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentscope.tool import Toolkit
from agentscope.mcp import StdIOStatefulClient

# 模块级单例：arxiv MCP client
_arxiv_client: StdIOStatefulClient | None = None


async def create_arxiv_client(
    storage_path: str | None = None,
) -> StdIOStatefulClient:
    """创建并连接 arxiv-mcp-server 的 StdIO Stateful Client。

    Args:
        storage_path: 论文本地存储路径，默认 ./data/tmp/mcp_arxiv_files

    Returns:
        已连接的 StdIOStatefulClient 实例
    """
    global _arxiv_client

    # 使用默认路径（项目目录下统一管理）
    if storage_path is None:
        storage_path = "./data/tmp/mcp_arxiv_files"

    # 自动创建存储目录（避免 arxiv-mcp-server 启动时报目录不存在）
    os.makedirs(storage_path, exist_ok=True)

    args: list[str] = []
    if storage_path:
        args.extend(["--storage-path", storage_path])

    _arxiv_client = StdIOStatefulClient(
        name="arxiv_mcp",
        command="arxiv-mcp-server",
        args=args,
    )
    await _arxiv_client.connect()
    return _arxiv_client


def get_arxiv_client() -> StdIOStatefulClient:
    """获取已创建的 arxiv client。

    Raises:
        RuntimeError: 如果 client 尚未创建
    """
    if _arxiv_client is None:
        raise RuntimeError(
            "arxiv client 尚未创建，请先调用 create_arxiv_client()"
        )
    return _arxiv_client


async def register_arxiv_tools(
    toolkit: Toolkit,
    group_name: str = "basic",
    execution_timeout: float = 60.0,
) -> None:
    """将 arxiv MCP 工具批量注册到指定的 AgentScope构造出来的智能体中的Toolkit中。

    Args:
        toolkit: AgentScope 的 Toolkit 实例（可以已有其他工具）
        group_name: 工具分组名，默认 "basic"（混入现有工具组）。
                    如需独立分组，先调用 toolkit.create_tool_group("arxiv", "...")
        execution_timeout: 单个 MCP 工具调用的超时时间（秒），默认 60 秒。
                           防止网络慢或论文过大时 ReActAgent 的 tool call loop 卡住。
    """
    client = get_arxiv_client()
    await toolkit.register_mcp_client(
        client,
        group_name=group_name,
        execution_timeout=execution_timeout,
    )

    # schemas = toolkit.get_json_schemas()
    # print(f"[MCP] arxiv 工具已注册，当前 Toolkit 共 {len(schemas)} 个工具")
    # for schema in schemas:
    #     func_name = schema.get("function", {}).get("name", "unknown")
    #     # 只打印 arxiv 相关工具（避免把其他 group 的工具也打印出来造成混淆）
    #     if func_name in {
    #         "search_papers",
    #         "download_paper",
    #         "list_papers",
    #         "read_paper",
    #         "get_abstract",
    #         "semantic_search",
    #         "reindex",
    #         "citation_graph",
    #         "watch_topic",
    #         "check_alerts",
    #     }:
    #         print(f"       📄 {func_name}")


async def close_arxiv_client() -> None:
    """关闭 arxiv MCP client。

    注意：如果同时使用了多个 StatefulClient（如文件系统 MCP + arxiv MCP），
    必须按 **LIFO（后开先关）** 顺序关闭，否则可能报错。
    """
    global _arxiv_client
    if _arxiv_client is not None:
        await _arxiv_client.close()
        # print("[MCP] arxiv-mcp-server 已关闭")
        _arxiv_client = None


async def list_arxiv_tools() -> list:
    """列出 arxiv MCP server 提供的所有工具（调试用）。

    Returns:
        MCP Tool 对象列表
    """
    client = get_arxiv_client()
    tools = await client.list_tools()
    print(f"[MCP] arxiv 可用工具列表（共 {len(tools)} 个）：")
    for tool in tools:
        desc = (tool.description or "")[:60]
        print(f"       - {tool.name}: {desc}...")
    return tools


# ---------------------------------------------------------------------------
# 独立测试入口
# ---------------------------------------------------------------------------

async def _test() -> None:
    """独立测试：连接 arxiv MCP server 并打印工具信息。"""
    import json

    # 1. 创建并连接
    await create_arxiv_client()

    # 2. 列出工具
    await list_arxiv_tools()

    # 3. 注册到 Toolkit（混入已有工具的场景）
    tk = Toolkit()
    await register_arxiv_tools(tk)

    # 4. 查看 JSON schemas（LLM 可见的工具描述）
    schemas = tk.get_json_schemas()
    print(f"\n[MCP Test] JSON Schemas（共 {len(schemas)} 个）：")
    print(json.dumps(schemas, indent=2, ensure_ascii=False, default=str))

    # 5. 关闭
    await close_arxiv_client()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test())
