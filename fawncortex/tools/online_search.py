"""
联网搜索工具
============
为智能体提供实时联网搜索信息的能力，
用于检索时效性新闻、实时动态、未知知识或需要最新数据的任务。
"""

import json
from http.client import HTTPSConnection

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

import config


HOST = "api.wsa.cloud.tencent.com"
URI = "/SearchPro"
TIMEOUT = 15


def online_search(query: str) -> ToolResponse:
    """使用搜索引擎查询互联网上的实时信息。

    当用户询问涉及最新新闻、时事动态、实时数据、
    未知知识或需要联网获取最新内容时，调用此工具进行搜索。

    Args:
        query: 搜索关键词或问题，例如 "今天的北京天气"、
            "最新人工智能新闻"、"某部电影的上映时间"
    """
    secret_key = config.TENCENT_CLOUD_WSA_APIKEY
    if not secret_key:
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text="联网搜索配置错误：未设置 TENCENT_CLOUD_WSA_APIKEY，请在环境变量中配置。"
            )]
        )

    payload = json.dumps({"Query": query}, ensure_ascii=False)
    headers = {
        "Authorization": "Bearer " + secret_key,
        "Content-Type": "application/json; charset=utf-8",
    }

    try:
        conn = HTTPSConnection(HOST, timeout=TIMEOUT)
        conn.request("POST", URI, body=payload.encode("utf-8"), headers=headers)
        resp = conn.getresponse()
        raw_body = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=f"联网搜索请求失败，HTTP 状态码：{resp.status}，响应：{raw_body}"
                )]
            )

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"搜索返回非 JSON 数据：{raw_body}")]
            )

        # 尝试解析腾讯云 WSA 的常见响应结构
        result_text = _format_search_results(data)
        return ToolResponse(
            content=[TextBlock(type="text", text=result_text)]
        )

    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"联网搜索调用异常：{e}")]
        )


def _format_search_results(data: dict) -> str:
    """将 WSA API 返回的 JSON 格式化为易读的文本。"""
    # 处理错误返回
    if isinstance(data, dict) and data.get("Error"):
        err = data["Error"]
        return f"搜索服务返回错误：{err}"

    # 某些版本返回在 Response / Result 字段里
    results = None
    if isinstance(data, dict):
        if "Response" in data:
            inner = data["Response"]
            # 尝试找到结果列表的常见键名
            for key in ("Results", "Result", "Data", "Items", "Answer", "SearchResult"):
                if key in inner and inner[key] is not None:
                    results = inner[key]
                    break
            # 如果都没命中，把整个 inner 当结果
            if results is None:
                results = inner
        else:
            # 扁平结构，直接找常见键
            for key in ("Results", "Result", "Data", "Items", "Answer", "SearchResult"):
                if key in data and data[key] is not None:
                    results = data[key]
                    break
            if results is None:
                results = data

    if results is None:
        return f"搜索返回数据异常，原始响应：{json.dumps(data, ensure_ascii=False)}"

    # 如果是字符串（例如直接返回 Answer），直接返回
    if isinstance(results, str):
        return results

    # 尝试格式化为列表
    lines = []
    if isinstance(results, list):
        for idx, item in enumerate(results, 1):
            if isinstance(item, dict):
                title = item.get("Title") or item.get("title") or ""
                url = item.get("Url") or item.get("url") or item.get("Link") or item.get("link") or ""
                summary = item.get("Summary") or item.get("summary") or item.get("Content") or item.get("content") or item.get("Snippet") or item.get("snippet") or ""
                # 拼接字段
                parts = []
                if title:
                    parts.append(f"[{idx}] {title}")
                if url:
                    parts.append(f"链接：{url}")
                if summary:
                    parts.append(f"摘要：{summary}")
                if not parts:
                    parts.append(f"[{idx}] {json.dumps(item, ensure_ascii=False)}")
                lines.append("\n".join(parts))
            else:
                lines.append(f"[{idx}] {item}")
    elif isinstance(results, dict):
        # 如果是字典，尝试取常见摘要字段
        answer = results.get("Answer") or results.get("answer") or results.get("ResponseText") or results.get("responseText")
        if answer:
            lines.append(answer)
        else:
            # 直接格式化整个字典
            for k, v in results.items():
                lines.append(f"{k}：{v}")
    else:
        lines.append(str(results))

    if not lines:
        return f"搜索未返回有效结果，原始响应：{json.dumps(data, ensure_ascii=False)}"

    return "\n\n".join(lines)
