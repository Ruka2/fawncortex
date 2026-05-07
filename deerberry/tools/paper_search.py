"""
Semantic Scholar 学术论文搜索工具集
====================================
封装 Semantic Scholar Academic Graph API，提供论文/作者搜索、详情查询、
引用与参考文献检索等功能，适合 Agent 在学术讨论场景下调用。

官方文档：https://api.semanticscholar.org/api-docs/

使用方式（同步工具函数）：
    from deerberry.tools.paper_search import search_papers, get_paper_details
    toolkit.register_tool_function(search_papers)
    toolkit.register_tool_function(get_paper_details)

API Key 从项目根目录 config.py 的 S2_API_KEY 读取（默认读取环境变量 S2_API_KEY）。
"""

import json
import urllib.request
import urllib.parse
import tempfile
import os
from typing import Optional

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

try:
    import fitz  # pymupdf
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from config import S2_API_KEY


# =============================================================================
# 配置
# =============================================================================

_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# 默认返回字段（足够 Agent 理解论文基本信息）
_DEFAULT_PAPER_FIELDS = (
    "paperId,title,authors,year,abstract,citationCount,venue,url,tldr"
)
_DEFAULT_PAPER_DETAIL_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,referenceCount,"
    "references,citations,venue,url,tldr,fieldsOfStudy,isOpenAccess,openAccessPdf"
)
_DEFAULT_CITE_FIELDS = (
    "paperId,title,authors,year,venue,url,abstract"
)
_DEFAULT_AUTHOR_FIELDS = (
    "authorId,name,aliases,affiliations,paperCount,citationCount,hIndex,url"
)


# =============================================================================
# 内部请求工具
# =============================================================================

def _build_request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if S2_API_KEY:
        req.add_header("x-api-key", S2_API_KEY)
    return req


def _fetch_json(url: str) -> dict:
    """执行 GET 请求并返回 JSON。"""
    req = _build_request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_params(params: dict) -> str:
    """将字典编码为 URL 查询字符串，过滤 None 值。"""
    filtered = {k: v for k, v in params.items() if v is not None}
    return urllib.parse.urlencode(filtered, doseq=True)


def _format_error(action: str, exc: Exception) -> ToolResponse:
    """统一格式化异常返回。"""
    return ToolResponse(
        content=[TextBlock(type="text", text=f"{action} 失败：{exc}")]
    )


# =============================================================================
# 工具函数（Agent 可调用的接口）
# =============================================================================

def search_papers(
    query: str,
    limit: int = 5,
    fields: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sort: str = "relevance",
) -> ToolResponse:
    """在 Semantic Scholar 中按关键词搜索学术论文。

    适用场景：用户提到某个研究主题、技术方向或问题，需要查找相关论文。

    Args:
        query: 搜索关键词或自然语言查询。支持布尔逻辑，例如：
               "transformer architecture"、"GPT-3"、"attention mechanism"
        limit: 返回结果数量上限（1~100，默认 5）。建议首次搜索设为 5~10。
        fields: 逗号分隔的返回字段列表。默认返回：
                paperId,title,authors,year,abstract,citationCount,venue,url,tldr
                完整可选字段见官方文档，常用扩展：
                - externalIds, isOpenAccess, openAccessPdf, fieldsOfStudy
                - authors.name, authors.affiliations
        year_start: 起始发表年份筛选（可选），例如 2020。
        year_end: 结束发表年份筛选（可选），例如 2024。
        sort: 排序方式。可选：
              - "relevance"（默认，按相关性）
              - "citationCount:desc"（按引用量降序）
              - "publicationDate:desc"（按发表日期降序）

    Returns:
        ToolResponse，content 中为 JSON 格式的论文列表，包含 total 字段表示总数。
    """
    try:
        params: dict = {
            "query": query,
            "limit": max(1, min(100, limit)),
            "fields": fields if fields else _DEFAULT_PAPER_FIELDS,
            "sort": sort,
        }
        if year_start is not None or year_end is not None:
            # Semantic Scholar 支持年份范围格式 2020:2024 或单年 2020
            start = year_start if year_start is not None else ""
            end = year_end if year_end is not None else ""
            params["publicationDateOrYear"] = f"{start}:{end}"

        query_str = _safe_params(params)
        url = f"{_BASE_URL}/paper/search?{query_str}"
        data = _fetch_json(url)

        # 提取核心字段，减少 Token 占用
        papers = data.get("data", [])
        total = data.get("total", 0)
        result = {
            "total": total,
            "query": query,
            "papers": papers,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("论文搜索", e)


def get_paper_details(
    paper_id: str,
    fields: Optional[str] = None,
) -> ToolResponse:
    """根据论文 ID 获取单篇论文的详细信息。

    适用场景：已从搜索结果中获得 paperId，需要进一步了解该论文的摘要、
    参考文献、被引情况、开放获取 PDF 链接等。

    Args:
        paper_id: 论文标识符。支持多种形式：
                  - Semantic Scholar ID（如：649def34f8be52c8b66281af98ae884c09aef38b）
                  - DOI（前缀加 DOI:，如：DOI:10.1145/263690.263806）
                  - arXiv ID（前缀加 arXiv:，如：arXiv:1705.04304）
        fields: 逗号分隔的返回字段列表。默认返回：
                paperId,title,abstract,year,authors,citationCount,referenceCount,
                references,citations,venue,url,tldr,fieldsOfStudy,isOpenAccess,
                openAccessPdf

    Returns:
        ToolResponse，content 中为 JSON 格式的论文详情。
    """
    try:
        params = {
            "fields": fields if fields else _DEFAULT_PAPER_DETAIL_FIELDS,
        }
        query_str = _safe_params(params)
        encoded_id = urllib.parse.quote(paper_id, safe="")
        url = f"{_BASE_URL}/paper/{encoded_id}?{query_str}"
        data = _fetch_json(url)

        text = json.dumps(data, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("获取论文详情", e)


def search_authors(
    query: str,
    limit: int = 5,
) -> ToolResponse:
    """在 Semantic Scholar 中搜索学者/作者。

    适用场景：用户提到某位研究者姓名，想查找其学术档案、所属机构、
    论文数量、H-index 等。

    Args:
        query: 作者姓名关键词，例如 "Yann LeCun"、"Geoffrey Hinton"。
        limit: 返回结果数量上限（1~100，默认 5）。

    Returns:
        ToolResponse，content 中为 JSON 格式的作者列表，包含 total 字段。
    """
    try:
        params = {
            "query": query,
            "limit": max(1, min(100, limit)),
            "fields": _DEFAULT_AUTHOR_FIELDS,
        }
        query_str = _safe_params(params)
        url = f"{_BASE_URL}/author/search?{query_str}"
        data = _fetch_json(url)

        result = {
            "total": data.get("total", 0),
            "query": query,
            "authors": data.get("data", []),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("作者搜索", e)


def get_paper_citations(
    paper_id: str,
    limit: int = 10,
    fields: Optional[str] = None,
) -> ToolResponse:
    """获取某篇论文的引用列表（被哪些后续论文引用）。

    适用场景：评估某篇论文的影响力、追踪该方向的最新进展。

    Args:
        paper_id: 论文标识符（格式同 get_paper_details）。
        limit: 返回引用数量上限（默认 10，最大 1000）。
        fields: 引用论文的返回字段，默认：
                paperId,title,authors,year,venue,url,abstract

    Returns:
        ToolResponse，content 中为 JSON 格式的引用论文列表，含 total 字段。
    """
    try:
        params = {
            "limit": max(1, min(1000, limit)),
            "fields": fields if fields else _DEFAULT_CITE_FIELDS,
        }
        query_str = _safe_params(params)
        encoded_id = urllib.parse.quote(paper_id, safe="")
        url = f"{_BASE_URL}/paper/{encoded_id}/citations?{query_str}"
        data = _fetch_json(url)

        result = {
            "paper_id": paper_id,
            "total": data.get("total", 0),
            "citations": data.get("data", []),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("获取引用列表", e)


def get_paper_references(
    paper_id: str,
    limit: int = 10,
    fields: Optional[str] = None,
) -> ToolResponse:
    """获取某篇论文的参考文献列表（该论文引用了哪些先前论文）。

    适用场景：了解某篇工作的理论基础、追溯原始方法来源。

    Args:
        paper_id: 论文标识符（格式同 get_paper_details）。
        limit: 返回参考文献数量上限（默认 10，最大 1000）。
        fields: 参考文献的返回字段，默认：
                paperId,title,authors,year,venue,url,abstract

    Returns:
        ToolResponse，content 中为 JSON 格式的参考文献列表，含 total 字段。
    """
    try:
        params = {
            "limit": max(1, min(1000, limit)),
            "fields": fields if fields else _DEFAULT_CITE_FIELDS,
        }
        query_str = _safe_params(params)
        encoded_id = urllib.parse.quote(paper_id, safe="")
        url = f"{_BASE_URL}/paper/{encoded_id}/references?{query_str}"
        data = _fetch_json(url)

        result = {
            "paper_id": paper_id,
            "total": data.get("total", 0),
            "references": data.get("data", []),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("获取参考文献列表", e)


# =============================================================================
# 论文全文读取工具
# =============================================================================

def get_paper_pdf_url(
    paper_id: str,
) -> ToolResponse:
    """查询某篇论文的开放获取 PDF 链接。

    适用场景：Agent 需要确认某篇论文是否可以免费下载，或获取 PDF 直链。

    Args:
        paper_id: 论文标识符（格式同 get_paper_details）。

    Returns:
        ToolResponse，content 中为 JSON，包含 openAccessPdf.url 和
        openAccessPdf.status 字段；若论文不开放获取，则 url 为空。
    """
    try:
        params = {
            "fields": "paperId,title,openAccessPdf,isOpenAccess",
        }
        query_str = _safe_params(params)
        encoded_id = urllib.parse.quote(paper_id, safe="")
        url = f"{_BASE_URL}/paper/{encoded_id}?{query_str}"
        data = _fetch_json(url)

        result = {
            "paper_id": paper_id,
            "title": data.get("title", ""),
            "is_open_access": data.get("isOpenAccess", False),
            "pdf_url": (data.get("openAccessPdf") or {}).get("url", ""),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return ToolResponse(content=[TextBlock(type="text", text=text)])
    except Exception as e:
        return _format_error("获取 PDF 链接", e)


def read_paper_by_url(
    pdf_url: str,
    max_pages: int = 0,
) -> ToolResponse:
    """通过 PDF 直链下载并提取论文正文文本。

    适用场景：已知某篇论文的 PDF 链接，需要让 Agent 阅读其内容。

    Args:
        pdf_url: PDF 文件的网络直链（如 Semantic Scholar 的 openAccessPdf.url、
                 arXiv PDF 链接、期刊 OA 链接等）。
        max_pages: 最大读取页数。0 表示读取全部页数（默认）；
                   建议长论文设为 10~20 页以避免 Token 爆炸。

    Returns:
        ToolResponse，content 中为提取的纯文本字符串，包含论文标题/段落/表格文本。
        若 PDF 为扫描版图片，可能无法提取文字，会在文本中提示。
    """
    if not _HAS_FITZ:
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=(
                    "读取论文需要安装 PDF 解析库 pymupdf。"
                    "请执行：pip install pymupdf"
                ),
            )]
        )

    tmp_path = None
    try:
        # 1. 下载 PDF 到临时文件
        req = urllib.request.Request(pdf_url, headers={
            "Accept": "application/pdf",
            "User-Agent": "Mozilla/5.0 (Academic Paper Reader)",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()

        suffix = os.path.splitext(urllib.parse.urlparse(pdf_url).path)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        # 2. 使用 pymupdf 提取文本
        doc = fitz.open(tmp_path)
        total_pages = doc.page_count
        pages_to_read = total_pages if max_pages <= 0 else min(max_pages, total_pages)

        lines: list[str] = []
        lines.append(f"【论文阅读报告】共 {total_pages} 页，本次读取前 {pages_to_read} 页\n")

        for i in range(pages_to_read):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if not text:
                # 尝试检测是否为扫描页（图片）
                images = page.get_images()
                if images:
                    lines.append(f"\n--- 第 {i + 1} 页 ---\n[该页为扫描图片，无法提取文字]\n")
                else:
                    lines.append(f"\n--- 第 {i + 1} 页 ---\n[该页无文本内容]\n")
            else:
                lines.append(f"\n--- 第 {i + 1} 页 ---\n{text}\n")

        doc.close()

        full_text = "\n".join(lines)
        # 简单清理：合并多余空行
        full_text = "\n".join(line for line in full_text.splitlines() if line.strip())

        return ToolResponse(content=[TextBlock(type="text", text=full_text)])
    except Exception as e:
        return _format_error("读取论文 PDF", e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_paper(
    paper_id: str,
    max_pages: int = 0,
) -> ToolResponse:
    """根据论文 ID 自动获取开放获取 PDF 并提取正文文本。

    本工具采用双路策略：
    1) 优先通过 Semantic Scholar API 查询 openAccessPdf 链接（支持 DOI、S2 ID 等）；
    2) 若 S2 API 失败（如 429 限流）且 ID 为 arXiv 格式，则直接构造 arXiv PDF 链接兜底。

    Args:
        paper_id: 论文标识符。支持格式：
                  - arXiv ID（如：arXiv:1706.03762）
                  - DOI（如：DOI:10.1145/263690.263806，需 S2 API 正常）
                  - Semantic Scholar ID（需 S2 API 正常）
        max_pages: 最大读取页数。0 表示全部（默认）；
                   建议长论文设为 10~20 页。

    Returns:
        ToolResponse，content 中为提取的论文文本。
        若该论文无开放获取 PDF 且无法兜底，会返回提示信息。
    """
    pdf_url: str | None = None
    paper_title = paper_id
    s2_error = ""

    # ============================================================
    # 方式 1：通过 S2 API 查询 openAccessPdf 链接
    # ============================================================
    try:
        url_resp = get_paper_pdf_url(paper_id)
        url_text = url_resp.content[0].get("text", "")
        url_data = json.loads(url_text)
        pdf_url = url_data.get("pdf_url", "")
        paper_title = url_data.get("title", "") or paper_id
    except Exception as e:
        s2_error = str(e)
        pdf_url = None

    # ============================================================
    # 方式 2：若方式 1 失败，对 arXiv ID 直接构造 PDF 链接兜底
    # ============================================================
    if not pdf_url:
        pid_lower = paper_id.strip().lower()
        if pid_lower.startswith("arxiv:"):
            arxiv_id = paper_id.split(":", 1)[1].strip()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            paper_title = f"arXiv:{arxiv_id}"
        elif not s2_error:
            # S2 API 正常返回了，但论文本身没有开放获取 PDF
            msg = (
                f"论文《{paper_title}》（ID: {paper_id}）暂无开放获取 PDF 链接。\n"
                "建议尝试：1) 查看作者个人主页；2) 在 arXiv / ResearchGate 搜索预印本；"
                "3) 或直接提供 PDF 链接调用 read_paper_by_url。"
            )
            return ToolResponse(content=[TextBlock(type="text", text=msg)])
        else:
            # S2 API 失败，且不是 arXiv ID，无法兜底
            msg = (
                f"获取 PDF 链接失败：{s2_error}\n"
                f"ID: {paper_id}\n"
                "该 ID 非 arXiv 格式，无法自动兜底获取 PDF。\n"
                "建议：1) 配置 S2_API_KEY 以提高限额；2) 使用 search_papers 查找 arXiv 版本；"
                "3) 或直接调用 read_paper_by_url(pdf_url='已知PDF链接')。"
            )
            return ToolResponse(content=[TextBlock(type="text", text=msg)])

    # ============================================================
    # 读取 PDF 内容
    # ============================================================
    content_resp = read_paper_by_url(pdf_url, max_pages=max_pages)
    content_text = content_resp.content[0].get("text", "")

    # 拼接论文标题信息
    header = f"【正在阅读论文】\n标题：{paper_title}\nID：{paper_id}\nPDF来源：{pdf_url}\n"
    full_text = header + "\n" + content_text

    return ToolResponse(content=[TextBlock(type="text", text=full_text)])
