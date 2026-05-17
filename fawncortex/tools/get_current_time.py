"""
获取当前时间工具
================
为智能体提供获取实时日期时间的能力，
用于检索实时性或需要时间上下文的任务。
"""

from datetime import datetime

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock


def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> ToolResponse:
    """获取目前最新的日期和时间。

    当用户询问涉及时间、日期、时效性、最近动态、
    实时信息或需要以当前时间为参考进行推理时，
    调用此工具获取准确的系统时间。

    Args:
        format: 日期时间格式字符串，默认为 "%Y-%m-%d %H:%M:%S"。
            常用格式示例：
            - "%Y-%m-%d %H:%M:%S" → 2026-05-09 12:30:00
            - "%Y-%m-%d" → 2026-05-09
            - "%H:%M:%S" → 12:30:00
            - "%Y年%m月%d日" → 2026年05月09日
    """
    try:
        now = datetime.now()
        time_str = now.strftime(format)

        # 额外提供 ISO 格式和常用格式，方便模型使用
        iso_str = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")
        weekday = now.strftime("%A")  # 英文星期
        weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]

        result = (
            f"当前时间：{time_str}\n"
            f"日期：{date_str}\n"
            f"ISO 格式：{iso_str}\n"
            f"星期：{weekday_cn} ({weekday})"
        )

        return ToolResponse(
            content=[TextBlock(type="text", text=result)]
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"获取当前时间失败：{e}")]
        )
