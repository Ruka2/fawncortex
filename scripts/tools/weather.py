import urllib.request
import urllib.parse

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock


def get_weather(city: str) -> ToolResponse:
    """查询指定城市今天的天气状况。

    Args:
        city: 城市名称，例如 "北京"、"上海"、"广州"
    """
    try:
        encoded_city = urllib.parse.quote(city)
        # wttr.in 是一个免费的命令行天气服务，无需 API Key
        url = f"https://wttr.in/{encoded_city}?format=%l:+%C,+%t&lang=zh"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "curl/7.68.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            weather = response.read().decode("utf-8").strip()

        return ToolResponse(
            content=[TextBlock(type="text", text=weather)]
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"查询天气失败：{e}")]
        )
