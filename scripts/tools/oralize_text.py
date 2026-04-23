import re

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock



def oralize_for_speech(text: str) -> ToolResponse:
    """将文本整理为适合语音朗读的口头语版本。
    
    Args:
        text: 填入已整理好的口语文本，因此 Agent 传入到该工具前应该提前准备好讲需要口头朗读的内容填入到入参中。
    """

    # 1. 去掉 Markdown 粗体/斜体
    cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)

    # 2. 去掉 Markdown 标题标记
    cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)

    # 3. 去掉 Markdown 链接，保留链接文字
    cleaned = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', cleaned)

    # 4. 去掉独立 URL
    cleaned = re.sub(r'https?://\S+', '', cleaned)

    # 5. 去掉 Emoji（仅匹配常见 emoji 区域，避免误伤中文）
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F1E0-\U0001F1FF"  # 国旗
        "\U00002702-\U000027B0"  # 装饰符号
        "\U0001F900-\U0001F9FF"  # 补充符号
        "\U0001FA00-\U0001FAFF"  # 扩展-A
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub('', cleaned)

    # 6. 合并换行为空格（适合连续朗读）
    cleaned = cleaned.replace('\n', ' ')

    # 7. 去掉多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return ToolResponse(
        content=[TextBlock(type="text", text=cleaned)]
    )
