_KNOWN_ACTIONS = {
    "smile", "happy", "laugh", "sad", "cry", "angry",
    "surprise", "shy", "sleepy", "disgust", "neutral",
    "blink", "close_eyes", "wink",
    "lean_left", "lean_right", "nod", "tilt",
    "talk",
}


def parse_action(text: str) -> str:
    """从 emotion_agent 的输出中提取动作名称，失败则返回 'smile'。"""
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        word = word.strip()
        if word in _KNOWN_ACTIONS:
            return word
    return "smile"