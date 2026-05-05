import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


url = "https://api.siliconflow.cn/v1/uploads/audio/voice"
headers = {
    "Authorization": f"Bearer {config.TTS_API_KEY}" # 从 https://cloud.siliconflow.cn/account/ak 获取
}
files = {
    "file": open("data/my_voice2.mp3", "rb") # 参考音频文件
}
data = {
    "model": "FunAudioLLM/CosyVoice2-0.5B", # 模型名称
    "customName": "my_voice_20260429_12_47", # 参考音频名称
    # "text": "清晨的阳光洒进房间，新的一天开始了。有时候我会想，科技的意义到底是什么？是让生活更便捷，还是让我们更贴近彼此？当我听到一段熟悉的声音，答案似乎变得清晰起来。" # 参考音频的文字内容
    "text": "大家好，我是今天的讲述者。声音是每个人独特的印记，而技术正在让这种印记被更好地理解和重现。"
}

response = requests.post(url, headers=headers, files=files, data=data)

print(response.status_code)
print(response.json())  # 打印响应内容（如果是JSON格式）