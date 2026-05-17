"""
defaultboy06V2 VTube Studio 交互式动作调试工具
直接注入 VTS 内置 Input 参数，由 VTS 根据 vtube.json 的 ParameterSettings
映射到 Live2D 参数。
"""

import asyncio
import random
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deerberry.components.body.vts_controller import VTSController

from emotion_animate import (
    animate_open_mouse, animate_smile,
    animate_angry, animate_wink_left, animate_wink_right,
    animate_close_eyes, animate_confused,
    animate_lean_left, animate_lean_right,
    animate_look_up, animate_look_down,
    animate_surprised, animate_smirk,
)


async def main():
    vts = VTSController()
    try:
        await vts.connect_and_auth()
    except Exception as e:
        print(f"\n连接失败: {e}")
        return


    default_duration = random.uniform(5.0, 10.0)

    while True:
        
        # 在后台线程运行 input，避免阻塞 asyncio 事件循环
        user_input = await asyncio.to_thread(input)
        user_input = user_input.strip()


        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break


        action = user_input
        duration = default_duration

        # ---- 独立动态动画循环 ----
        if action == "open_mouse":
            await animate_open_mouse(vts, duration=duration)
            continue
        if action == "smile":
            await animate_smile(vts, duration=duration)
            continue
        if action == "angry":
            await animate_angry(vts, duration=duration)
            continue
        if action == "wink_left":
            await animate_wink_left(vts, duration=duration)
            continue
        if action == "wink_right":
            await animate_wink_right(vts, duration=duration)
            continue
        if action == "close_eyes":
            await animate_close_eyes(vts, duration=duration)
            continue
        if action == "confused":
            await animate_confused(vts, duration=duration)
            continue
        if action == "lean_left":
            await animate_lean_left(vts, duration=duration)
            continue
        if action == "lean_right":
            await animate_lean_right(vts, duration=duration)
            continue
        if action == "look_up":
            await animate_look_up(vts, duration=duration)
            continue
        if action == "look_down":
            await animate_look_down(vts, duration=duration)
            continue
        if action == "surprised":
            await animate_surprised(vts, duration=duration)
            continue
        if action == "smirk":
            await animate_smirk(vts, duration=duration)
            continue


    await vts.close()


if __name__ == "__main__":
    asyncio.run(main())
