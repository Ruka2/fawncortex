"""
终端日志收集器（TeeLogger）
===========================
解耦式日志模块：通过拦截 sys.stdout，自动收集所有 print() 输出到文件，
无需修改代码中任何现有的 print 语句。

使用方式：
    from scripts.pipeline.logger import enable_file_logging
    enable_file_logging()   # 从此处开始，所有 print 都会同时写入文件
"""

import sys
import os
import atexit
from datetime import datetime

# 项目配置：日志目录
import config


class TeeLogger:
    """Tee 式日志收集器：同时输出到终端和文件。

    替换 sys.stdout 后，所有 print() / sys.stdout.write() 的输出会被自动收集，
    终端实时显示的同时追加写入日志文件。
    """

    def __init__(self, log_dir: str | None = None, filename: str | None = None) -> None:
        # 保留原始终端 stdout，用于自身初始化信息和后续恢复
        self.terminal = sys.stdout

        # 使用配置文件中的路径，未传参时取默认值
        if log_dir is None:
            log_dir = getattr(config, "LOG_DIR", "./logs")
        os.makedirs(log_dir, exist_ok=True)

        # 自动生成文件名：run_YYYYMMDD_HHMMSS.log
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"run_{timestamp}.log"

        self.log_path = os.path.join(log_dir, filename)
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        # 用原始终端打印初始化信息（避免递归进入 self.write）
        self.terminal.write(f"[Logger] 终端日志已收集到文件: {self.log_path}\n")
        self.terminal.flush()

    # -------------------------------------------------------------------------
    # 文件-like 接口（被 sys.stdout 替换后由 print() 调用）
    # -------------------------------------------------------------------------
    def write(self, message: str) -> None:
        """同时写入终端和日志文件。

        文件已关闭时（如 Ctrl+C 中断后）仅写终端，避免 ValueError。
        """
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self) -> None:
        """同时刷新终端和文件缓冲区。"""
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self) -> None:
        """关闭日志文件并恢复原始 stdout。

        必须先恢复 sys.stdout，再关文件，否则解释器关闭期间的
        内部打印会走到已关闭的文件上。
        """
        # 1. 恢复原始 stdout，防止后续 flush/write 再走 TeeLogger
        if sys.stdout is self:
            sys.stdout = self.terminal

        # 2. 关闭日志文件
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()
            self.terminal.write(f"[Logger] 日志文件已关闭: {self.log_path}\n")
            self.terminal.flush()

    def __del__(self) -> None:
        """析构时仅做 stdout 恢复，不再操作文件（避免 interpreter shutdown 时竞态）。"""
        if sys.stdout is self:
            sys.stdout = self.terminal


# -------------------------------------------------------------------------
# 公共 API
# -------------------------------------------------------------------------

def enable_file_logging(log_dir: str | None = None, filename: str | None = None) -> TeeLogger:
    """启用文件日志收集。

    调用后，所有 print() / sys.stdout.write() 的输出会同时写入终端和文件。
    程序退出时自动关闭日志文件。

    Args:
        log_dir: 日志存放目录，默认当前工作目录下的 "logs"。
        filename: 日志文件名，默认自动生成（run_YYYYMMDD_HHMMSS.log）。

    Returns:
        TeeLogger 实例，可用于后续手动关闭或恢复 stdout。
    """
    logger = TeeLogger(log_dir, filename)
    sys.stdout = logger
    atexit.register(logger.close)
    return logger


def disable_file_logging() -> None:
    """恢复原始 stdout，停止文件日志收集。"""
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
