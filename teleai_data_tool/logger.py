import torch.distributed as dist
from loguru import logger as loguru_logger
import sys


class DistributedLogger:
    def __init__(self):
        self.initialized = False

    def is_main_process(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def initialize(self, save_path=None, level="INFO"):
        if self.initialized:
            return

        loguru_logger.remove()

        if self.is_main_process():
            if save_path is not None:
                loguru_logger.add(
                    save_path,
                    level=level,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
                    enqueue=True,
                )
            loguru_logger.add(
                sink=sys.stdout,
                level=level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
                enqueue=True,
            )
        self.initialized = True

    def log(self, message, level="info"):
        if self.is_main_process():
            if level.lower() == "info":
                loguru_logger.opt(depth=2).info(message)
            elif level.lower() == "debug":
                loguru_logger.opt(depth=2).debug(message)
            elif level.lower() == "warning":
                loguru_logger.opt(depth=2).warning(message)
            elif level.lower() == "error":
                loguru_logger.opt(depth=2).error(message)
            elif level.lower() == "critical":
                loguru_logger.opt(depth=2).critical(message)
            else:
                raise ValueError(f"Unsupported log level: {level}")

    def info(self, message):
        self.log(message, level="info")

    def debug(self, message):
        self.log(message, level="debug")

    def warning(self, message):
        self.log(message, level="warning")

    def error(self, message):
        self.log(message, level="error")

    def critical(self, message):
        self.log(message, level="critical")


logger = DistributedLogger()


def initialize_logger(name="teleai_logger", level="INFO"):
    logger.initialize(name, level)
