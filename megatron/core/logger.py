# utils/logger.py

import sys
import os
from loguru import logger
import torch.distributed as dist

# 1. 直接从 loguru 导入 logger 对象。
#    此时它只有一个默认的、未配置的 handler。
#    所有文件 `from .logger import logger` 导入的都将是这同一个对象。

# 2. 将上一问的配置逻辑封装成一个函数。
def setup_logger(log_dir: str):
    """
    配置全局的 logger 对象。
    这个函数应该在分布式环境初始化之后，在程序的主要逻辑开始之前被调用。
    """
    # 先移除默认的 handler，以便完全控制
    logger.remove()

    # 获取 rank
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0  # 默认为非分布式环境

    # 日志格式，其中包含了 {extra[rank]} 以便显示 rank
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "rank:<cyan>{extra[rank]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 所有 rank 的日志都写入到各自独立的文件
    log_file_path = os.path.join(log_dir, f"rank_{rank}.log")
    logger.add(
        log_file_path,
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8"
    )

    # 只有 rank 0 才输出到控制台
    if rank == 4:
        logger.add(
            sys.stdout,
            format=log_format,
            level="INFO"  # 控制台可以设置更高的日志级别
        )

    # 绑定 rank 到 extra，这样日志格式化时才能获取到
    logger.configure(extra={"rank": rank})

    if rank == 0:
        logger.info("Logger has been configured for all ranks.")