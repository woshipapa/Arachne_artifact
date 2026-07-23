# utils/logger.py

import sys
import os
from loguru import logger
import torch.distributed as dist


def setup_logger(log_dir: str):
    logger.remove()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "rank:<cyan>{extra[rank]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"rank_{rank}.log")
    logger.add(
        log_file_path,
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8"
    )

    if rank == 4:
        logger.add(
            sys.stdout,
            format=log_format,
            level="INFO"
        )

    logger.configure(extra={"rank": rank})

    if rank == 0:
        logger.info("Logger has been configured for all ranks.")