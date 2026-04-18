# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import torch.distributed as dist
# 创建Logger对象
logger = logging.getLogger('global_logger')
logger.setLevel(logging.INFO)  # 设置全局日志级别

# 防止重复添加Handler
if not logger.hasHandlers():
    # 创建控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件Handler，使用RotatingFileHandler以支持日志轮转
    log_file = 'dual-single.log'  # 指定日志文件路径
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    # 创建一个日志格式，包含 rank 信息
    # log_format = f'%(asctime)s - Rank: {dist.get_rank()} - %(levelname)s - %(message)s'

    # 创建格式化器
    # formatter = logging.Formatter(log_format)
    # console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    # 将Handlers添加到Logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
