# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import torch.distributed as dist
logger = logging.getLogger('global_logger')
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_file = 'dual-single.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    # log_format = f'%(asctime)s - Rank: {dist.get_rank()} - %(levelname)s - %(message)s'

    # formatter = logging.Formatter(log_format)
    # console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
