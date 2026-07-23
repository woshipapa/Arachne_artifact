import logging
import os
from datetime import datetime
import torch
from megatron.core import mpu
from megatron.core.models.hunyuan.model import HunyuanParams

logger = None

def get_logger():
    if torch.distributed.get_rank() == 0:
        global logger
        return logger
    return None

def log_first_rank(str):
    if torch.distributed.get_rank() == 0:
        global logger
        logger.info(str)

def set_logger(args):
    global logger
    if torch.distributed.get_rank() == 0:
        logging.getLogger().handlers.clear()

        LOG_DIR = args.debug_dir
        LOGGER_NAME = "debug"
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        frame = args.num_frames
        video_resolution = f"{args.video_resolution[0]}x{args.video_resolution[1]}" if args.video_resolution else "default"
        log_file = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_tp{tp_size}pp{pp_size}cp{cp_size}dp{dp_size}_f{frame}res{video_resolution}.log'
        log_file = os.path.join(args.debug_dir, log_file)
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)