import os
import logging
import torch.distributed as dist

# 定义一个全局唯一的 logger 名称
# 项目中所有地方都将通过这个名字获取 logger
_GLOBAL_LOGGER_NAME = "GLOBAL_LOGGER"

def setup_logger(log_dir: str, level: int = logging.INFO):
    """
    初始化全局唯一的 logger。

    这个函数应该在分布式组初始化后被调用一次。
    它会为每个 rank 设置一个 logger，该 logger 会同时输出到控制台和独立的日志文件。
    如果 logger 已经被设置过，重复调用此函数不会产生任何效果。

    Args:
        log_dir (str): 存放日志文件的目录。
        level (int, optional): 日志级别。默认为 logging.INFO。
    """

    # exit(1)
    logger = logging.getLogger(_GLOBAL_LOGGER_NAME)

    # # 1. 防止重复设置：通过检查 logger 是否已有 handlers 来判断
    # if logger.hasHandlers():
    #     return

    # 2. 动态获取 rank，如果分布式未初始化，则默认为 0
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    logger.setLevel(level)

    # 3. 设置日志格式
    # 这个格式与你原来的完全相同
    formatter = logging.Formatter(
        f"[%(asctime)s] [Rank {rank}] [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # print(f"Rank {rank} setup_logger=================================")
    # exit(1)
    # 4. 配置 StreamHandler (输出到控制台)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 5. 配置 FileHandler (输出到文件)
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 禁止向上传播日志，防止 root logger 重复打印
    logger.propagate = False
    
    if rank == 0:
        logger.info(f"Global logger '{_GLOBAL_LOGGER_NAME}' initialized. Log files will be saved in '{log_dir}'.")


def get_logger():
    """
    获取全局 logger 实例。

    Returns:
        logging.Logger: 配置好的全局 logger 对象。
    """
    return logging.getLogger(_GLOBAL_LOGGER_NAME)