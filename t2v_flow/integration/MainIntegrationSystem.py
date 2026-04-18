# my_system/integration.py (我们将所有集成系统放一起)

import torch.distributed as dist
from .manager import SystemManager
from .base import CustomSystem
from my_utils import GlobalLogger
from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
from my_utils import global_timer, get_global_logger
import os
import logging
class MainIntegrationSystem(CustomSystem):
    @property
    def name(self) -> str: return "MainIntegrationLayer"

    def initialize(self, args):
        # --- 1. 获取必要信息 (来自 torch.dist) ---
        if not dist.is_initialized():
            print("Warning: Distributed environment not initialized. Custom systems might not function correctly.")
            rank = 0
            world_size = 1
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        # print(f"===================== enter mainintegrationSystem=========================")
        # --- 2. 尝试获取可选信息，构建 extra_config 字典 ---
        extra_config = {}

        # 尝试从 Megatron mpu 获取信息
        # try:
        #     from megatron.core import mpu
        #     extra_config['data_parallel_rank'] = mpu.get_data_parallel_rank()
        #     extra_config['data_parallel_world_size'] = mpu.get_data_parallel_world_size()
        #     print("Successfully gathered optional info from Megatron MPU.")
        # except (ImportError, AttributeError):
        #     # 如果 mpu 不存在或没有对应方法，静默跳过
        #     print("Megatron MPU not found or incompatible. Proceeding with basic info.")
        #     pass
        
        # 尝试从 Megatron args 获取信息
        # if args is not None:
        #     if hasattr(args, 'initial_plan_path'):
        #         extra_config['initial_plan_path'] = args.initial_plan_path
        #     if hasattr(args, 'extra_log_label'):
        #         extra_config['extra_log_label'] = args.extra_log_label
        #     print("Gathered optional info from args.")


         # --- 3. 初始化 GlobalLogger ---
        # 这是一个备选 logger，必须先于 Timer 初始化
        global_logger_instance = GlobalLogger()
        if not hasattr(args, 'log_dir') or args.logdir is None:
            base_log_dir = "logs"
        else:
            base_log_dir = args.logdir

        log_dir = os.path.join(base_log_dir, "global_logger")  
        handler_log_dir = os.path.join(base_log_dir, "handler_logs")   
        if os.environ.get("MODEL_TYPE") is not None:
            log_dir = os.path.join(log_dir, os.environ.get("MODEL_TYPE"))
            handler_log_dir = os.path.join(handler_log_dir, os.environ.get("MODEL_TYPE"))
            if os.environ.get("RESOLUTION") is not None:
                log_dir = os.path.join(log_dir, os.environ.get("RESOLUTION"))
                handler_log_dir = os.path.join(handler_log_dir, os.environ.get("RESOLUTION"))
                if os.environ.get("MAX_FRAMES") is not None:
                    log_dir = os.path.join(log_dir, os.environ.get("MAX_FRAMES"))
                    handler_log_dir = os.path.join(handler_log_dir, os.environ.get("MAX_FRAMES"))
                    if os.environ.get("SCHEDULE_TYPE") is not None:
                        log_dir = os.path.join(log_dir, os.environ.get("SCHEDULE_TYPE"))
                        handler_log_dir = os.path.join(handler_log_dir, os.environ.get("SCHEDULE_TYPE"))
        global_logger_instance.setup(
            log_dir=log_dir,
            level=logging.INFO, 
            rank=rank,
            world_size=world_size,
            **extra_config
        )
        # 获取配置好的 logger 对象
        global_logger = get_global_logger()
        global_logger.info("Component 'GlobalLogger' initialized by Integration Layer.")

        # --- 4. 初始化 DynamicForwardStepHandler ---
        # 获取单例实例会触发其内部的自初始化逻辑，包括创建它自己的 logger
        
        handler = DynamicForwardStepHandler()
        handler.prepare_after_dist_initialized(log_dir=handler_log_dir)
        
        # 验证 Handler 是否成功初始化
        assert handler.logger is not None, "Handler failed to self-initialize!"
        handler.logger.info("Component 'DynamicForwardStepHandler' initialized by Integration Layer.")

        # ==========================================================
        # ==== 阶段二：为 Timer 注入 Logger (核心逻辑) ====
        # ==========================================================
        
        # --- 5. 决定 Timer 应该使用哪个 Logger ---
        chosen_logger_for_timer = None

        # a. 优先尝试获取 Handler 的 Logger
        if handler.logger:
            chosen_logger_for_timer = handler.logger
            global_logger.info("Integration Layer: Found Handler's logger. It will be used for MyTimer.")
        
        # b. 如果 Handler 没有 logger，则回退到 GlobalLogger
        if chosen_logger_for_timer is None:
            chosen_logger_for_timer = global_logger
            global_logger.warning("Integration Layer: Handler's logger not found. MyTimer will fall back to GlobalLogger.")

        # --- 6. 将选定的 Logger 注入到全局的 Timer 实例中 ---
        # global_timer 是从 your_timer_file.py 导入的那个全局实例
        global_timer.set_logger(chosen_logger_for_timer)
        
        # 现在，当您使用 global_timer 时，它会向正确的 logger 输出日志
        # 我们可以用 timer 自带的 logger 来打印一条确认信息
        timer_logger = global_timer.logger 
        if timer_logger:
             timer_logger.info(f"Component 'MyTimer' has been successfully configured to use logger: '{chosen_logger_for_timer.name}'.")

# --- 注册这个统一的集成系统 ---

SystemManager().register(MainIntegrationSystem())