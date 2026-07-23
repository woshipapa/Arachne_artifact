
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
        if not dist.is_initialized():
            print("Warning: Distributed environment not initialized. Custom systems might not function correctly.")
            rank = 0
            world_size = 1
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        # print(f"===================== enter mainintegrationSystem=========================")
        extra_config = {}

        # try:
        #     from megatron.core import mpu
        #     extra_config['data_parallel_rank'] = mpu.get_data_parallel_rank()
        #     extra_config['data_parallel_world_size'] = mpu.get_data_parallel_world_size()
        #     print("Successfully gathered optional info from Megatron MPU.")
        # except (ImportError, AttributeError):
        #     print("Megatron MPU not found or incompatible. Proceeding with basic info.")
        #     pass
        
        # if args is not None:
        #     if hasattr(args, 'initial_plan_path'):
        #         extra_config['initial_plan_path'] = args.initial_plan_path
        #     if hasattr(args, 'extra_log_label'):
        #         extra_config['extra_log_label'] = args.extra_log_label
        #     print("Gathered optional info from args.")


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
        global_logger = get_global_logger()
        global_logger.info("Component 'GlobalLogger' initialized by Integration Layer.")

        
        handler = DynamicForwardStepHandler()
        handler.prepare_after_dist_initialized(log_dir=handler_log_dir)
        
        assert handler.logger is not None, "Handler failed to self-initialize!"
        handler.logger.info("Component 'DynamicForwardStepHandler' initialized by Integration Layer.")

        # ==========================================================
        # ==========================================================
        
        chosen_logger_for_timer = None

        if handler.logger:
            chosen_logger_for_timer = handler.logger
            global_logger.info("Integration Layer: Found Handler's logger. It will be used for MyTimer.")
        
        if chosen_logger_for_timer is None:
            chosen_logger_for_timer = global_logger
            global_logger.warning("Integration Layer: Handler's logger not found. MyTimer will fall back to GlobalLogger.")

        global_timer.set_logger(chosen_logger_for_timer)
        
        timer_logger = global_timer.logger 
        if timer_logger:
             timer_logger.info(f"Component 'MyTimer' has been successfully configured to use logger: '{chosen_logger_for_timer.name}'.")


SystemManager().register(MainIntegrationSystem())