import logging
import os
import sys
import threading
import time
# --------------------------------------------------
# 1. singleton metaclass (unchanged)
# --------------------------------------------------
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

# --------------------------------------------------
# 2. the GlobalLogger class
# --------------------------------------------------

# a handler that flushes on every record
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # <--- key: flush to disk as soon as a record is written

# in your setup function, replace:
#   file_handler = logging.FileHandler(log_file, mode="a")
# with:



class GlobalLogger(metaclass=SingletonMeta):
    """
    A portable, framework-agnostic global logger.

    The singleton pattern keeps exactly one instance per application.
    """
    _GLOBAL_LOGGER_NAME = "MySystemGlobalLogger"

    def __init__(self):
        """
        The constructor stays minimal: it only initialises internal state
        and depends on no external framework.
        """
        self.logger = logging.getLogger(self._GLOBAL_LOGGER_NAME)
        self.is_configured = False

        self.profile_file = None
        self.machine_id = "Unknown"
        self.profile_enabled = False
        self.time_offset = 0.0 # defaults to 0

    def set_time_offset(self, offset: float):
            """ Set the time offset. """
            self.time_offset = offset
            self.logger.info(f"GlobalLogger time offset set to: {offset:.6f}s")


    def setup(self, log_dir: str, level: int = logging.INFO, rank: int = 0, world_size: int = 1, **kwargs):
        """
        Take configuration from the caller and finish setting the logger up.

        Idempotent: safe to call repeatedly, but only the first call takes effect.
        It calls no distributed library directly; rank and world_size are parameters.

        Args:
            log_dir (str): directory holding the log files.
            level (int): log level.
            rank (int): this process's rank.
            world_size (int): total number of processes.
        """
        # 1. guard against reconfiguration
        if self.is_configured:
            self.logger.warning("Logger is already configured. Ignoring subsequent setup call.")
            return
        
        # robustness: drop any pre-existing handler before adding new ones
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 2. set the log level
        self.logger.setLevel(level)

        # --- build the log format string dynamically ---
        log_format = f"[%(asctime)s] [Rank {rank}/{world_size}]"

        # optional Megatron mpu information in kwargs
        dp_rank = kwargs.get('data_parallel_rank')
        if dp_rank is not None:
            dp_size = kwargs.get('data_parallel_world_size', '?')
            log_format += f" [DP_Rank {dp_rank}/{dp_size}]"
        
        # custom tags in kwargs
        extra_label = kwargs.get('extra_log_label')
        if extra_label:
            log_format += f" [{extra_label}]"

        log_format += " [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s"
        
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # 4. StreamHandler (console output)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # 5. FileHandler (file output)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
        # file_handler = logging.FileHandler(log_file, mode="a")

        file_handler = FlushingFileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 6. do not propagate upward, so the root logger does not print twice
        self.logger.propagate = False
        
        # configure the high-resolution machine log
        self.machine_id = extra_label if extra_label else f"Rank_{rank}"
        
        csv_path = os.path.join(log_dir, f"profile_rank_{rank}.csv")
        
        # buffering=1 means line buffered: every line is flushed automatically.
        # That matters for profiling -- it prevents losing recent data on a crash.
        self.profile_file = open(csv_path, "a", buffering=1, encoding="utf-8")
        
        # write the CSV header for a new file
        if os.path.getsize(csv_path) == 0:
            header = "timestamp_unix,readable_time,machine_id,step,event_name,event_type,duration_ms,metadata\n"
            self.profile_file.write(header)
        # 7. mark as configured
    
        self.profile_enabled = True
        self.is_configured = True
        
        if rank == 0:
            self.logger.info(f"GlobalLogger ready. Human logs: {log_file}, Machine logs: {csv_path}")


    def log_profile_event(self, timestamp: float, step: int, event_name: str, event_type: str, duration_ms: float = 0.0, metadata: str = ""):
            """
            Args:
                timestamp (float): must be a high-resolution float from time.time()
                step (int): current iteration
                event_name (str): operation name (e.g. "GPU_Encode")
                event_type (str): "START" or "END"
                duration_ms (float): GPU time, recorded on END events only; 0 on START
            """
            if not self.profile_enabled:
                return
            
            timestamp = timestamp + self.time_offset  # apply the clock offset

            # 1. a human-readable helper time (handy for grep, not used for plotting)
            # only the fractional part of the timestamp is kept
            readable = time.strftime("%H:%M:%S", time.localtime(timestamp)) + f".{int(timestamp % 1 * 1000):03d}"
            
            # 2. build the CSV line by hand -- faster than the csv module and one less dependency
            # format: timestamp, readable, machine, step, name, type, duration, meta
            line = f"{timestamp:.6f},{readable},{self.machine_id},{step},{event_name},{event_type},{duration_ms:.3f},{metadata}\n"
            
            # 3. write (buffering=1 flushes automatically)
            try:
                self.profile_file.write(line)
            except Exception:
                pass # profiling must never interrupt training

    def close(self):
        if self.profile_file:
            self.profile_file.close()
    def get_logger(self) -> logging.Logger:
        """
        Return the global logger instance.

        If it has not been configured yet, warn and fall back to a basic configuration.
        """
        if not self.is_configured:
            # a safe fallback so using the logger before setup does not crash the program
            self.logger.warning("GlobalLogger is being used before it was properly configured. "
                               "Applying a basic default configuration.")
            
            # console output only, to avoid failing silently
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter("[%(asctime)s] [UNCONFIGURED] - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            # mark as configured (basic) so the warning is printed once
            self.is_configured = True 

        return self.logger
    

# ==========================================================
# === NEW: Lightweight Global Accessor Function ===
# ==========================================================
def get_global_logger() -> logging.Logger:
    """
    A lightweight, global access point to the GlobalLogger singleton.

    This is the recommended way to get the logger from anywhere in the project.
    It hides the complexity of the singleton class and provides a simple,
    memorable function call.
    """
    return GlobalLogger().get_logger()