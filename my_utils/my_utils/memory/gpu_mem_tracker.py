import torch
import threading
import time
import json
import matplotlib.pyplot as plt
import torch.distributed as dist

# try to import pynvml and record whether it is available
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

class GPU_Performance_Tracker:
    def __init__(self, prefix='', interval=1, use_distributed: bool = True, track_hardware_metrics: bool = True):
        """
        Initialise the GPU performance tracker.
        :param prefix: output filename prefix
        :param interval: sampling interval in seconds
        :param use_distributed: whether this runs in a distributed setting
        :param track_hardware_metrics: also track GPU hardware metrics (utilisation, power, ...)
        """
        self.interval = interval
        self.prefix = prefix
        
        # --- basic attributes ---
        self._monitor_thread = None
        self._is_running = False
        
        # --- distributed ---
        self.use_distributed = use_distributed and dist.is_initialized()
        self.device = f'cuda:{dist.get_rank()}' if self.use_distributed else 'cuda:0'
        self.rank = dist.get_rank() if self.use_distributed else 0

        # --- data containers ---
        self._reset_metrics() # initialise every container
        
        # --- NVML hardware monitoring ---
        self.track_hardware_metrics = track_hardware_metrics and HAS_PYNVML
        self.nvml_handle = None
        if self.track_hardware_metrics:
            try:
                pynvml.nvmlInit()
                # resolve the GPU handle for the current PyTorch device
                device_index = torch.cuda.current_device()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                print(f"[Rank {self.rank}] PyNVML initialized for GPU {device_index}.")
            except Exception as e:
                print(f"[Rank {self.rank}] Warning: Failed to initialize PyNVML. Hardware metrics will be disabled. Error: {e}")
                self.track_hardware_metrics = False
    
    def __del__(self):
        """Close NVML on destruction so the handle is not leaked."""
        if self.track_hardware_metrics:
            pynvml.nvmlShutdown()

    def _reset_metrics(self):
        """Reset every data container."""
        # PyTorch Memory Metrics
        self.allocated_memory_gb = []
        self.reserved_memory_gb = []
        
        # NVML Hardware Metrics
        self.gpu_util_percent = []
        self.mem_util_percent = []
        self.power_watts = []
        
        # Timestamps
        self.timestamps = []
        self.counter = 0

    def _monitor_metrics(self):
        """Background thread: sample all metrics on a fixed interval."""
        while self._is_running:
            # 1. PyTorch memory usage (in GB)
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.allocated_memory_gb.append(allocated)
            self.reserved_memory_gb.append(reserved)

            # 2. NVML hardware metrics, if enabled
            if self.track_hardware_metrics:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    self.gpu_util_percent.append(util.gpu)
                    self.mem_util_percent.append(util.memory)
                    
                    power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    self.power_watts.append(power_milliwatts / 1000.0)
                except pynvml.NVMLError as e:
                    print(f"[Rank {self.rank}] Warning: NVMLError while sampling hardware metrics: {e}")
                    # on a failed sample, pad with defaults to keep the series aligned
                    self.gpu_util_percent.append(0)
                    self.mem_util_percent.append(0)
                    self.power_watts.append(0)

            self.timestamps.append(self.counter * self.interval)
            self.counter += 1
            time.sleep(self.interval)

    def start_monitoring(self):
        """Start monitoring."""
        if not self._is_running:
            print(f'[Rank {self.rank}] Starting performance monitoring...')
            self._is_running = True
            self._reset_metrics() # drop any data from a previous run
            self._monitor_thread = threading.Thread(target=self._monitor_metrics, daemon=True)
            self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and produce the report."""
        if not self._is_running:
            return
            
        self._is_running = False
        print(f'[Rank {self.rank}] Stopping performance monitoring...')
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        
        self.plot_usage_graphs()

        max_alloc_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
        max_reserv_gb = torch.cuda.max_memory_reserved(self.device) / 1024**3
        print(f'[Rank {self.rank}]: Peak Memory -> Allocated: {max_alloc_gb:.2f} GB, Reserved: {max_reserv_gb:.2f} GB')
        
        # reset PyTorch's peak statistics
        torch.cuda.reset_peak_memory_stats(self.device)

    def plot_usage_graphs(self):
        """Plot memory, utilisation and power on one figure."""
        if not self.timestamps:
            print(f'[Rank {self.rank}] No data to plot.')
            return
            
        num_plots = 2 if self.track_hardware_metrics else 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
        fig.suptitle(f'GPU Performance Profile on Rank {self.rank}', fontsize=16)

        # --- subplot 1: memory usage ---
        ax1 = axes[0] if num_plots > 1 else axes
        ax1.plot(self.timestamps, self.allocated_memory_gb, label="PyTorch Allocated Memory (GB)", color='blue')
        ax1.plot(self.timestamps, self.reserved_memory_gb, label="PyTorch Reserved Memory (GB)", color='cyan', linestyle='--')
        ax1.set_ylabel("Memory (GB)")
        ax1.set_title("Memory Usage")
        ax1.legend()
        ax1.grid(True)

        # --- subplot 2: hardware utilisation and power ---
        if self.track_hardware_metrics:
            ax2 = axes[1]
            # twin y-axes
            ax2_power = ax2.twinx()

            # utilisation
            p1, = ax2.plot(self.timestamps, self.gpu_util_percent, label="GPU Utilization (%)", color='green')
            p2, = ax2.plot(self.timestamps, self.mem_util_percent, label="Memory Bandwidth Util (%)", color='orange', linestyle=':')
            ax2.set_ylabel("Utilization (%)")
            ax2.set_ylim(0, 105) # utilisation caps at 100
            
            # power
            p3, = ax2_power.plot(self.timestamps, self.power_watts, label="Power (Watts)", color='red', linestyle='-.')
            ax2_power.set_ylabel("Power (W)")
            
            ax2.set_title("Hardware Utilization & Power")
            ax2.legend(handles=[p1, p2, p3], loc='upper left')
            ax2.grid(True)
        
        plt.xlabel(f"Time (seconds, interval={self.interval}s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # leave room for the suptitle
        
        save_path = f'{self.prefix}_rank{self.rank}_performance_profile.png'
        plt.savefig(save_path)
        print(f'[Rank {self.rank}] Performance profile plot saved to {save_path}')
        plt.close(fig) # close the figure to release memory