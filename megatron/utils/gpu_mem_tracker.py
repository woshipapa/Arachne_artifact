import torch
import threading
import time
import json
import matplotlib.pyplot as plt
import torch.distributed as dist
class GPU_Memory_Tracker:

    def line_init(self):
        self.allocated_memory = []
        self.reserved_memory = []
        self.timestamps = []

    def __init__(self, prefix='',interval=1,
                 use_distributed: bool = True):
        self.interval = interval
        self.allocated_memory = []
        self.reserved_memory = []
        self.timestamps = []
        self._monitor_thread = None
        self._is_running = False
        self.counter = 0
        self.device = f'cuda:{dist.get_rank()}' if use_distributed else 'cuda:0'
        self.rank = dist.get_rank() if use_distributed else 0
        self.prefix = prefix

        self.memory_log_file = 'memory_log.json'

    def get_allocated_memory(self):
        return torch.cuda.memory_allocated(self.device) / 1024**3

    def _monitor_memory(self):
        while self._is_running:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)

            self.allocated_memory.append(allocated / 1024**3)
            self.reserved_memory.append(reserved / 1024**3)
            self.timestamps.append(self.counter)
            self.counter = self.counter+1
            time.sleep(self.interval)

    def start_monitoring(self):
        if not self._is_running:
            print(f'[Rank {self.rank}]start monitoring....')
            self._is_running = True
            self.line_init()
            self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self._monitor_thread.start()

    def stop_monitoring(self):
        self._is_running = False
        self.counter = 0
        print(f'[Rank {self.rank}] stop monitoring....')
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        # if self.rank == 0:   
        self.plot_memory_usage()

        print(f'[Rank {self.rank}]: max allocated_mm is {torch.cuda.max_memory_allocated() / 1024**3 }, max reserved_mm is {torch.cuda.max_memory_reserved() / 1024**3 }')
        # reset max_allocated_memory
        torch.cuda.reset_peak_memory_stats()
        # self.line_init() 
        # self.record_max_allocated_memory()    

    def record_max_allocated_memory(self):

        max_memory = torch.cuda.max_memory_allocated()

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        try:
            with open(self.memory_log_file, 'r') as f:
                memory_data = json.load(f)
        except FileNotFoundError:
            memory_data = []
        memory_summary = torch.cuda.memory_summary(self.device)
        memory_data.append({'timestamp': timestamp, 'max_memory_mb': max_memory / 1024 ** 3,'memory_summary': memory_summary})

        with open(self.memory_log_file,'w') as f:
            json.dump(memory_data, f, indent=4)


    def plot_memory_usage(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.allocated_memory, label="Allocated Memory (GB)")
        plt.plot(self.timestamps, self.reserved_memory, label="Reserved Memory (GB)")

        # print(f'[memory_tracker]: len(times)={len(self.timestamps)}, len(allocated)={len(self.allocated_memory)}, len(res) = {len(self.reserved_memory)}')
        plt.xlabel("Time (s)")
        plt.ylabel("Memory (GB)")
        plt.title(f"GPU Memory Usage Over Time({self.device})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_train_memory_usage.png')
        print('plot_memory_usage completed!')