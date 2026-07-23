import torch
import sys
import torch.distributed as dist
import time

# from megatron.core.tensor_parallel.mappings import (
#     gather_from_tensor_model_parallel_region,
# )
import hashlib
import torch.nn as nn

# from .logging import get_logger
from .logger import get_logger





def print_model_params(model):
    print("Model Parameters:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(param.data)  # 仅打印数值，不计算梯度
            print("-" * 50)


def tensor_md5(tensor: torch.Tensor) -> str:
    tensor = tensor.to(torch.float64)
    # 确保 Tensor 在 CPU 上，并转换为 numpy 数组
    tensor_np = tensor.detach().cpu().numpy()
    # 将 numpy 数组转换为 bytes
    tensor_bytes = tensor_np.tobytes()
    # 计算 MD5
    md5_hash = hashlib.md5(tensor_bytes).hexdigest()
    return md5_hash


class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


filename = "_output_14_backward_2.log"


def register_hooks(model, print_values=True, max_elements=20):
    hooks = []

    def print_shape_and_values(tensor, label, max_elements=20):
        """Helper function to print shape and values of a tensor"""
        rank = 0
        with open(str(rank) + filename, "a") as f:
            if isinstance(tensor, torch.Tensor):
                get_logger().info(f"[rank {rank}] {label} shape: {tensor.shape}")
                if print_values:
                    if tensor.numel() < max_elements:
                        max_elements = tensor.numel()
                    if tensor.flatten()[0].dtype != torch.bool:
                        # get_logger().info(f"[rank {rank}]  {label} first 20 values: {tensor.flatten()[:max_elements]}{'...' if tensor.numel() > max_elements else ''} ")
                        # x, _ = torch.topk(tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} max 20 values: {x} ")
                        # x, _ = torch.topk(-tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} min 20 values: {x} ")
                        tensor = tensor.float()
                        x = torch.norm(tensor)
                        get_logger().info(f"[rank {rank}] {label} norm Values: {x} ")
                        # get_logger().info(f'[rank {rank}] {label} md5 value: {tensor_md5(tensor)} ')
                        get_logger().info(
                            f"[rank {rank}] {label} shape: {tensor.shape} "
                        )
            else:
                get_logger().info(f"[rank {rank}]  {label} is not tensor, is {tensor} ")

    def forward_hook_fn(module, input, output, name):
        """Hook 函数，打印输入和输出的 shape 和具体数值"""
        # print(f"Layer: {module.__class__.__name__}")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name}")
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight.data
                print_shape_and_values(weight, f"{name}_Weight")
            else:
                get_logger().info(f"[rank {rank}] Layer: {name} has no weight ")
        # 打印输入
        if isinstance(input, tuple):
            for idx, inp in enumerate(input):
                if isinstance(inp, tuple):  # Check for nested tuple
                    for sub_idx, sub_inp in enumerate(inp):
                        print_shape_and_values(sub_inp, f"Input {idx}-{sub_idx}")
                else:
                    print_shape_and_values(inp, f"{name}_Input {idx}")
        else:
            print_shape_and_values(input, "Input")

        # Print output shapes and values
        if isinstance(output, tuple):
            for i, tensor in enumerate(output):
                if isinstance(tensor, tuple):  # Check for nested tuple
                    for sub_idx, sub_tensor in enumerate(tensor):
                        print_shape_and_values(
                            sub_tensor, f"{name}_Output {i}-{sub_idx}"
                        )
                else:
                    print_shape_and_values(tensor, f"{name}_Output {i}")

        else:
            print_shape_and_values(output, "Output")
        with open(str(rank) + filename, "a") as f:
            get_logger().info("-" * 100)
            get_logger().info(" ")

    def backward_hook_fn(module, grad_input, grad_output, name):
        """Hook 函数，打印反向传播时的梯度"""
        # print(f"Layer: {module.__class__.__name__} (backward)")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name} (backward)")
            # 打印输入梯度
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Input {idx}")

            # 打印输出梯度
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Output {idx}")

            # if hasattr(module, 'weight') and module.weight is not None:
            #     print_shape_and_values(module.weight.grad, f"{name}_weight_grad")
            # if hasattr(module, 'bias') and module.bias is not None:
            #     print_shape_and_values(module.bias.grad, f"{name}_bias_grad")

            get_logger().info("-" * 100)
            get_logger().info(" ")

    # 遍历模型的所有层并注册 hook
    # for layer in model.modules():
    #     if not isinstance(layer, torch.nn.Sequential) and not isinstance(layer, torch.nn.ModuleList):
    #         forward_hook = layer.register_forward_hook(forward_hook_fn)
    #         backward_hook = layer.register_backward_hook(backward_hook_fn)
    #         hooks.append(forward_hook)
    #         hooks.append(backward_hook)

    # if dist.is_available() and dist.is_initialized():
    #     rank = 0
    # else:
    #     rank = 0

    def watch_parameter(param_name, param):
        rank = 0

        def param_hook(grad):
            if grad is not None:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad norm: {grad.norm()} ")
            else:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad is None ")

        param.register_hook(param_hook)

    for name, module in model.named_modules():
        forward_hook = module.register_forward_hook(
            lambda m, i, o, name=name: forward_hook_fn(m, i, o, name)
        )
        backward_hook = module.register_full_backward_hook(
            lambda m, i, o, name=name: backward_hook_fn(m, i, o, name)
        )
        hooks.append(forward_hook)
        hooks.append(backward_hook)

    for name, param in model.named_parameters():
        if param.requires_grad:
            watch_parameter(name, param)

    return hooks  # 返回 hook 句柄列表，方便后续清理


import time, os, re
from collections import defaultdict
import numpy as np
import logging
# from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
from logging import LoggerAdapter

class MyTimer:
    # __init__, start, stop, next_iteration, _gather_records, summarize, summarize_per_rank, dump 等方法保持不变...
    # (此处省略了之前已展示的、未改动的方法代码，以保持简洁)
    def __init__(self, use_cuda=True, tag="timer", verbose=True, log_dir="my_timer_log/"):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.tag = tag
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.log_dir = log_dir
        self._stage_times = {}
        self.records = []
        self.current_iteration = 0
        
        self.log_context = {'log_type': 'timing'}
        # self.logger = self._get_logger()
        self.logger = None
        # if verbose: self.logger.setLevel(logging.INFO)
        # else: self.logger.setLevel(logging.WARNING)



    def set_logger(self, logger_instance: logging.Logger):
        """
        【公共接口】允许外部项目注入自己的 logger 实例。
        注入后会立即用 LoggerAdapter 包装，以支持过滤器。
        """
        # 即使外部注入，也用 Adapter 包装以确保 log_context 存在
        self.logger = LoggerAdapter(logger_instance, self.log_context)


    def _create_default_logger(self) -> logging.Logger:
        """
        【全新】按您 handler 项目的样式，创建一个功能完备的默认 logger。
        """
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger = logging.getLogger(f"MyTimer.default_rank_{self.rank}")
        
        # 如果已经配置过，则直接返回，防止重复添加 handler
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        logger.propagate = False

        formatter = logging.Formatter(
            f"[%(asctime)s] [Rank {self.rank}] [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. 配置控制台 Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 2. 配置文件 Handler (所有日志)
        main_log_file = os.path.join(self.log_dir, f"timer_rank_{self.rank}.log")
        main_file_handler = logging.FileHandler(main_log_file, mode="a")
        main_file_handler.setFormatter(formatter)
        logger.addHandler(main_file_handler)
        
        return logger

    # def _get_logger(self):
    #     handler = DynamicForwardStepHandler()
    #     if handler.logger is None:
    #         return None
    #     adapter = LoggerAdapter(handler.logger, self.log_context)
    #     return adapter


    def _ensure_logger(self):
        """
        【全新核心逻辑】即时解析 Logger，解决初始化时序问题。
        """
        # 如果 logger 已经被外部通过 set_logger 注入，则什么都不做
        if self.logger:
            return

        # 1. 优先尝试获取 handler 项目的 logger
        raw_logger = None
        try:
            from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
            handler = DynamicForwardStepHandler()
            if handler.logger:
                raw_logger = handler.logger
        except (ImportError, AttributeError):
            # 导入失败或属性不存在，说明不在 handler 项目中，正常现象
            pass

        # 2. 如果没获取到，则创建我们自己的高质量默认 logger
        if raw_logger is None:
            raw_logger = self._create_default_logger()
            
        # 3. 无论来源如何，都用 LoggerAdapter 包装以添加上下文，使过滤器生效
        self.logger = LoggerAdapter(raw_logger, self.log_context)



    def start(self, stage_name):
        # if self.logger is None:
        #     self.logger = self._get_logger()
        self._ensure_logger()
        if self.rank != dist.get_rank():
            self.rank = dist.get_rank()
        entry = {"cpu_start": time.time()}
        if self.use_cuda:
            entry["cuda_start"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_end"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_start"].record()
        self._stage_times[stage_name] = entry

    def stop(self, stage_name):
        if stage_name not in self._stage_times:
            return
        entry = self._stage_times.pop(stage_name)
        cpu_end = time.time()
        cpu_elapsed_ms = (cpu_end - entry.get("cpu_start", cpu_end)) * 1000
        cuda_elapsed_ms = None
        if self.use_cuda and "cuda_end" in entry:
            entry["cuda_end"].record()
            torch.cuda.synchronize()
            cuda_elapsed_ms = entry["cuda_start"].elapsed_time(entry["cuda_end"])
        self.records.append(
            {
                "stage": stage_name,
                "rank": self.rank,
                "iteration": self.current_iteration,
                "cpu_duration_ms": cpu_elapsed_ms,
                "cuda_duration_ms": cuda_elapsed_ms,
            }
        )
        if self.verbose:
            self.logger.info(
                f"[Iter {self.current_iteration}] Stage '{stage_name}': CPU {cpu_elapsed_ms:.3f}ms, CUDA {cuda_elapsed_ms or 0.0:.3f}ms"
            )

    def next_iteration(self):
        self.current_iteration += 1
        self.logger.info(
            f"--- MyTimer: Switched to iteration {self.current_iteration} ---"
        )

    def _gather_records(self):
        if not dist.is_initialized() or self.world_size == 1:
            return self.records
        all_records = None
        if self.rank == 0:
            all_records_list = [None] * self.world_size
            dist.gather_object(self.records, all_records_list, dst=0)
            all_records = [item for sublist in all_records_list for item in sublist]
        else:
            dist.gather_object(self.records, None, dst=0)
        return all_records

    def dump(self, sort_records: bool = False):
        """
        【修改后】将当前 Rank 的原始计时记录追加到日志文件中。

        Args:
            sort_records (bool, optional): 是否在写入前对记录进行排序。
                                         默认为 False，即按原始执行顺序写入。
                                         设置为 True 则按 iteration 和 stage name 排序。
        """
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{self.tag}_rank{self.rank}.log")

        # 【修改点 1】: 根据 sort_records 参数决定是否排序
        if sort_records:
            records_to_write = sorted(
                self.records, key=lambda x: (x["iteration"], x["stage"])
            )
            sort_info = "(Sorted)"
        else:
            # 默认情况下，直接使用原始记录列表，保留执行顺序
            records_to_write = self.records
            sort_info = "(Execution Order)"

        with open(log_path, "a") as f:
            f.write(
                f"\n==================== DUMP {sort_info} (Up to Iteration {self.current_iteration}) ====================\n"
            )

            # 【修改点 2】: 遍历处理后的列表
            for r in records_to_write:
                # 确保 cuda_duration_ms 存在，提供一个默认值以避免错误
                cuda_time = r.get("cuda_duration_ms")
                cuda_str = (
                    f"{cuda_time:>8.3f}ms" if cuda_time is not None else "N/A".rjust(8)
                )

                f.write(
                    f"[Iter {r['iteration']}][Rank {r['rank']}] Stage: {r['stage']:<30} | "
                    f"CPU: {r['cpu_duration_ms']:>8.3f}ms, CUDA: {cuda_str}\n"
                )
    def generate_report(self, stage_pattern, output_filename, iteration_filter=None):
        """
        【最终正确版 V2】生成详细的性能分析报告，并保存到文件。
        此版本会聚合所有 rank 的数据，按独立的 Stage Name 进行统一统计。
        """
        # _gather_records() 应该返回一个包含所有 rank 记录的列表
        all_records = self._gather_records()

        if self.rank == 0:
            # 1. 筛选符合条件的记录 (逻辑不变)
            pattern = re.compile(stage_pattern)
            filtered_records = [
                r
                for r in all_records
                if pattern.match(r["stage"])
                and (iteration_filter is None or iteration_filter(r["iteration"]))
            ]

            if not filtered_records:
                self.logger.warning(
                    f"No records found for pattern '{stage_pattern}' to generate report."
                )
                return {}

            # --- 核心修改区域 ---

            # 2. 按 stage_name 对所有 rank 的数据进行分组
            # 新逻辑：将所有 rank 中 name 相同的 stage 聚合在一起
            grouped_data = defaultdict(list)
            for r in filtered_records:
                # 不再关心 r["rank"]，只要 stage name 相同，就聚合
                if r["cuda_duration_ms"] is not None:
                    grouped_data[r["stage"]].append(r["cuda_duration_ms"])

            # 3. 计算每个聚合后 stage 的统计数据
            report_data = {} # 不再需要按 rank 分组
            for stage_name, durations in grouped_data.items():
                # 每个 stage_name 都是一个独立的条目，其 durations 是来自所有 rank 的数据列表
                report_data[stage_name] = {
                    "count": len(durations),
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                }
            
            # --- 修改结束 ---

            # 4. 生成格式化的报告字符串 (现在只有一个聚合后的总表)
            report_string = ""
            report_header = (
                f"--- 📊 Aggregated Performance Report (All Ranks) ---\n"
                f"Pattern: '{stage_pattern}'\n"
                f"Filename: {output_filename}\n"
                f"{'-'*80}\n"
            )
            report_string += report_header

            # 不再需要 for rank_id in ... 的循环
            report_string += f"\n[Aggregated Statistics]\n"
            report_string += f" {'STAGE':<60} {'COUNT':<7} {'MEAN (ms)':<12} {'MEDIAN (ms)':<13} {'STD (ms)':<12}\n"
            report_string += f" {'-'*59} {'-'*6} {'-'*11} {'-'*12} {'-'*11}\n"

            # 表格内容
            for stage_name in sorted(report_data.keys()):
                stats = report_data[stage_name]
                report_string += (
                    f" {stage_name:<60} {stats['count']:<7} "
                    f"{stats['mean']:<12.3f} {stats['median']:<13.3f} {stats['std']:<12.3f}\n"
                )

            # 5. 打印到控制台 (无需改动)
            print(report_string)

            # 6. 保存到文件 (无需改动)
            if self.log_dir:
                log_path = os.path.join(self.log_dir, output_filename)
                # os.makedirs(self.log_dir, exist_ok=True)
                # log_path = os.path.join(self.log_dir, output_filename)
                try:
                    with open(log_path, "w") as f:
                        f.write(report_string)
                    self.logger.info(f"Report successfully saved to '{log_path}'")
                except IOError as e:
                    self.logger.error(f"Failed to save report to '{log_path}': {e}")
            else:
                self.logger.warning("log_dir not set, cannot save report file.")

            return report_data

        return None
   
    def generate_csv(
        self, report_data: dict, csv_filename: str = "suffix_median_report.csv"
    ):
        """
        从 generate_report 的结果中提取后缀参数 bs/f/h/w/sp 和中位数，并保存为 CSV 文件。

        Args:
            report_data (dict): generate_report 返回的字典，结构为 report_data[rank][suffix]。
            csv_filename (str): 要保存的 CSV 文件名（默认为 suffix_median_report.csv）
        """
        import csv

        if self.rank != 0:
            return  # 只在 Rank 0 上执行

        if self.log_dir is None:
            self.logger.warning("log_dir not set, cannot save CSV file.")
            return

        csv_path = os.path.join(self.log_dir, csv_filename)

        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["bs", "f", "h", "w", "sp", "median"])  # 表头

                for suffix, stats in report_data[0].items():
                    match = re.match(
                        r"bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp(\d+)", suffix
                    )
                    if match:
                        bs, f, h, w, sp = match.groups()
                        median = stats["median"]
                        writer.writerow([bs, f, h, w, sp, median])

            self.logger.info(f"CSV report successfully saved to '{csv_path}'")

        except Exception as e:
            self.logger.error(f"Failed to save CSV report to '{csv_path}': {e}")


# 保持你原来的全局实例用法
global_timer = MyTimer()


def print_cuda_memory_gb(step_name=""):
    """
    打印当前进程（Rank）的已分配和已缓存的 CUDA 显存。
    单位为 GB。
    """
    # 确保 CUDA 可用且分布式环境已初始化
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"✅ [Rank {rank}] [CUDA Memory] {step_name}: "
            f"Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB"
        )
    else:
        # 如果没有分布式环境或 CUDA，只打印普通信息
        print(f"✅ {step_name}: CUDA not available or distributed not initialized.")



import threading
import traceback

class DebuggingEvent(threading.Event):
    """
    一个增强的 Event 类，用于调试。
    它在初始化时接收一个 logger 对象，并在 .set() 方法被调用时，
    使用该 logger 记录堆栈信息。
    """
    def __init__(self, *args, logger=None, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        
        # 保存 logger 对象，如果未提供，则创建一个默认的 print logger
        if logger:
            self.logger = logger
        else:
            # Fallback: 如果没有提供 logger，就退回到打印到控制台的行为
            self.logger = logging.getLogger("DebuggingEvent")
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def set(self):
        # 使用 StringIO 来捕获堆栈信息，而不是直接打印
        import io
        s = io.StringIO()
        traceback.print_stack(file=s)
        stack_info = s.getvalue()
        s.close()

        # 使用我们保存的 logger 来记录信息
        self.logger.info(
            f"\n{'='*30} [EVENT SET TRACE] {'='*30}\n"
            f">>> Event object {id(self)} is being set() by:\n"
            f"{stack_info}"
            f"{'='*80}"
        )
        
        # 调用父类的原始 set 方法
        super().set()



def record_oom_threshold(failing_bs: int, failing_frame: int, step: int = 4):
    """
    当发生OOM时，记录下对应batch size的安全帧数上限。

    这个函数是幂等的：
    1. 如果文件不存在，会自动创建。
    2. 如果记录已存在，只有在新的上限更严格（更小）时才会更新。

    Args:
        failing_bs (int): 导致OOM的batch size。
        failing_frame (int): 导致OOM的帧数。
        step (int): 帧数递增的步长，用于计算上一个安全点。
    """
    import json
    # from megatron.core import mpu
    sp = os.environ.get('sp')
    model = os.environ.get('model_type') 
    resolution = os.environ.get('resolution')

    threshold_file = f"oom_thresholds_{model}_{resolution}_sp{sp}.json"
    print(f"--- OOM Detected! Recording threshold for bs={failing_bs} ---")
    
    # 根据失败的帧数，计算上一个已知的“安全”点
    # 例如：frame=73 失败了, 上限则为 69 (73-4)
    new_max_frame = failing_frame - step
    
    # 读取已有的阈值文件，如果不存在或为空则创建一个新的字典
    thresholds = {}
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content: # 确保文件不为空
                    thresholds = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read or parse '{threshold_file}'. Starting with empty thresholds.")
            thresholds = {} # 出错时重置
    
    # JSON的key必须是字符串
    failing_bs_str = str(failing_bs)
    
    # 获取当前bs已记录的上限，如果不存在则设为无穷大
    current_max = thresholds.get(failing_bs_str, float('inf'))
    
    # 只有在新的上限比旧的更严格（更小）时才更新
    if new_max_frame < current_max:
        print(f"Updating bs={failing_bs} max frame from {current_max} to {new_max_frame}")
        thresholds[failing_bs_str] = new_max_frame
        
        # 将更新后的阈值漂亮地写回文件
        with open(threshold_file, 'w', encoding='utf-8') as f:
            json.dump(thresholds, f, indent=4)
            print(f"Successfully saved new thresholds to '{threshold_file}'.")
    else:
        print(f"New max frame ({new_max_frame}) is not stricter than existing ({current_max}). No update needed.")



def print_tensor_info(tensor: torch.Tensor, name: str = ""):
        """
        将一个 PyTorch Tensor 的 Rank, Shape, Device, 和 Dtype 打印在同一行。

        Args:
            tensor (torch.Tensor): 需要检查的 PyTorch Tensor.
            name (str, optional): Tensor 的名字，用于在打印时区分. Defaults to "".
        """
        if not isinstance(tensor, torch.Tensor):
            print(f"提供的输入 '{name}' 不是一个有效的 PyTorch Tensor。")
            return

        # 使用 f-string 将所有信息格式化到一行
        # 如果提供了 name，则在前面加上 "name: "
        prefix = f"{name}: " if name else ""
        print(
            f"{prefix}"
            f"Rank {dist.get_rank()}, "
            f"Shape={tensor.shape}, "
            f"Device='{tensor.device}', "
            f"Dtype={tensor.dtype}"
        )
