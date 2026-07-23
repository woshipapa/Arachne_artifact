import torch
import sys
import torch.distributed as dist
import time
from typing import Any, Optional

# from megatron.core.tensor_parallel.mappings import (
#     gather_from_tensor_model_parallel_region,
# )
import hashlib
import torch.nn as nn

# from .logging import get_logger
# from .logger import get_logger

from contextlib import contextmanager



def print_model_params(model):
    print("Model Parameters:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(param.data)  # values only; no gradient computation
            print("-" * 50)


def tensor_md5(tensor: torch.Tensor) -> str:
    tensor = tensor.to(torch.float64)
    # move the tensor to CPU and convert it to a numpy array
    tensor_np = tensor.detach().cpu().numpy()
    # convert the numpy array to bytes
    tensor_bytes = tensor_np.tobytes()
    # compute the MD5
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
            elif False:
                get_logger().info(f"[rank {rank}]  {label} is not tensor, is {tensor} ")

    def forward_hook_fn(module, input, output, name):
        """Hook that prints the shapes and values of the inputs and outputs."""
        # print(f"Layer: {module.__class__.__name__}")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name}")
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight.data
                print_shape_and_values(weight, f"{name}_Weight")
            elif False:
                get_logger().info(f"[rank {rank}] Layer: {name} has no weight ")
        # print the inputs
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
        """Hook that prints gradients during the backward pass."""
        # print(f"Layer: {module.__class__.__name__} (backward)")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name} (backward)")
            # print the input gradients
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Input {idx}")

            # print the output gradients
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Output {idx}")

            # if hasattr(module, 'weight') and module.weight is not None:
            #     print_shape_and_values(module.weight.grad, f"{name}_weight_grad")
            # if hasattr(module, 'bias') and module.bias is not None:
            #     print_shape_and_values(module.bias.grad, f"{name}_bias_grad")

            get_logger().info("-" * 100)
            get_logger().info(" ")

    # walk every layer of the model and register the hook
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

    return hooks  # handles are returned so they can be removed later


import time, os, re
from collections import defaultdict
import numpy as np
import logging
# from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
from logging import LoggerAdapter
from ..tracing.nvtx_utils import LabelerProtocol, NoOpLabeler, create_labeler

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class MyTimer:
    # __init__, start, stop, next_iteration, _gather_records, summarize,
    # summarize_per_rank and dump are unchanged and omitted here for brevity.
    def __init__(self, use_cuda=True, tag="timer", 
                 verbose=True, log_dir="my_timer_log/",
                 profile_memory=False, use_nvtx=False,
                 labeler: Optional[LabelerProtocol] = None,
                 nvtx_domain: Optional[str] = None):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.tag = tag
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.log_dir = log_dir
        
        self.records = []
        self.current_iteration = 0
        
        self.profile_memory = profile_memory and PSUTIL_AVAILABLE and self.use_cuda
        self.log_context = {'log_type': 'timing'}
        # self.logger = self._get_logger()

        # either wait for configuration to be passed in, or look it up directly (coupled variant)
        self.logger = None
        # if verbose: self.logger.setLevel(logging.INFO)
        # else: self.logger.setLevel(logging.WARNING)

        self.set_labeler(
            labeler if labeler is not None else create_labeler(
                enabled=bool(use_nvtx),
                default_domain=nvtx_domain,
            )
        )


        # async
        # records finished this iteration whose duration is not yet computed
        self._pending_records = []

        # self._stage_times = {}

        # --- hierarchical stack ---
        
        # 1. (replaces) self._stage_times = {}
        # 2. a unique node id, used to rebuild the parent/child
        #    relationships during 'summarize'.
        self.next_node_id = 1 # 0 is the root
        
        # 3. the root node. current_node always points at the top
        #    of the stack.
        self.root_node = {
            "name": "root",
            "node_id": 0,
            "parent_id": None,
            "start_cpu": time.perf_counter(),
            "children": [] # for debugging; the real data lives in records
        }

        self.current_node = self.root_node

    def _get_domain(self, domain_name: str):
        raise RuntimeError("MyTimer no longer exposes raw NVTX domains; use register_stage() via the configured labeler.")
        """Fetch and cache the Domain object."""
        if domain_name is None:
            return None
        if domain_name not in self._domains:
            # create and cache a new Domain when it does not exist yet
            self._domains[domain_name] = nvtx.get_domain(domain_name)

        return self._domains[domain_name]

    def set_labeler(self, labeler: Optional[LabelerProtocol]) -> None:
        self.labeler = labeler if labeler is not None else NoOpLabeler()
        self.use_nvtx = bool(getattr(self.labeler, "enabled", False))

    def register_stage(self, stage_name: str, color: str = "blue", domain_name: str = None, category = None):
        if not self.use_nvtx:
            return
        self.labeler.register_label(
            stage_name,
            color=color,
            domain_name=domain_name,
            category=category,
        )
        return
            
        if domain_name is None:
            raise ValueError("Pre-registration for NVTX optimization requires a valid `domain_name`.")
            
        domain = self._get_domain(domain_name)
        
        # nvtx.Domain's get_event_attributes method
        attrs = domain.get_event_attributes(
            message=stage_name, color=color, category=category
        )
        
        self._registered_attrs[stage_name] = (domain, attrs)

    def disable_cuda_time(self):
        self.use_cuda = False

    def set_logger(self, logger_instance: logging.Logger):
        """
        Public API: let an external project inject its own logger instance.
        It is wrapped in a LoggerAdapter immediately so filters keep working.
        """
        # wrap even an injected logger in the adapter, so log_context exists
        # logger_instance is the logger returned by global_logger.get_logger()
        self.logger = LoggerAdapter(logger_instance, self.log_context)
        

        # singleton
        global_logger = GlobalLogger()

        setattr(self.logger, 'log_profile_event', global_logger.log_profile_event)

    def _create_default_logger(self) -> logging.Logger:
        """
        Build a fully configured default logger.
        """
        # make sure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger = logging.getLogger(f"MyTimer.default_rank_{self.rank}")
        
        # already configured: return early so handlers are not added twice
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        logger.propagate = False

        formatter = logging.Formatter(
            f"[%(asctime)s] [Rank {self.rank}] [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 2. file handler (all logs)
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
        Resolve the logger lazily, which avoids an initialisation-ordering problem.
        """
        # nothing to do when a logger was already injected via set_logger
        if self.logger:
            return

        # 1. prefer the host project's logger
        raw_logger = None
        try:
            from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
            handler = DynamicForwardStepHandler()
            if handler.logger:
                raw_logger = handler.logger
        except (ImportError, AttributeError):
            # import failure or missing attribute simply means we are not in that project
            pass

        # 2. otherwise fall back to the default global logger created at startup
        if raw_logger is None:
            from .logger import GlobalLogger
            raw_logger = GlobalLogger().get_logger()
            
        # 3. whatever the source, wrap it in a LoggerAdapter so the filters apply
        self.logger = LoggerAdapter(raw_logger, self.log_context)

    @contextmanager
    def time_stage(self, stage_name: str):
        """
        A context manager for timing a block of code conveniently and safely.

        Args:
            stage_name (str): name of the stage or block being timed.
        
        Usage:
            timer = MyTimer()
            with timer.time_stage('data_loading'):
                # your code to be timed
                time.sleep(1)
        """
        self.start(stage_name)
        try:
            yield
        finally:
            self.stop(stage_name)

    def start(self, stage_name: str, 
              color: str = "blue",
              domain_name: str = None):
        # if self.logger is None:
        #     self.logger = self._get_logger()
        # the logger used to be wired in here; it is now passed in after the Megatron distributed init
        self._ensure_logger()
        if torch.distributed.is_initialized() and self.rank != dist.get_rank():
            self.rank = dist.get_rank()
        # entry = {"cpu_start": time.time()}
        # CPU code start time
        entry = {"cpu_start": time.perf_counter()}
        
        # GPU profile using cudaEvent
        if self.use_cuda:
            entry["cuda_start"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_end"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_start"].record()



        if self.use_nvtx:
            entry["nvtx_token"] = self.labeler.start(
                stage_name,
                color=color,
                domain_name=domain_name,
            )
            entry["nvtx_domain"] = None  # tells stop() which end_range to call
            entry["nvtx_range_id"] = None

            if False:
                # fast path: use the pre-cached (domain, attrs)
                domain, attrs = self._registered_attrs[stage_name]
                entry["nvtx_domain"] = domain
                entry["nvtx_range_id"] = domain.start_range(attributes=attrs)
            elif False:
                # exploratory path: build it on the fly
                entry["nvtx_range_id"] = None
                # entry["nvtx_domain"] stays None, marking the global-function path


        new_node_id = self.next_node_id
        self.next_node_id += 1
        
        new_node = {
            "name": stage_name,
            "node_id": new_node_id,
            "parent_id": self.current_node["node_id"],
            "children": [],
            # (used when stop() walks back up the stack)
            "parent": self.current_node,
            "abs_start_time": time.time(),
            # (holds everything in the 'entry' dict)
            "cpu_start": entry["cpu_start"],
            "cuda_start": entry.get("cuda_start"),
            "cuda_end":  entry.get("cuda_end"),
            "nvtx_domain": entry.get("nvtx_domain"),
            "nvtx_range_id": entry.get("nvtx_range_id"),
            "nvtx_token": entry.get("nvtx_token"),
        }

        self.current_node["children"].append(new_node)
        self.current_node = new_node
        # self._stage_times[stage_name] = entry

    # def stop(self, stage_name):
    #     if stage_name not in self._stage_times:
    #         return
    #     entry = self._stage_times.pop(stage_name)
    #     # cpu_end = time.time()
    #     cpu_end = time.perf_counter()
    #     cpu_elapsed_ms = (cpu_end - entry.get("cpu_start", cpu_end)) * 1000
    #     cuda_elapsed_ms = None
    #     if self.use_cuda and "cuda_end" in entry:
    #         entry["cuda_end"].record()
    #         # torch.cuda.synchronize()
    #         # cuda_elapsed_ms = entry["cuda_start"].elapsed_time(entry["cuda_end"])

    #     if self.use_nvtx and "nvtx_range_id" in entry:
    #         domain = entry.get("nvtx_domain")
    #         range_id = entry.get("nvtx_range_id")

    #         if range_id is not None:
    #             if domain:
    #                 # if start was called on a domain object, end must be too
    #                 domain.end_range(range_id)
    #             else:
    #                 # if start used the global function, end must as well
    #                 nvtx.end_range(range_id)



    #      # stage the record: CPU time is known, the GPU event is recorded but not yet timed
    #     pending_record = {
    #         "stage": stage_name,
    #         "cpu_duration_ms": (cpu_end - entry["cpu_start"]) * 1000,
    #         "cuda_events": (entry.get("cuda_start"), entry.get("cuda_end"))
    #     }
    #     self._pending_records.append(pending_record)    
    #     # self.records.append(
    #     #     {
    #     #         "stage": stage_name,
    #     #         "rank": self.rank,
    #     #         "iteration": self.current_iteration,
    #     #         "cpu_duration_ms": cpu_elapsed_ms,
    #     #         "cuda_duration_ms": cuda_elapsed_ms,
    #     #     }
    #     # )

    #     # # use the injected logger to record the timer data
    #     # if self.verbose:
    #     #     self.logger.info(
    #     #         f"[Iter {self.current_iteration}] Stage '{stage_name}': CPU {cpu_elapsed_ms:.3f}ms, CUDA {cuda_elapsed_ms or 0.0:.3f}ms"
    #     #     )
    def stop(self, stage_name: str):
        cpu_end = time.perf_counter()

        # (V2) 1. stack check -- the most important part
        if self.current_node["name"] != stage_name:
            # robustness: on a name mismatch, search back up the stack
            # (this handles "stop('A')" implicitly closing "B")
            
            print(f"TimerWarning: Mismatched stop call on Rank {self.rank}! "
                  f"Expected to stop '{self.current_node['name']}' but got '{stage_name}'.")
            
            node_to_stop = self._find_node_in_stack(stage_name)
            
            if node_to_stop is None:
                print(f"TimerError: Could not find active timer '{stage_name}' in the stack.")
                return

            # once found, every child must be closed automatically until we reach it
            # 'node_to_stop'。
            while self.current_node != node_to_stop:
                print(f"TimerWarning: Auto-stopping child '{self.current_node['name']}' "
                      f"due to explicit stop of ancestor '{stage_name}'.")
                # (pass cpu_end, the only "stop" time available)
                self._finalize_and_record_node(self.current_node, cpu_end) 
                self.current_node = self.current_node['parent']
            
        # (V2) 2. finalise and record the current node
        self._finalize_and_record_node(self.current_node, cpu_end)

        # (V2) 3. ascend
        # (we *always* ascend to the current node's parent)
        if self.current_node["parent"] is not None:
            self.current_node = self.current_node["parent"]
        else:
            print(f"TimerError: Attempted to stop root node?")


    def _find_node_in_stack(self, name: str):
        """ Helper: search up the stack from the current node. """
        temp_node = self.current_node
        while temp_node is not None and temp_node["name"] != "root":
            if temp_node["name"] == name:
                return temp_node
            temp_node = temp_node["parent"]
        return None
    
    def _finalize_and_record_node(self, node: dict, cpu_end: float):
        """ Helper: holds the core logic of the 'stop' method. """
        
        # (V2) this is 90% of the 'stop' method
        
        # (from 'stop')
        if self.use_cuda and "cuda_end" in node and node["cuda_end"] is not None:
            node["cuda_end"].record()

        if self.use_nvtx:
            self.labeler.stop(node.get("nvtx_token"))

        if self.use_nvtx and "nvtx_range_id" in node:
            domain = node.get("nvtx_domain")
            range_id = node.get("nvtx_range_id")
            if range_id is not None:
                if domain:
                    domain.end_range(range_id)
                else:
                    # print(f"Timer NVTX: Using global end_range for '{node['name']}'")
                    pass

        # (from 'stop': create the pending_record)
        pending_record = {
            "stage": node["name"],
            "cpu_duration_ms": (cpu_end - node["cpu_start"]) * 1000,
            "cuda_events": (node.get("cuda_start"), node.get("cuda_end")),
            "abs_start_time": node.get("abs_start_time", 0.0),
            # (V2) the hierarchy fields
            # (these let 'summarize' rebuild the tree)
            "node_id": node["node_id"],
            "parent_id": node["parent_id"]
        }
        self._pending_records.append(pending_record)

    def set_step(self, iteration: int):
        self.current_iteration = iteration
    def synchronize_and_log(self):
        """
        [V2 - hierarchical]
        Called at the end of an iteration. Synchronises, then does three things:
        1. compute the CUDA time of every pending record and move it into self.records;
        2. rebuild this iteration's call tree from self.records (a flat list);
        3. walk that tree, compute self time, and log it in indented form.
        """
        if self.use_cuda:
            torch.cuda.synchronize()

        # --- step 1: drain _pending_records into self.records ---
        
        # clear this iteration's records
        self.records = []
        
        for record in self._pending_records:
            cuda_elapsed_ms = None
            if self.use_cuda:
                start_event, end_event = record["cuda_events"]
                if start_event and end_event:
                    # make sure the events are ready
                    try:
                        cuda_elapsed_ms = start_event.elapsed_time(end_event)
                    except torch.cuda.Error as e:
                        # (handle a possible CUDA error)
                        self.logger.warning(f"CUDA event error for {record['stage']}: {e}")
            
            abs_start_ts = record.get("abs_start_time", 0.0)
            
            # B. pick the duration (CUDA time when available, CPU time for CPU-only ops)
            final_duration_ms = cuda_elapsed_ms if cuda_elapsed_ms is not None else record["cpu_duration_ms"]
            
            # C. derive the end timestamp (start + duration)
            # note the ms to s conversion
            abs_end_ts = abs_start_ts + (final_duration_ms / 1000.0)

            self.logger.log_profile_event(
                timestamp=abs_start_ts,
                step=self.current_iteration,
                event_name=record["stage"],
                event_type="START",
                metadata=f"node_id={record['node_id']}"
            )
            
            # E. write the END event
            self.logger.log_profile_event(
                timestamp=abs_end_ts,
                step=self.current_iteration,
                event_name=record["stage"],
                event_type="END",
                duration_ms=final_duration_ms,
                metadata=f"node_id={record['node_id']}" # more metadata can be added here
            )
            full_record = {
                "stage": record["stage"],
                "rank": self.rank,
                "iteration": self.current_iteration,
                "cpu_duration_ms": record["cpu_duration_ms"],
                "cuda_duration_ms": cuda_elapsed_ms,
                
                # important: keep the hierarchy id supplied by the V2 stack timer
                "node_id": record["node_id"],
                "parent_id": record["parent_id"],
                
                # (temporary field used by step 2)
                "children": []
            }
            self.records.append(full_record)
        
        self._pending_records.clear()

        if not self.records or not self.verbose:
            # nothing recorded, or not in verbose mode: return early
            # (self.records is still populated; only the logging is skipped)
            return

        # --- step 2: rebuild the call tree from self.records (a flat list) ---
        
        # a dict for fast lookup
        nodes_map = {node['node_id']: node for node in self.records}
        
        # (self.root_node (id=0) is the global root; only this iteration's
        #  subtree is built here)
        
        tree_roots = [] # this iteration's top-level calls
        
        for node in self.records:
            parent_id = node['parent_id']
            if parent_id in nodes_map:
                # a child: append it to its parent's 'children' list
                parent_node = nodes_map[parent_id]
                parent_node['children'].append(node)
            else:
                # a top-level node (its parent is not among this iteration's
                # records, most likely the global root id=0)
                tree_roots.append(node)

        # --- step 3: compute self time recursively ---
        
        def calculate_self_time_recursive(node):
            """
            Walk the tree and compute each node's self time.
            Returns: this node's total CUDA time (used by its parent).
            """
            # CUDA time is used as the "total time"
            total_time_ms = node.get('cuda_duration_ms') or 0.0
            
            if not node['children']:
                # a leaf: self time == total time
                node['self_time_ms'] = total_time_ms
                return total_time_ms
                
            # recursively total the children
            children_total_time = 0.0
            for child in node['children']:
                children_total_time += calculate_self_time_recursive(child)
            
            # Self Time = Total Time - Children's Total Time
            node['self_time_ms'] = total_time_ms - children_total_time
            
            # return *total time* to the parent
            return total_time_ms

        # start the computation from every top-level node
        for root_node in tree_roots:
            calculate_self_time_recursive(root_node)

        # --- step 4: log the hierarchy recursively ---
        
        def log_tree_recursive(node, indent_prefix=""):
            """
            Walk the tree and print an indented log.
            """
            # build the log entry
            stage = node['stage']
            cpu_ms = node['cpu_duration_ms']
            cuda_ms = node.get('cuda_duration_ms') or 0.0
            self_ms = node.get('self_time_ms', 0.0) # self_time was just computed
            
            # this is what makes the log parseable
            node_id = node['node_id']
            parent_id = node['parent_id']
            
            # format self time (shown only when there are children and self > 0)
            self_time_str = ""
            if node['children'] and self_ms > 0.001:
                self_time_str = f", Self {self_ms:.3f}ms"

            # the final parseable, hierarchical log message
            log_msg = (
                f"[Iter {self.current_iteration}] {indent_prefix}"
                f"Stage '{stage}' [id={node_id}, p_id={parent_id}]: "
                f"CPU {cpu_ms:.3f}ms, CUDA {cuda_ms:.3f}ms{self_time_str}"
            )
            
            self.logger.info(log_msg)
            
            # recurse into the children
            new_indent = indent_prefix + "  L "
            
            # (sorted by node_id so log order roughly matches call order)
            sorted_children = sorted(node['children'], key=lambda x: x['node_id'])
            
            for child in sorted_children:
                log_tree_recursive(child, indent_prefix=new_indent)

        # make sure self.logger is available
        self._ensure_logger()

        # (sort the root nodes by node_id)
        sorted_roots = sorted(tree_roots, key=lambda x: x['node_id'])
        
        # start logging
        for root_node in sorted_roots:
            log_tree_recursive(root_node, indent_prefix="") # no indent at the top level

    def step(self):
        self.synchronize_and_log()

        
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
        Append this rank's raw timing records to the log file.

        Args:
            sort_records (bool, optional): whether to sort before writing.
                                         Default False, i.e. written in execution order.
                                         True sorts by iteration and stage name.
        """
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{self.tag}_rank{self.rank}.log")

        # sort only when sort_records asks for it
        if sort_records:
            records_to_write = sorted(
                self.records, key=lambda x: (x["iteration"], x["stage"])
            )
            sort_info = "(Sorted)"
        else:
            # by default use the original list, preserving execution order
            records_to_write = self.records
            sort_info = "(Execution Order)"

        with open(log_path, "a") as f:
            f.write(
                f"\n==================== DUMP {sort_info} (Up to Iteration {self.current_iteration}) ====================\n"
            )

        # walk the processed list
            for r in records_to_write:
                # default cuda_duration_ms so a missing key does not raise
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
        Generate the detailed performance report and save it to a file.
        Data from every rank is aggregated and reported per distinct stage name.
        """
        # _gather_records() returns a list holding the records of all ranks
        all_records = self._gather_records()

        if self.rank == 0:
            # 1. filter the qualifying records (unchanged)
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

            # --- core section ---

            # 2. group every rank's data by stage_name
            # all ranks sharing a stage name are aggregated together
            grouped_data = defaultdict(list)
            for r in filtered_records:
            # r["rank"] no longer matters; equal stage names aggregate
                if r["cuda_duration_ms"] is not None:
                    grouped_data[r["stage"]].append(r["cuda_duration_ms"])

            # 3. compute statistics for each aggregated stage
            report_data = {} # no longer grouped by rank
            for stage_name, durations in grouped_data.items():
            # each stage_name is one entry whose durations come from every rank
                report_data[stage_name] = {
                    "count": len(durations),
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                }
            
            # --- end of core section ---

            # 4. build the formatted report string (a single aggregated table)
            report_string = ""
            report_header = (
                f"--- 📊 Aggregated Performance Report (All Ranks) ---\n"
                f"Pattern: '{stage_pattern}'\n"
                f"Filename: {output_filename}\n"
                f"{'-'*80}\n"
            )
            report_string += report_header

            # the per-rank loop is no longer needed
            report_string += f"\n[Aggregated Statistics]\n"
            report_string += f" {'STAGE':<60} {'COUNT':<7} {'MEAN (ms)':<12} {'MEDIAN (ms)':<13} {'STD (ms)':<12}\n"
            report_string += f" {'-'*59} {'-'*6} {'-'*11} {'-'*12} {'-'*11}\n"

            # table body
            for stage_name in sorted(report_data.keys()):
                stats = report_data[stage_name]
                report_string += (
                    f" {stage_name:<60} {stats['count']:<7} "
                    f"{stats['mean']:<12.3f} {stats['median']:<13.3f} {stats['std']:<12.3f}\n"
                )

            # 5. print to the console
            print(report_string)

            # 6. save to file
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
        Extract the suffix parameters bs/f/h/w/sp and the median from generate_report's

        Args:
            report_data (dict): generate_report's return value, keyed report_data[rank][suffix].
            csv_filename (str): output CSV filename (default suffix_median_report.csv)
        """
        import csv

        if self.rank != 0:
            return  # rank 0 only

        if self.log_dir is None:
            self.logger.warning("log_dir not set, cannot save CSV file.")
            return

        csv_path = os.path.join(self.log_dir, csv_filename)

        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["bs", "f", "h", "w", "sp", "median"])  # header

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



class NoOpMyTimer:
    """
    A dummy timer that is interface-compatible with MyTimer.
    Every method is a no-op; it stands in for MyTimer when profiling is disabled.
    *args and **kwargs let it accept any arguments without raising.
    """
    def __init__(self, *args, **kwargs):
        pass

    def set_logger(self, logger_instance: logging.Logger):
        pass

    def set_labeler(self, labeler: Optional[LabelerProtocol]):
        pass

    def start(self, stage_name: str, *args, **kwargs):
        pass

    def stop(self, stage_name: str, *args, **kwargs):
        pass

    def synchronize_and_log(self):
        pass

    def step(self):
        """Alias of synchronize_and_log."""
        pass

    def next_iteration(self):
        pass

    def dump(self, *args, **kwargs):
        pass

    def generate_report(self, *args, **kwargs):
        # the real method returns a dict on rank 0 and None elsewhere;
        # returning None here keeps the behaviour consistent
        return None

    def generate_csv(self, *args, **kwargs):
        pass
    
    # other public methods (register_stage, disable_cuda_time, ...) can be added here
    def register_stage(self, *args, **kwargs):
        pass

    def disable_cuda_time(self, *args, **kwargs):
        pass

    # the context manager must be implemented too
    @contextmanager
    def time_stage(self, stage_name: str, *args, **kwargs):
        try:
            yield
        finally:
            pass # nothing to do

    def register_stage(self, *args, **kwargs):
            """No-op registration: accepts any arguments and does nothing."""
            pass



# the global instance used under SPMD
PROFILING_ENABLED = os.environ.get("ENABLE_TIMER", "0") == "1"
if PROFILING_ENABLED:
    global_timer = MyTimer()
else: 
    global_timer = NoOpMyTimer()


def get_global_timer():
    return global_timer

# (in my_utils.init_utils.py)

import os
import logging
import torch.distributed as dist
from .logger import GlobalLogger, get_global_logger
from ..memory.memory_snapshot import global_snapshotter
def setup_logging_and_timer(args, role_tag: str, use_cuda: bool, use_nvtx: bool, is_distributed: bool):
    """
    Initialise GlobalLogger and MyTimer for this process (worker or driver).
    
    Returns:
        (logging.Logger, MyTimer/NoOpTimer): the configured logger and timer.
    """
    
    # --- 1. configure GlobalLogger ---
    logger_instance = GlobalLogger()
    
    if not logger_instance.is_configured:
        if is_distributed:
            # worker process: take the rank from torch.dist
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        elif os.environ.get('LOCAL_RANK') is not None:
            # torchrun
            rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            print(f"Detected torchrun environment: LOCAL_RANK={rank}, WORLD_SIZE={world_size}")
        else:
            # driver process: always 0/1
            rank = 0
            world_size = 1
            
        if not hasattr(args, 'log_dir') or args.logdir is None:
            base_log_dir = "logs"
        else:
            base_log_dir = args.logdir

        # e.g. "logs/Critic" or "logs/Trainer_Driver"
        log_dir = os.path.join(base_log_dir, str(role_tag))
        
        logger_instance.setup(
            log_dir=log_dir,
            level=logging.INFO, # or args.log_level
            rank=rank,
            world_size=world_size,
            extra_log_label=str(role_tag)
        )
    
    logger = get_global_logger()
    logger.info(f"Logger for {role_tag} (Rank {rank} World_size {world_size}) configured.")
    timer = None
    # --- 2. configure MyTimer ---
    if hasattr(args, 'use_ray') and args.use_ray :
        if os.environ.get("ENABLE_TIMER", "0") == "1":
            logger.info(f"Performance Timer ENABLED for {role_tag}, nvtx={use_nvtx}, use_cuda={use_cuda}.")
            
            timer = MyTimer(
                use_cuda=use_cuda,
                tag=str(role_tag),
                log_dir=log_dir,
                labeler=create_labeler(enabled=use_nvtx),
                profile_memory=False
            )
            
            # inject the logger

            
        else:
            timer = NoOpMyTimer()
        timer.set_logger(logger)
    
    global_timer.set_logger(logger)
    global_timer.use_cuda = use_cuda
    global_timer.tag = str(role_tag)
    # global_timer.log_dir = logger_instance.log_dir # (reuse the logger's log_dir)
    # global_timer.rank = logger_instance.rank
    
    # (hard-coded in V1)
    global_timer.set_labeler(create_labeler(enabled=False))
    global_timer.profile_memory = False
    
    # 3. record which kind of global_timer we ended up with
    # (checking for NoOpTimer tells us the state of ENABLE_TIMER)
    if isinstance(global_timer, NoOpMyTimer):
        logger.info(f"Performance Timer is DISABLED for {role_tag}.")
    else:
        logger.info(f"Performance Timer ENABLED for {role_tag} (Rank {global_timer.rank}).")

    


    global_snapshotter.set_logger(logger=logger)
    
    return logger, timer
    


def print_cuda_memory_gb(step_name=""):
    """
    Print this rank's allocated and cached CUDA memory.
    Units are GB.
    """
    # requires CUDA and an initialised distributed environment
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"✅ [Rank {rank}] [CUDA Memory] {step_name}: "
            f"Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB"
        )
    else:
        # without a distributed environment or CUDA, print the plain message
        print(f"✅ {step_name}: CUDA not available or distributed not initialized.")




import threading
import traceback

class DebuggingEvent(threading.Event):
    """
    An Event subclass for debugging.
    It takes a logger at construction time and, whenever .set() is called,
    records the stack with that logger.
    """
    def __init__(self, *args, logger=None, **kwargs):
        # call the parent constructor
        super().__init__(*args, **kwargs)
        
        # keep the logger; build a default print-based one when none is given
        if logger:
            self.logger = logger
        else:
            # fallback: without a logger, print to the console
            self.logger = logging.getLogger("DebuggingEvent")
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def set(self):
        # capture the stack with StringIO instead of printing it directly
        import io
        s = io.StringIO()
        traceback.print_stack(file=s)
        stack_info = s.getvalue()
        s.close()

        # log through the stored logger
        self.logger.info(
            f"\n{'='*30} [EVENT SET TRACE] {'='*30}\n"
            f">>> Event object {id(self)} is being set() by:\n"
            f"{stack_info}"
            f"{'='*80}"
        )
        
        # call the parent's original set method
        super().set()



def record_oom_threshold(failing_bs: int, failing_frame: int, step: int = 4):
    """
    On OOM, record the safe frame-count ceiling for the corresponding batch size.

    The function is idempotent:
    1. the file is created when it does not exist;
    2. an existing record is updated only when the new ceiling is stricter (smaller).

    Args:
        failing_bs (int): the batch size that caused the OOM.
        failing_frame (int): the frame count that caused the OOM.
        step (int): the frame-count increment, used to derive the previous safe point.
    """
    import json
    # from megatron.core import mpu
    sp = os.environ.get('sp')
    model = os.environ.get('model_type') 
    resolution = os.environ.get('resolution')

    threshold_file = f"oom_thresholds_{model}_{resolution}_sp{sp}.json"
    print(f"--- OOM Detected! Recording threshold for bs={failing_bs} ---")
    
    # derive the last known "safe" point from the failing frame count
    # e.g. frame=73 failed, so the ceiling becomes 69 (73-4)
    new_max_frame = failing_frame - step
    
    # read the existing threshold file; start a new dict if it is missing or empty
    thresholds = {}
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content: # make sure the file is not empty
                    thresholds = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read or parse '{threshold_file}'. Starting with empty thresholds.")
            thresholds = {} # reset on error
    
    # JSON keys must be strings
    failing_bs_str = str(failing_bs)
    
    # current ceiling for this bs, or infinity when unrecorded
    current_max = thresholds.get(failing_bs_str, float('inf'))
    
    # update only when the new ceiling is stricter (smaller) than the old one
    if new_max_frame < current_max:
        print(f"Updating bs={failing_bs} max frame from {current_max} to {new_max_frame}")
        thresholds[failing_bs_str] = new_max_frame
        
        # write the updated thresholds back, pretty-printed
        with open(threshold_file, 'w', encoding='utf-8') as f:
            json.dump(thresholds, f, indent=4)
            print(f"Successfully saved new thresholds to '{threshold_file}'.")
    else:
        print(f"New max frame ({new_max_frame}) is not stricter than existing ({current_max}). No update needed.")



def print_tensor_info(tensor: torch.Tensor, name: str = ""):
        """
        Print a PyTorch tensor's rank, shape, device and dtype on one line.

        Args:
            tensor (torch.Tensor): the tensor to inspect.
            name (str, optional): a name to distinguish it in the output. Defaults to "".
        """
        if not isinstance(tensor, torch.Tensor):
            print(f"the input '{name}' is not a valid PyTorch tensor.")
            return

        # format everything onto one line with an f-string
        # prefix with "name: " when a name was given
        prefix = f"{name}: " if name else ""
        print(
            f"{prefix}"
            f"Rank {dist.get_rank()}, "
            f"Shape={tensor.shape}, "
            f"Device='{tensor.device}', "
            f"Dtype={tensor.dtype}"
        )


try:
    from tensordict import TensorDict  # optional dependency
except Exception:
    TensorDict = Any


IS_ENABLED = os.environ.get("DEBUG_DATA_CONSISTENCY", "0") == "1"
CSUM_PREFIX = "_csum_"

def _get_checksum_for_slice(tensor_slice: torch.Tensor) -> float:
    """
    [V6] Compute the checksum of a *single* batch item (slice).
    """
    if not IS_ENABLED: return 0.0
    try:
        # (.float() guards against BFloat16 and similar precision issues)
        return torch.sum(tensor_slice.cpu().float()).item()
    except Exception:
        return -1.0

class ChecksumUtils:
    
    @staticmethod
    def sign(payload: dict):
        """
        [Called on the *sending* side - V6, per slice]
        
        Core behaviour:
        1. iterate every tensor in the payload;
        2. iterate *every item* of that tensor (0 to batch_size-1);
        3. compute a checksum for *each item*;
        4. wrap the [csum0, csum1, ...] list into a tensor of shape [BS].
        """
        if not IS_ENABLED:
            return

        checksums_to_add = {}
        for key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                continue

            csum_key = f"{CSUM_PREFIX}{key}"
            
            # V6: iterate the batch
            batch_size = value.shape[0]
            csum_list = []
            for i in range(batch_size):
                # (checksum of the i-th slice)
                csum_list.append(_get_checksum_for_slice(value[i]))
            
            # (build the [BS]-shaped tensor)
            checksums_to_add[csum_key] = torch.tensor(
                csum_list, 
                dtype=torch.float32, 
                device=value.device # (.cpu() also works, but matching .device is better)
            )
        
        # note: this *modifies* the payload
        payload.update(checksums_to_add)

    @staticmethod
    def verify(batch: TensorDict, logger: logging.Logger):
        """
        [Called on the *receiving* side - V6, per slice]
        
        Compares checksums inside the *already sliced* 'TensorDict' (.batch).
        """
        if not IS_ENABLED:
            return
            
        if not logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] ChecksumUtils.verify: No logger provided, skipping.")
            return

        csum_keys = [k for k in batch.keys() if k.startswith(CSUM_PREFIX)]
        if not csum_keys:
            logger.info("[Checksum] No checksum tensors found in TensorDict.")
            return

        for csum_key in csum_keys:
            original_key = csum_key[len(CSUM_PREFIX):]
            
            # 2. take the *checksum tensor* (e.g. shape [2])
            expected_csums_tensor = batch[csum_key]
            
            # 3. take the *data tensor* (e.g. shape [2, F, C, H, W])
            received_tensors = batch.get(original_key)
            
            if received_tensors is None:
                logger.warning(f"[Checksum] Found '{csum_key}' but "
                                 f"missing '{original_key}' in TensorDict!")
                continue
            
            batch_size = received_tensors.shape[0]
            if expected_csums_tensor.shape[0] != batch_size:
                 logger.error(f"[Checksum] FAILED: Mismatched batch size for '{original_key}'. "
                                f"Data has {batch_size} but csum has {expected_csums_tensor.shape[0]}.")
                 continue

            # 4. V6 core: iterate the *sliced* batch
            for i in range(batch_size):
                
                tensor_slice = received_tensors[i] # (the i-th tensor, BS=1)
                expected_csum = expected_csums_tensor[i].item() # (the i-th checksum, float)
                
                try:
                    # recompute the checksum of *this slice*
                    new_csum = _get_checksum_for_slice(tensor_slice)
                    
                    if not torch.allclose(torch.tensor(expected_csum), torch.tensor(new_csum)):
                         logger.error(
                            f"[!!] CHECKSUM MISMATCH (Key: {original_key}, Batch Index: {i}) [!!]\n"
                            f"  Sender (Generator)   Calculated: {expected_csum}\n"
                            f"  Receiver (Critic)  Re-calculated: {new_csum}"
                        )
                    else:
                        logger.info(
                            f"[Checksum OK] Key: {original_key} (Index: {i}, Sum: {new_csum})"
                        )
                except Exception as e:
                    logger.error(f"[Checksum] FAILED to verify '{original_key}': {e}")
