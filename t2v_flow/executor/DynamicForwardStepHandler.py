import threading
import logging
from ..planner import Planner
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
from megatron.training.utils import get_batch_on_this_tp_cp_rank_vast
from megatron.core.models.hunyuan.pipeline import HunyuanPipeline
from megatron.core.models.hunyuan.WrapedModel import TrainingWrapperModel
import os
import re
from megatron.core import mpu
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Tuple
from ..planner import Task
import time
import yaml
from ..communication import DATA_CACHE
import queue
import threading
import pickle
import traceback
import socket
from collections import Counter, defaultdict
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


from typing import Optional

BROKER_RANK = 0


def _is_distributed_fatal_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return (
        "out of memory" in error_str
        or "timeout" in error_str
        or "nccl" in error_str
        or "connection" in error_str
        or "unhandled system error" in error_str
        or "closed" in error_str
    )


def _record_resume_step(resume_iter: int) -> None:
    try:
        with open("oom_resume_step.txt", "w") as f:
            f.write(str(resume_iter))
    except Exception:
        pass


def _abort_distributed() -> None:
    if not dist.is_initialized():
        return
    if hasattr(dist, "abort"):
        try:
            dist.abort()
            return
        except Exception:
            pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass

def _abort_distributed_hard(exit_code: int = 100):
                                                  
    try:
        if dist.is_initialized() and hasattr(dist, "abort"):
            dist.abort()
    except Exception:
        pass

    try:
        import signal
        os.kill(os.getpid(), signal.SIGKILL)        
    except Exception:
        os._exit(exit_code)

class TimingLogFilter(logging.Filter):
    def filter(self, record):
        return getattr(record, 'log_type', None) == 'timing'

class NonTimingLogFilter(logging.Filter):
    def filter(self, record):
        return getattr(record, 'log_type', None) != 'timing'

class DynamicForwardStepHandler(metaclass=SingletonMeta):
    """Singleton runtime executor: holds a Planner instance and schedules
    forward_step according to the per-iteration plan."""

    def __init__(self, log_dir: str = "handler_logs"):
        """Constructor body runs only on first instantiation (singleton)."""

        self.logger = None               
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.log_dir = log_dir

        self.planner: Planner = Planner()

        self.my_tasks_for_this_iteration: List[Task] = []

        self.stage: int = 0           
        self.iter : int = 0

        self.active_task: Optional[Task] = None                  

        self.group_cache_lookahead = max(
            0, int(os.environ.get("CUSTOM_GROUP_CACHE_LOOKAHEAD", "3"))
        )
        self.group_cache_gc_interval = max(
            1, int(os.environ.get("CUSTOM_GROUP_GC_INTERVAL", "3"))
        )
        self.group_cache_max_groups = max(
            1, int(os.environ.get("CUSTOM_GROUP_CACHE_MAX_GROUPS", "32"))
        )
        self.group_cache_max_local_memberships = max(
            1, int(os.environ.get("CUSTOM_GROUP_CACHE_MAX_LOCAL_PER_RANK", "8"))
        )
        self.model_type = os.environ.get("MODEL_TYPE", "").lower()
        self.enable_group_cache = self.model_type in {"wan", "cogvideox"}
        self.enable_multi_broker = os.environ.get("ENABLE_MULTI_BROKER", "0") == "1"
        self.enable_remote_get_prefetch = (
            os.environ.get("ENABLE_REMOTE_GET_PREFETCH", "0") == "1"
        )
        self.remote_get_prefetch_lookahead = max(
            0, int(os.environ.get("REMOTE_GET_PREFETCH_LOOKAHEAD", "2"))
        )
        self.remote_get_prefetch_max_inflight = max(
            1, int(os.environ.get("REMOTE_GET_PREFETCH_MAX_INFLIGHT", "1"))
        )
        self.remote_get_prefetch_wait_timeout = max(
            1.0, float(os.environ.get("REMOTE_GET_PREFETCH_WAIT_TIMEOUT", "120.0"))
        )
        self.broker_get_wait_timeout = max(
            1.0,
            float(
                os.environ.get(
                    "BROKER_GET_WAIT_TIMEOUT",
                    str(self.remote_get_prefetch_wait_timeout),
                )
            ),
        )
        self.remote_get_prefetch_sync_fallback_timeout = max(
            0.0,
            float(
                os.environ.get(
                    "REMOTE_GET_PREFETCH_SYNC_FALLBACK_TIMEOUT",
                    "0",
                )
            ),
        )
        self._remote_get_prefetch_lock = threading.Lock()
        self._remote_get_prefetch_states: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._remote_get_prefetch_queue: queue.Queue = queue.Queue()
        self._remote_get_prefetch_thread: Optional[threading.Thread] = None
        self.rank_to_node_idx: Dict[int, int] = {}
        self.node_to_broker_rank: Dict[int, int] = {}
        self.broker_ranks = {BROKER_RANK}
        self.source_task_to_broker_rank: Dict[str, int] = {}
        self.broker_publishers_by_rank: Dict[int, List[int]] = {}
        self.broker_getters_by_rank: Dict[int, List[int]] = {}

        self._dist_setup_done = False
        self._initialized = True
                                                
            
                                                   


    def _initialize_broker_topology(self) -> None:
        local_host = socket.gethostname()
        hosts = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            hosts,
            local_host,
            group=self.cpu_service_group,
        )

        host_to_node_idx = {}
        node_to_ranks = defaultdict(list)
        self.rank_to_node_idx = {}
        for rank, host in enumerate(hosts):
            if host not in host_to_node_idx:
                host_to_node_idx[host] = len(host_to_node_idx)
            node_idx = host_to_node_idx[host]
            self.rank_to_node_idx[rank] = node_idx
            node_to_ranks[node_idx].append(rank)

        self.node_to_broker_rank = {
            node_idx: min(ranks)
            for node_idx, ranks in node_to_ranks.items()
            if ranks
        }
        if self.enable_multi_broker:
            self.broker_ranks = set(self.node_to_broker_rank.values())
        else:
            self.broker_ranks = {BROKER_RANK}

    def _get_broker_rank_for_source_task(self, source_task_name: str) -> int:
        return self.source_task_to_broker_rank.get(source_task_name, BROKER_RANK)

    def _collect_source_task_comm_graph(self):
        source_to_publisher = {}
        source_to_getters = defaultdict(list)

        if not self.planner._tasks:
            return source_to_publisher, source_to_getters

        for task in self.planner._tasks.values():
            if task.task_type == "VAE" and not task.has_overlapping_consumer:
                source_to_publisher[task.name] = min(task.gpus)

        for task in self.planner._tasks.values():
            if task.task_type not in ["DiT", "DIT"] or not task.data_source_task:
                continue
            source_task_name = task.data_source_task
            if source_task_name not in source_to_publisher:
                continue

            producer_task = self.planner.get_task_by_name(source_task_name)
            if producer_task.task_type != "VAE":
                continue

            producer_gpus = set(producer_task.gpus)
            consumer_gpus = set(task.gpus)
            if producer_gpus.isdisjoint(consumer_gpus):
                source_to_getters[source_task_name].append(min(task.gpus))

        return source_to_publisher, source_to_getters

    def _select_broker_for_source_task(
        self,
        source_task_name: str,
        producer_leader: int,
        getter_leaders: List[int],
        broker_load_counter: Dict[int, int],
    ) -> int:
        if not self.enable_multi_broker or not self.node_to_broker_rank:
            return BROKER_RANK

        producer_node = self.rank_to_node_idx.get(producer_leader)
        if producer_node is None:
            return BROKER_RANK

        if not getter_leaders:
            return self.node_to_broker_rank.get(producer_node, BROKER_RANK)

        getter_nodes = [
            self.rank_to_node_idx.get(rank)
            for rank in getter_leaders
            if rank in self.rank_to_node_idx
        ]
        getter_nodes = [node for node in getter_nodes if node is not None]
        if not getter_nodes:
            return self.node_to_broker_rank.get(producer_node, BROKER_RANK)

        candidate_nodes = {producer_node}
        candidate_nodes.update(getter_nodes)

        getter_node_set = set(getter_nodes)

        best_broker = None
        best_score = None
        for node_idx in sorted(candidate_nodes):
            broker_rank = self.node_to_broker_rank.get(node_idx)
            if broker_rank is None:
                continue

            cross_node_publish_cost = 0 if node_idx == producer_node else 1
            cross_node_get_cost = sum(0 if node_idx == getter_node else 1 for getter_node in getter_nodes)

                                                                                   
                                                                                   
                                                                               
                                                                              
                                                                                
                                                                                
            is_producer_side = 1 if node_idx == producer_node and node_idx not in getter_node_set else 0

            load_cost = broker_load_counter.get(broker_rank, 0)
            score = (cross_node_publish_cost + cross_node_get_cost, is_producer_side, load_cost, broker_rank)
            if best_score is None or score < best_score:
                best_score = score
                best_broker = broker_rank

        if best_broker is None:
            return BROKER_RANK

        broker_load_counter[best_broker] = broker_load_counter.get(best_broker, 0) + 1
        return best_broker

    def _compute_broker_routing_payload(self):
        source_to_publisher, source_to_getters = self._collect_source_task_comm_graph()
        source_to_broker = {}
        publishers_by_broker = defaultdict(list)
        getters_by_broker = defaultdict(list)
        broker_load_counter = {}

        for source_task_name in sorted(source_to_publisher.keys()):
            producer_leader = source_to_publisher[source_task_name]
            getter_leaders = source_to_getters.get(source_task_name, [])
            broker_rank = self._select_broker_for_source_task(
                source_task_name=source_task_name,
                producer_leader=producer_leader,
                getter_leaders=getter_leaders,
                broker_load_counter=broker_load_counter,
            )
            source_to_broker[source_task_name] = broker_rank
            publishers_by_broker[broker_rank].append(producer_leader)
            getters_by_broker[broker_rank].extend(getter_leaders)

        for broker_rank in self.broker_ranks:
            publishers_by_broker.setdefault(broker_rank, [])
            getters_by_broker.setdefault(broker_rank, [])

        return (
            source_to_broker,
            {k: list(v) for k, v in publishers_by_broker.items()},
            {k: list(v) for k, v in getters_by_broker.items()},
        )

    def _update_broker_routing_for_iteration(self):
        payload = None
        if self.rank == 0:
            payload = self._compute_broker_routing_payload()

        payload_obj = [payload]
        dist.broadcast_object_list(payload_obj, src=0)
        payload = payload_obj[0]

        if payload is None:
            self.source_task_to_broker_rank = {}
            self.broker_publishers_by_rank = {broker_rank: [] for broker_rank in self.broker_ranks}
            self.broker_getters_by_rank = {broker_rank: [] for broker_rank in self.broker_ranks}
            return

        source_to_broker, publishers_by_broker, getters_by_broker = payload
        self.source_task_to_broker_rank = dict(source_to_broker)
        self.broker_publishers_by_rank = {
            broker_rank: list(publishers_by_broker.get(broker_rank, []))
            for broker_rank in self.broker_ranks
        }
        self.broker_getters_by_rank = {
            broker_rank: list(getters_by_broker.get(broker_rank, []))
            for broker_rank in self.broker_ranks
        }


    def prepare_after_dist_initialized(self,  log_dir: str = "handler_logs"):

        if self._dist_setup_done:
            return

        if not dist.is_initialized():
            print("Warning: Distributed environment not ready. Cannot run prepare_after_dist_initialized.")
            return

        print(f"DynamicForwardStepHandler prepare_after_dist_initialized called (Phase 2: Configuration)")


        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.log_dir = log_dir 
        self._setup_logger()
        self.logger.info(
            "--- DynamicForwardStepHandler Singleton is being initialized (once) ---"
        )
        self.logger.info(
            f"[Rank {self.rank}] Multi broker: enabled={self.enable_multi_broker}"
        )
        self.logger.info(
            f"[Rank {self.rank}] Remote GET prefetch: enabled={self.enable_remote_get_prefetch}, "
            f"lookahead={self.remote_get_prefetch_lookahead}, "
            f"max_inflight={self.remote_get_prefetch_max_inflight}, "
            f"wait_timeout={self.remote_get_prefetch_wait_timeout}s, "
            f"broker_get_wait_timeout={self.broker_get_wait_timeout}s, "
            f"sync_fallback_timeout={self.remote_get_prefetch_sync_fallback_timeout}s"
        )


        self.cpu_service_group = dist.new_group(backend='gloo')
                        
        dist.barrier(group = self.cpu_service_group)
        self.logger.info(f"Rank {self.rank}: CPU service communication group (gloo) created.")
        self._initialize_broker_topology()
        self.logger.info(
            f"Rank {self.rank}: broker_ranks={sorted(self.broker_ranks)}, "
            f"node_to_broker={self.node_to_broker_rank}"
        )


                                                                    
                                                                    
              
                                                                                        

                                 
                                                                                   
                                                                                    
                                                                                          
            
                                
                                                                              
                                                                                     
                                               
                                                                                                                            
                       
                                                                                                       

                                                                                      
                           
                                                                                                                      

            
                                    
                                  
        self.PUBLISH_CTRL_TAG = 9901
        self.GET_CTRL_TAG = 9902
        self.DATA_XFER_TAG = 9903

        self.publish_stream = torch.cuda.Stream()
        self.publish_queue = queue.Queue()
        self.publisher_thread = threading.Thread(
            target=self._publisher_worker_loop,
            daemon=True                        
        )
        self.publisher_thread.start()
        
        self.logger.info(f"Rank {self.rank}: Publisher thread started.")

        if self.enable_remote_get_prefetch:
            self._remote_get_prefetch_thread = threading.Thread(
                target=self._remote_get_prefetch_worker_loop,
                daemon=True,
            )
            self._remote_get_prefetch_thread.start()
            self.logger.info(f"Rank {self.rank}: Remote GET prefetch worker thread started.")


 
        if self.rank in self.broker_ranks:
            self.central_data_store: Dict[str, Tuple[list, torch.Tensor]] = {}
            self.data_events: Dict[str, threading.Event] = {}
            self.broker_lock = threading.Lock()
            self.broker_shutdown_event = threading.Event()
                                                                                                 

            self.broker_publish_task_queue = queue.Queue()
            self.broker_get_task_queue = queue.Queue()
            self.broker_publish_thread = threading.Thread(target=self._publish_listener_loop, daemon=True)
            self.broker_get_thread = threading.Thread(target=self._get_listener_loop, daemon=True)
            
                                        
                                                                                   
            self.broker_publish_thread.start()
            self.broker_get_thread.start()
            self.logger.info(f"Rank {self.rank}: Broker service with dedicated PUBLISH/GET listener threads started.")

        self._dist_setup_done = True

    def _send_control_msg(self, dst: int, msg: dict, tag: int):
        """Serialize and send a tagged control-message dict over the GLOO group."""
        try:
            self.logger.info(f"Rank {self.rank} -> Rank {dst} (Tag:{tag}): Sending control msg: {msg}")
            msg_bytes = pickle.dumps(msg)
            size_tensor = torch.tensor([len(msg_bytes)], dtype=torch.long, device="cpu")
            
            dist.send(tensor=size_tensor, dst=dst, group=self.cpu_service_group, tag=tag)
            self.logger.info(f"Rank {self.rank} -> Rank {dst}: Sent control msg size ({size_tensor.item()} bytes).")

            msg_tensor = torch.frombuffer(bytearray(msg_bytes), dtype=torch.uint8).to("cpu")
            dist.send(tensor=msg_tensor, dst=dst, group=self.cpu_service_group, tag=tag)
            self.logger.info(f"Rank {self.rank} -> Rank {dst}: Control msg sent successfully.")
        except Exception:
            self.logger.error(f"Rank {self.rank}: Failed to send control msg to Rank {dst}.", exc_info=True)
            raise


    def _recv_control_msg_from_size(self, src: int, size_tensor: torch.Tensor, tag: int) -> dict:
        """Receive a control message of known size over the GLOO group;
        used by the broker polling loop."""
        try:
            msg_size = size_tensor.item()
            self.logger.info(f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Receiving control msg payload ({msg_size} bytes)...")
            
            msg_tensor = torch.empty(msg_size, dtype=torch.uint8, device="cpu")
            dist.recv(tensor=msg_tensor, src=src, group=self.cpu_service_group, tag=tag)
            
            control_msg = pickle.loads(msg_tensor.numpy().tobytes())
            self.logger.info(f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Control msg payload received successfully: {control_msg}")
            return control_msg
        except Exception:
            self.logger.error(f"Rank {self.rank}: Failed to receive control msg payload from Rank {src}.", exc_info=True)
            raise

    def _send_prepacked_data(self, dst: int, metadata: list, packed_cpu_buffer: torch.Tensor, tag: int):
            """Send prepacked metadata + buffer to the target rank (with logging)."""
            try:
                metadata_bytes = pickle.dumps(metadata)
                metadata_size = torch.tensor([len(metadata_bytes)], dtype=torch.long, device="cpu")
                buffer_size = packed_cpu_buffer.numel()

                self.logger.info(
                    f"Rank {self.rank} -> Rank {dst}: Sending pre-packed data. "
                    f"Metadata: {metadata_size.item()} bytes, Buffer: {buffer_size} bytes."
                )
                
                dist.send(tensor=metadata_size, dst=dst, group=self.cpu_service_group, tag=tag)
                self.logger.debug(f"Rank {self.rank} -> Rank {dst} (Tag:{tag}): Sent metadata size.")

                metadata_tensor = torch.frombuffer(bytearray(metadata_bytes), dtype=torch.uint8).to("cpu")
                dist.send(tensor=metadata_tensor, dst=dst, group=self.cpu_service_group, tag=tag)
                self.logger.debug(f"Rank {self.rank} -> Rank {dst} (Tag:{tag}): Sent metadata payload.")

                dist.send(tensor=packed_cpu_buffer, dst=dst, group=self.cpu_service_group, tag=tag)
                self.logger.info(f"Rank {self.rank} -> Rank {dst} (Tag:{tag}): Pre-packed data sent successfully.")
            except Exception:
                self.logger.error(f"Rank {self.rank} (Tag:{tag}): Failed to send pre-packed data to Rank {dst}.", exc_info=True)
                raise

    def _send_prepacked_data_error(self, dst: int, error: str, tag: int) -> None:
            """Send an error frame on the DATA channel; convention: size < 0
            marks an error and |size| is the error-string byte length."""
            try:
                error_msg = error if error else "unknown broker error"
                error_bytes = error_msg.encode("utf-8", errors="replace")
                if len(error_bytes) == 0:
                    error_bytes = b"unknown broker error"

                error_size = torch.tensor([-len(error_bytes)], dtype=torch.long, device="cpu")
                self.logger.error(
                    f"Rank {self.rank} -> Rank {dst} (Tag:{tag}): Sending error frame ({-error_size.item()} bytes): {error_msg}"
                )
                dist.send(tensor=error_size, dst=dst, group=self.cpu_service_group, tag=tag)

                error_tensor = torch.frombuffer(bytearray(error_bytes), dtype=torch.uint8).to("cpu")
                dist.send(tensor=error_tensor, dst=dst, group=self.cpu_service_group, tag=tag)
            except Exception:
                self.logger.error(
                    f"Rank {self.rank} (Tag:{tag}): Failed to send error frame to Rank {dst}.",
                    exc_info=True,
                )
                raise


    def _recv_prepacked_data(self, src: int, tag: int) -> Tuple[list, torch.Tensor]:
            """Receive prepacked metadata + buffer from the source rank (with logging)."""
            try:
                self.logger.info(f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Waiting to receive pre-packed data...")
                
                size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
                dist.recv(tensor=size_tensor, src=src, group=self.cpu_service_group, tag=tag)
                metadata_size = size_tensor.item()

                if metadata_size < 0:
                    error_size = -metadata_size
                    error_tensor = torch.empty(error_size, dtype=torch.uint8, device="cpu")
                    dist.recv(tensor=error_tensor, src=src, group=self.cpu_service_group, tag=tag)
                    remote_error = error_tensor.numpy().tobytes().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Remote broker returned error: {remote_error}"
                    )

                if metadata_size <= 0:
                    raise RuntimeError(
                        f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Invalid metadata size {metadata_size}."
                    )

                self.logger.info(f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Received metadata size ({metadata_size} bytes). Receiving metadata payload...")

                metadata_tensor = torch.empty(metadata_size, dtype=torch.uint8, device="cpu")
                dist.recv(tensor=metadata_tensor, src=src, group=self.cpu_service_group, tag=tag)
                metadata = pickle.loads(metadata_tensor.numpy().tobytes())
                self.logger.info(f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Received metadata payload. Receiving main data buffer...")

                from t2v_flow.executor import TensorUtils
                packed_cpu_buffer = TensorUtils._create_empty_buffer_from_metadata(metadata, "cpu")
                dist.recv(tensor=packed_cpu_buffer, src=src, group=self.cpu_service_group, tag=tag)
                
                self.logger.info(
                    f"Rank {self.rank} <- Rank {src} (Tag:{tag}): Pre-packed data (Buffer: {packed_cpu_buffer.numel()} bytes) received successfully."
                )
                return metadata, packed_cpu_buffer
            except Exception:
                self.logger.error(f"Rank {self.rank} (Tag:{tag}): Failed to receive pre-packed data from Rank {src}.", exc_info=True)
                raise
    


    def _publisher_worker_loop(self):
        """Publisher thread: pop tasks off the queue, pack, offload to CPU,
        and publish to the broker."""
        while True:
            item = self.publish_queue.get()
            if item is None:
                break

            try:
                task_name, original_gpu_package = item
                self.logger.info(f"Publisher on Rank {self.rank}: Got task '{task_name}' to process and publish.")

                from t2v_flow.executor import TensorUtils
                with torch.cuda.stream(self.publish_stream):
                    self.publish_stream.wait_stream(torch.cuda.default_stream())
                    self.logger.info(f"Publisher on Rank {self.rank}: publish_stream wait finished!")
                    metadata = TensorUtils._extract_metadata(original_gpu_package)
                    packed_gpu_buffer = TensorUtils._pack_tensors_to_buffer(
                        original_gpu_package, metadata, f"cuda:{self.rank % torch.cuda.device_count()}"
                    )
                    self.logger.info(f"Publisher on Rank {self.rank}: Packed '{task_name}' into a contiguous buffer on GPU.")
                    packed_cpu_buffer = packed_gpu_buffer.to('cpu', non_blocking=True)
                                                                            
                self.publish_stream.synchronize()
                self.logger.info(f"Publisher on Rank {self.rank}: Offloaded packed buffer for '{task_name}' to CPU.")       
                target_broker_rank = self._get_broker_rank_for_source_task(task_name)
                if self.rank == target_broker_rank:
                    self.logger.info(
                        f"Publisher on Rank {self.rank} (Broker): Publishing '{task_name}' to local store."
                    )
                    with self.broker_lock:
                        self.central_data_store[task_name] = (metadata, packed_cpu_buffer)
                        from my_utils import DebuggingEvent
                        event = self.data_events.setdefault(task_name, DebuggingEvent(logger=self.logger))
                        event.set()               
                        self.logger.info(f"Publisher on Rank {self.rank} (Broker): Local publish for '{task_name}'  and set event !")
                else:
                    self.logger.info(
                        f"Publisher on Rank {self.rank}: Publishing '{task_name}' to remote Broker Rank {target_broker_rank}."
                    )
                    control_msg = {"action": "PUBLISH", "task_name": task_name, "sender_rank": self.rank}
                    self._send_control_msg(target_broker_rank, control_msg, tag=self.PUBLISH_CTRL_TAG)
                    self._send_prepacked_data(target_broker_rank, metadata, packed_cpu_buffer, self.DATA_XFER_TAG)
                    self.logger.info(f"Publisher on Rank {self.rank}: Successfully published data for '{task_name}'.")
                

            except Exception as e:
                traceback_str = traceback.format_exc()
                
                self.logger.error(
                    f"Publisher on Rank {self.rank}: An exception occurred while processing item: {item}\n"
                    f"--- FULL TRACEBACK ---\n"
                    f"{traceback_str}"
                    f"--- END TRACEBACK ---"
                )
            finally:
                self.publish_queue.task_done()

    def _handle_publish_request(self, src_rank: int, task_name: str):
        """Broker worker thread: handle a single PUBLISH request."""
        try:
            self.logger.info(f"Broker Handler Thread: Handling PUBLISH for '{task_name}' from Rank {src_rank}.")
            metadata, buffer = self._recv_prepacked_data(src_rank, tag = self.DATA_XFER_TAG)
            self.logger.info(f"Broker Handler Thread: Received data payload for '{task_name}'.")
            
            with self.broker_lock:
                self.central_data_store[task_name] = (metadata, buffer)
                from my_utils import DebuggingEvent           
                event = self.data_events.setdefault(task_name, DebuggingEvent(logger=self.logger))
                event.set()
        except Exception:
            self.logger.error(f"Broker Handler Thread: Exception during PUBLISH from {src_rank} for '{task_name}'", exc_info=True)

    def _handle_get_request(self, src_rank: int, task_name: str):
        """Broker worker thread: handle a single GET request."""
        error_frame_sent = False
        data_send_started = False
        try:
            self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Started for task '{task_name}'.")
            
            event = None
            metadata = None
            buffer = None
            with self.broker_lock:
                if task_name in self.central_data_store:
                    self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Data for '{task_name}' found in store immediately.")
                    metadata, buffer = self.central_data_store[task_name]
                else:
                    self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Data for '{task_name}' not yet available. Preparing to wait.")
                    from my_utils import DebuggingEvent
                    event = self.data_events.setdefault(task_name, DebuggingEvent(logger=self.logger))

            if metadata is not None and buffer is not None:
                data_send_started = True
                self._send_prepacked_data(src_rank, metadata, buffer, tag=self.DATA_XFER_TAG)
                self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Immediately fulfilled GET request for '{task_name}'.")
                return            

            is_ready = event.wait(timeout=self.broker_get_wait_timeout)
            
            if not is_ready:
                err = (
                    f"Broker Handler Thread (for Rank {src_rank}): Timeout after "
                    f"{self.broker_get_wait_timeout:.1f}s waiting for data '{task_name}'. "
                    "Aborting request."
                )
                self.logger.error(err)
                self._send_prepacked_data_error(src_rank, err, tag=self.DATA_XFER_TAG)
                error_frame_sent = True
                return

            self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Event for '{task_name}' is set. Retrieving data from store.")
            with self.broker_lock:
                if task_name not in self.central_data_store:
                    err = (
                        f"Broker Handler Thread (for Rank {src_rank}): Event was set but "
                        f"data '{task_name}' is missing from store."
                    )
                    self.logger.error(err)
                    self._send_prepacked_data_error(src_rank, err, tag=self.DATA_XFER_TAG)
                    error_frame_sent = True
                    return

                metadata, buffer = self.central_data_store[task_name]

                del self.central_data_store[task_name]
                if task_name in self.data_events:
                    self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): successfully delete {task_name} events!")
                    del self.data_events[task_name]
            
            self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Sending data for '{task_name}'...")
            data_send_started = True
            self._send_prepacked_data(src_rank, metadata, buffer, tag=self.DATA_XFER_TAG)
            self.logger.info(f"Broker Handler Thread (for Rank {src_rank}): Successfully fulfilled GET request for '{task_name}'.")

        except Exception as e:
            self.logger.error(f"Broker Handler Thread: Exception during handling GET from {src_rank} for '{task_name}'", exc_info=True)
            if not error_frame_sent and not data_send_started:
                try:
                    self._send_prepacked_data_error(
                        src_rank,
                        f"{type(e).__name__}: {e}",
                        tag=self.DATA_XFER_TAG,
                    )
                except Exception:
                    self.logger.error(
                        f"Broker Handler Thread: Failed to send GET error frame to Rank {src_rank} for '{task_name}'.",
                        exc_info=True,
                    )


    def _publish_listener_loop(self):
        """Broker per-iteration listener: PUBLISH requests only."""
        while True:
            publishers_list = self.broker_publish_task_queue.get()
            if publishers_list is None:          
                self.logger.info("PUBLISH listener: Received stop signal. Exiting.")
                break

            comm_counts = Counter(publishers_list)
            remote_comm_counts = {r: c for r, c in comm_counts.items() if r != self.rank}

            if not remote_comm_counts:
                self.logger.info(f"PUBLISH listener: No remote PUBLISH requests for this iteration.")
                self.broker_publish_task_queue.task_done()
                continue
            
            num_expected_requests = sum(remote_comm_counts.values())
            self.logger.info(f"PUBLISH listener: New iteration. Expecting {num_expected_requests} PUBLISH requests from {remote_comm_counts}")

            for i in range(num_expected_requests):
                try:
                    size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
                    sender_rank = dist.recv(tensor=size_tensor, src=None, group=self.cpu_service_group, tag=self.PUBLISH_CTRL_TAG)
                    
                    if sender_rank not in remote_comm_counts or remote_comm_counts[sender_rank] == 0:
                        drained_msg = self._recv_control_msg_from_size(sender_rank, size_tensor, tag=self.PUBLISH_CTRL_TAG)
                        self.logger.warning(
                            f"PUBLISH listener: Received an unexpected PUBLISH from Rank {sender_rank}. "
                            f"Drained payload and ignored message: {drained_msg}"
                        )
                        try:
                            _metadata, _buffer = self._recv_prepacked_data(
                                sender_rank, tag=self.DATA_XFER_TAG
                            )
                            del _metadata, _buffer
                        except Exception:
                            self.logger.error(
                                f"PUBLISH listener: Failed to drain unexpected publish payload from Rank {sender_rank}.",
                                exc_info=True,
                            )
                        continue

                    self.logger.info(f"PUBLISH listener: Received PUBLISH request #{i+1}/{num_expected_requests} from Rank {sender_rank}.")
                    
                    msg = self._recv_control_msg_from_size(sender_rank, size_tensor, tag=self.PUBLISH_CTRL_TAG)
                    task_name = msg.get("task_name")
                    
                    self._handle_publish_request(sender_rank, task_name)
                    
                    remote_comm_counts[sender_rank] -= 1

                except Exception:
                    self.logger.error("Exception in PUBLISH listener's processing loop.", exc_info=True)
            
            self.logger.info(f"PUBLISH listener: Finished all {num_expected_requests} expected requests for this iteration.")
            self.broker_publish_task_queue.task_done()


    def _get_listener_loop(self):
        """Broker per-iteration listener: GET requests only."""
        while True:
            getters_list = self.broker_get_task_queue.get()
            if getters_list is None:
                self.logger.info("GET listener: Received stop signal. Exiting.")
                break

            comm_counts = Counter(getters_list)
            remote_comm_counts = {r: c for r, c in comm_counts.items() if r != self.rank}

            if not remote_comm_counts:
                self.logger.info(f"GET listener: No remote GET requests for this iteration.")
                self.broker_get_task_queue.task_done()
                continue

            num_expected_requests = sum(remote_comm_counts.values())
            self.logger.info(f"GET listener: New iteration. Expecting {num_expected_requests} GET requests from {remote_comm_counts}")

            for i in range(num_expected_requests):
                try:
                    size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
                    sender_rank = dist.recv(tensor=size_tensor, src=None, group=self.cpu_service_group, tag=self.GET_CTRL_TAG)

                    if sender_rank not in remote_comm_counts or remote_comm_counts[sender_rank] == 0:
                        drained_msg = self._recv_control_msg_from_size(sender_rank, size_tensor, tag=self.GET_CTRL_TAG)
                        self.logger.warning(
                            f"GET listener: Received an unexpected GET from Rank {sender_rank}. "
                            f"Drained payload and ignored message: {drained_msg}"
                        )
                        task_name = (
                            drained_msg.get("task_name", "<unknown>")
                            if isinstance(drained_msg, dict)
                            else "<unknown>"
                        )
                        try:
                            self._send_prepacked_data_error(
                                sender_rank,
                                f"Unexpected GET from rank {sender_rank} for task '{task_name}'",
                                tag=self.DATA_XFER_TAG,
                            )
                        except Exception:
                            self.logger.error(
                                f"GET listener: Failed to send error frame for unexpected GET from Rank {sender_rank}.",
                                exc_info=True,
                            )
                        continue

                    self.logger.info(f"GET listener: Received GET request #{i+1}/{num_expected_requests} from Rank {sender_rank}.")
                    
                    msg = self._recv_control_msg_from_size(sender_rank, size_tensor, tag=self.GET_CTRL_TAG)
                    task_name = msg.get("task_name")
                    
                    handler_thread = threading.Thread(target=self._handle_get_request, args=(sender_rank, task_name))
                    handler_thread.start()
                    
                    remote_comm_counts[sender_rank] -= 1

                except Exception:
                    self.logger.error("Exception in GET listener's processing loop.", exc_info=True)

            self.logger.info(f"GET listener: Finished all {num_expected_requests} expected requests for this iteration.")
            self.broker_get_task_queue.task_done()

    def _prefetch_key(self, source_task_name: str) -> Tuple[int, str]:
        return (self.iter, source_task_name)

    def _drain_remote_get_prefetch_queue(self) -> int:
        dropped = 0
        while True:
            try:
                _item = self._remote_get_prefetch_queue.get_nowait()
                self._remote_get_prefetch_queue.task_done()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def _clear_remote_get_prefetch_states(self) -> None:
        dropped = 0
        if self.enable_remote_get_prefetch:
            dropped = self._drain_remote_get_prefetch_queue()

        with self._remote_get_prefetch_lock:
            stale_states = self._remote_get_prefetch_states
            self._remote_get_prefetch_states = {}

        if stale_states or dropped > 0:
            unfinished = sum(
                1 for state in stale_states.values() if not state["done_event"].is_set()
            )
            if (unfinished > 0 or dropped > 0) and self.logger:
                self.logger.warning(
                    f"[Rank {self.rank}] Clearing remote GET prefetch states at iter boundary: "
                    f"unfinished={unfinished}, dropped_queue_items={dropped}."
                )

    def _get_remote_get_source_for_task(self, task: Task) -> Optional[str]:
        if task is None or task.task_type not in ["DiT", "DIT"] or not task.data_source_task:
            return None

        try:
            source_task = self.planner.get_task_by_name(task.data_source_task)
        except Exception:
            return None
        if source_task.task_type != "VAE":
            return None

        producer_gpus = set(source_task.gpus)
        consumer_gpus = set(task.gpus)
        if not producer_gpus.isdisjoint(consumer_gpus):
            return None

        consumer_leader = min(task.gpus)
        target_broker_rank = self._get_broker_rank_for_source_task(task.data_source_task)
        if self.rank != consumer_leader or consumer_leader == target_broker_rank:
            return None

        return task.data_source_task

    def _remote_get_prefetch_worker_loop(self):
        while True:
            item = self._remote_get_prefetch_queue.get()
            if item is None:
                self.logger.info(
                    f"[Rank {self.rank}] Remote GET prefetch worker received stop signal. Exiting."
                )
                self._remote_get_prefetch_queue.task_done()
                break

            prefetch_key = item
            prefetch_iter, source_task_name = prefetch_key

            try:
                with self._remote_get_prefetch_lock:
                    state = self._remote_get_prefetch_states.get(prefetch_key)
                if state is None:
                    continue
                now_ts = time.time()
                with self._remote_get_prefetch_lock:
                    state = self._remote_get_prefetch_states.get(prefetch_key)
                    if state is not None:
                        state["worker_started_at"] = now_ts
                target_broker_rank = state.get("broker_rank", BROKER_RANK)

                self.logger.info(
                    f"[Rank {self.rank}] Prefetch worker handling source '{source_task_name}' "
                    f"for iter {prefetch_iter}."
                )
                control_msg = {"action": "GET", "task_name": source_task_name}
                self._send_control_msg(target_broker_rank, control_msg, tag=self.GET_CTRL_TAG)
                metadata, packed_cpu_buffer = self._recv_prepacked_data(
                    target_broker_rank, tag=self.DATA_XFER_TAG
                )
                with self._remote_get_prefetch_lock:
                    state = self._remote_get_prefetch_states.get(prefetch_key)
                    if state is not None:
                        state["metadata"] = metadata
                        state["packed_cpu_buffer"] = packed_cpu_buffer
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                self.logger.error(
                    f"[Rank {self.rank}] Prefetch worker failed for source '{source_task_name}' "
                    f"(iter {prefetch_iter}): {err_msg}",
                    exc_info=True,
                )
                with self._remote_get_prefetch_lock:
                    state = self._remote_get_prefetch_states.get(prefetch_key)
                    if state is not None:
                        state["error"] = err_msg
            finally:
                with self._remote_get_prefetch_lock:
                    state = self._remote_get_prefetch_states.get(prefetch_key)
                    if state is not None:
                        state["done_at"] = time.time()
                        state["done_event"].set()
                self._remote_get_prefetch_queue.task_done()

    def _launch_remote_get_prefetch_if_needed(self, source_task_name: str) -> None:
        if not self.enable_remote_get_prefetch or not source_task_name:
            return
        target_broker_rank = self._get_broker_rank_for_source_task(source_task_name)
        if self.rank == target_broker_rank:
            return

        prefetch_key = self._prefetch_key(source_task_name)
        with self._remote_get_prefetch_lock:
            if prefetch_key in self._remote_get_prefetch_states:
                return

            inflight = sum(
                1
                for state in self._remote_get_prefetch_states.values()
                if not state["done_event"].is_set()
            )
            if inflight >= self.remote_get_prefetch_max_inflight:
                return

            state = {
                "done_event": threading.Event(),
                "metadata": None,
                "packed_cpu_buffer": None,
                "error": None,
                "broker_rank": target_broker_rank,
                "source_task_name": source_task_name,
                "created_at": time.time(),
                "worker_started_at": None,
                "done_at": None,
            }
            self._remote_get_prefetch_states[prefetch_key] = state

        self._remote_get_prefetch_queue.put(prefetch_key)

    def _schedule_remote_get_prefetches(self, task_idx: int) -> None:
        if (
            not self.enable_remote_get_prefetch
            or self.remote_get_prefetch_lookahead <= 0
            or task_idx >= len(self.my_tasks_for_this_iteration)
        ):
            return

        end_idx = min(
            len(self.my_tasks_for_this_iteration),
            task_idx + 1 + self.remote_get_prefetch_lookahead,
        )
        for idx in range(task_idx + 1, end_idx):
            source_task_name = self._get_remote_get_source_for_task(
                self.my_tasks_for_this_iteration[idx]
            )
            if source_task_name:
                self._launch_remote_get_prefetch_if_needed(source_task_name)

    def _consume_prefetched_remote_get_if_any(
        self, source_task_name: str
    ) -> Optional[Tuple[list, torch.Tensor]]:
        if not self.enable_remote_get_prefetch:
            return None

        prefetch_key = self._prefetch_key(source_task_name)
        with self._remote_get_prefetch_lock:
            state = self._remote_get_prefetch_states.get(prefetch_key)

        if state is None:
            return None

        consume_start = time.time()
        done = state["done_event"].wait(timeout=self.remote_get_prefetch_wait_timeout)
        if not done:
            created_at = state.get("created_at", consume_start)
            self.logger.warning(
                f"[Rank {self.rank}] Timeout waiting prefetched GET for '{source_task_name}' "
                f"after {self.remote_get_prefetch_wait_timeout}s "
                f"(age={consume_start - created_at:.3f}s). "
                "Falling back to synchronous wait on in-flight prefetch."
            )
            if self.remote_get_prefetch_sync_fallback_timeout > 0:
                done = state["done_event"].wait(
                    timeout=self.remote_get_prefetch_sync_fallback_timeout
                )
            else:
                state["done_event"].wait()
                done = True
            if not done:
                self.logger.warning(
                    f"[Rank {self.rank}] In-flight prefetch sync wait also timed out for "
                    f"'{source_task_name}' after extra "
                    f"{self.remote_get_prefetch_sync_fallback_timeout}s. "
                    "Will fallback to direct GET."
                )
                with self._remote_get_prefetch_lock:
                    self._remote_get_prefetch_states.pop(prefetch_key, None)
                return None

        with self._remote_get_prefetch_lock:
            state = self._remote_get_prefetch_states.pop(prefetch_key, None)

        if state is None:
            self.logger.warning(
                f"[Rank {self.rank}] Prefetch state for '{source_task_name}' disappeared; "
                "fallback to direct GET."
            )
            return None
        if state["error"]:
            self.logger.warning(
                f"[Rank {self.rank}] Prefetched GET for '{source_task_name}' failed: {state['error']}. "
                "Fallback to direct GET."
            )
            return None
        if state["metadata"] is None or state["packed_cpu_buffer"] is None:
            self.logger.warning(
                f"[Rank {self.rank}] Prefetched GET for '{source_task_name}' completed without payload. "
                "Fallback to direct GET."
            )
            return None

        created_at = state.get("created_at", consume_start)
        worker_started_at = state.get("worker_started_at", created_at)
        done_at = state.get("done_at", time.time())
        queue_delay = max(0.0, worker_started_at - created_at)
        worker_time = max(0.0, done_at - worker_started_at)
        total_age = max(0.0, done_at - created_at)
        consume_wait = max(0.0, time.time() - consume_start)

        self.logger.info(
            f"[Rank {self.rank}] Prefetch hit for source '{source_task_name}' "
            f"(broker={state.get('broker_rank', BROKER_RANK)}): "
            f"queue_delay={queue_delay:.3f}s, worker_time={worker_time:.3f}s, "
            f"total_age={total_age:.3f}s, consume_wait={consume_wait:.3f}s."
        )
        return state["metadata"], state["packed_cpu_buffer"]

    def set_active_task_by_index(self, task_idx: int):
        """Called by the outer loop to select the task to execute."""
        if task_idx < len(self.my_tasks_for_this_iteration):
            self.active_task = self.my_tasks_for_this_iteration[task_idx]
            self._schedule_remote_get_prefetches(task_idx)
        else:
            self.active_task = None                   

    def _setup_logger(self):
        """Per-rank logger setup."""
        if self.logger:          
            return

        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_rank_{self.rank}")
        self.logger.setLevel(logging.INFO)

                              
                              
        formatter = logging.Formatter(
            f"[%(asctime)s] [Rank {self.rank}] [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        main_log_file = os.path.join(self.log_dir, f"rank_{self.rank}.log")
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == main_log_file
            for h in self.logger.handlers
        ):
            main_file_handler = logging.FileHandler(main_log_file, mode="a")
            main_file_handler.setFormatter(formatter)
            self.logger.addHandler(main_file_handler)

        timing_log_file = os.path.join(self.log_dir, f"timing_rank_{self.rank}.log")
        timing_file_handler = logging.FileHandler(timing_log_file, mode="a")
        timing_file_handler.setFormatter(formatter)
        timing_file_handler.addFilter(TimingLogFilter())
        self.logger.addHandler(timing_file_handler)


        self.logger.propagate = False

    def update_stage(self, stage: int):
        self.stage = stage

    def get_my_tasks_for_this_iteration(self) -> List[Task]:
        """Return this rank's task list for the current iteration, fetching it
        from the planner when empty."""
        if not self.my_tasks_for_this_iteration:
            self.logger.info(
                f"Handler: No tasks found for rank {self.rank}, calling planner to get schedules."
            )
            self.my_tasks_for_this_iteration = self.planner.get_per_rank_schedules(
                self.world_size
            ).get(self.rank, [])
        return self.my_tasks_for_this_iteration

    def update_plan(self, filepath: str):
        """Public entry to refresh the plan; delegates to the internal planner."""
        self.logger.info(f"Handler: Received plan update request for '{filepath}'")
        self.planner.update_staged_plan_per_iter(filepath)

                                          

    def _extract_schedule_path_components(self, plan_filepath: str):
        if not plan_filepath:
            return None
        matched = re.match(r"^(.*schedule_)(\d+)(\.ya?ml)$", plan_filepath)
        if not matched:
            return None
        return matched.group(1), int(matched.group(2)), matched.group(3)

    def _collect_required_groups_from_schedule_yaml(self, plan_filepath: str):
        if not plan_filepath or not os.path.exists(plan_filepath):
            return set()
        try:
            with open(plan_filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(
                f"[Rank {self.rank}] Failed to read schedule yaml '{plan_filepath}': {e}"
            )
            return set()

        groups = set()
        for task in data.get("tasks", []) or []:
            gpus = task.get("gpus", [])
            if gpus:
                groups.add(frozenset(gpus))
        return groups

    def _collect_future_required_groups(
        self,
        plan_filepath: str,
        curr_iter: int,
    ):
        comps = self._extract_schedule_path_components(plan_filepath)
        if comps is None or self.group_cache_lookahead <= 0:
            return set()

        prefix, _, suffix = comps
        future_groups = set()
        for step in range(1, self.group_cache_lookahead + 1):
            future_iter = curr_iter + step
            future_path = f"{prefix}{future_iter}{suffix}"
            if not os.path.exists(future_path):
                continue
            future_groups |= self._collect_required_groups_from_schedule_yaml(future_path)

        return future_groups

    def update_and_setup_for_iteration(self, plan_filepath: str, curr_iter: int):
        """Per-iteration entry: load the plan and prepare this rank's tasks."""
        if self.rank is None:
            self.prepare_after_dist_initialized()

        self.logger.info(
            f"[Rank {self.rank}] Handler: Setting up for new iteration with plan '{plan_filepath}'"
        )

        self._clear_remote_get_prefetch_states()
        self.my_tasks_for_this_iteration = []
        if plan_filepath != None:
            self.update_plan(plan_filepath)
            per_rank_schedules = self.planner.get_per_rank_schedules(self.world_size)

            existing_group_keys = {
                frozenset(t.gpus) for t in self.planner._tasks.values()
            }

            from t2v_flow.planner.grad_sync_planner import build_group_aware_grad_sync_plan
            self.grad_sync_plan = build_group_aware_grad_sync_plan(
                per_rank_schedules=per_rank_schedules,
                world_size=self.world_size,
                existing_group_keys=existing_group_keys,
                max_combo_k=2,
            )

            if self.rank == 0:
                bcast_plan_for_log = {
                    int(rep): {
                        "group": (
                            sorted(list(group_key))
                            if group_key is not None
                            else "world"
                        ),
                        "ranks": sorted(list(ranks)),
                    }
                    for rep, (group_key, ranks) in sorted(
                        self.grad_sync_plan.bcast_plan.items()
                    )
                }
                self.logger.info(
                    f"[Iter {curr_iter}] GradSyncPlan: "
                    f"reps={self.grad_sync_plan.reps}, "
                    f"rep_group_key={sorted(self.grad_sync_plan.rep_group_key)}, "
                    f"reuse_rep_group={self.grad_sync_plan.reuse_rep_group}, "
                    f"created_groups={len(self.grad_sync_plan.created_groups)}"
                )
                self.logger.info(
                    f"[Iter {curr_iter}] GradSyncPlan detail: "
                    f"created_group_keys={ [sorted(list(g)) for g in self.grad_sync_plan.created_groups] }, "
                    f"bcast_plan={bcast_plan_for_log}"
                )

            grad_comm_groups = set()
            if self.grad_sync_plan.reps:
                grad_comm_groups.add(self.grad_sync_plan.rep_group_key)
            for _rep, (bcast_group_key, _ranks) in self.grad_sync_plan.bcast_plan.items():
                if bcast_group_key is not None:
                    grad_comm_groups.add(bcast_group_key)
            self.planner.set_extra_required_groups(grad_comm_groups)

            if self.enable_group_cache:
                required_now_groups = self.planner.get_all_required_gpu_groups()
                required_future_groups = self._collect_future_required_groups(
                    plan_filepath=plan_filepath,
                    curr_iter=curr_iter,
                )
                mpu.maybe_prune_custom_groups(
                    required_now_groups=required_now_groups,
                    required_future_groups=required_future_groups,
                    curr_iter=curr_iter,
                    logger=self.logger,
                    gc_interval=self.group_cache_gc_interval,
                    max_cached_groups=self.group_cache_max_groups,
                    max_local_group_memberships=self.group_cache_max_local_memberships,
                )
            else:
                mpu.destroy_custom_groups(self.logger)
            mpu.initialize_custom_groups(
                self.planner, self.logger
            )                

                                                                                       
            self.my_tasks_for_this_iteration = per_rank_schedules.get(self.rank, [])


        self.stage = 0          
        self.iter = curr_iter
        if plan_filepath is not None:
            self._update_broker_routing_for_iteration()
        else:
            self.source_task_to_broker_rank = {}
            self.broker_publishers_by_rank = {
                broker_rank: [] for broker_rank in self.broker_ranks
            }
            self.broker_getters_by_rank = {
                broker_rank: [] for broker_rank in self.broker_ranks
            }

        if self.rank == 0:
            self.logger.info(
                f"[Iter {curr_iter}] source->broker routes: {self.source_task_to_broker_rank}"
            )
            self.logger.info(
                f"[Iter {curr_iter}] broker publishers: {self.broker_publishers_by_rank}"
            )
            self.logger.info(
                f"[Iter {curr_iter}] broker getters: {self.broker_getters_by_rank}"
            )

        if self.rank in self.broker_ranks:
            publishers = self.broker_publishers_by_rank.get(self.rank, [])
            getters = self.broker_getters_by_rank.get(self.rank, [])
            self.logger.info(
                f"Broker Rank {self.rank}: Updating listener list for iter {curr_iter}. "
                f"Publishers: {publishers}, Getters: {getters}"
            )
            self.broker_publish_task_queue.put(publishers)
            self.broker_get_task_queue.put(getters)
                                                        
        
                        
        task_count = len(self.my_tasks_for_this_iteration)

        DATA_CACHE.clear()

        if self.rank in self.broker_ranks:
            with self.broker_lock:
                self.central_data_store.clear()
                self.data_events.clear()
        self.logger.info(
            f"[Rank {self.rank}] Handler setup complete. I have {task_count} tasks for this iteration."
        )

    def get_mock_batch(self, data_iterator=None, current_task = None):
        timers = get_timers()

                        
        timers("batch-generator", log_level=2).start()

        self.logger.info(
            f"rank {self.rank} {self.my_tasks_for_this_iteration[self.stage]})"
        )
        
        timers("batch-generator").stop()
                                                 
                                                       

        if os.environ.get("USE_FAKE_BATCH") == "1":

                                        
            if os.environ.get("EXPERIMENT") == "1":
                batch = self.create_tensors_from_rank_shape(current_task.name)
            else:
                                   
                                                    
                if os.environ.get('FIXED_SHAPE') == "1":
                    batch = self.get_mock_fixed_batch()
                else:
                    batch = self.get_mock_dynamic_batch(self.get_data_list())
                                                                                  
        else :
                                 
            batch = self.get_batch(data_iterator)
        os.environ["bs"] = str(batch["images"].shape[0])
        os.environ["height"] = str(batch["images"].shape[3])
        os.environ["width"] = str(batch["images"].shape[4])    
        os.environ["frames"] = str(batch["images"].shape[1])
        os.environ["sp"] = str(mpu.get_context_parallel_world_size())
        self.logger.info(
            f'rank {dist.get_rank()} images = {batch["images"].shape}, prompt_embeds = {batch["prompt_embeds"].shape}, clip = {batch["clip_text_embed"].shape}'
        )
        return batch

    def forward_step(
        self, data_iterator, model: HunyuanPipeline | TrainingWrapperModel
    ):
        """Forward training step.

        Args:
            data_iterator: Iterable dataset.
            model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

        Returns:
            output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
            loss_func (callable): Loss function with a loss mask specified.
        """
                                               
                                                                    
                          

                                                       
        try:
            output_tensor_list = self._execute_task(
                task=self.active_task,
                data_iterator=data_iterator,
                model=model,
            )
        except (RuntimeError, torch.distributed.DistBackendError) as e:
            error_str = str(e).lower()
            if any(k in error_str for k in [
                "out of memory",
                "timeout",
                "nccl",
                "connection",
                "unhandled system error",
                "closed"
            ]):
                self.logger.error(
                    f"[Rank {self.rank}] Distributed failure at task "
                    f"{self.active_task.name if self.active_task else 'unknown'}: {e}"
                )
                raise
            else:
                raise
                                           
                                                
                         
        self.stage += 1
        return output_tensor_list, self.loss_func

    def _get_dependency_data(
        self, consumer_task, consumer_group
    ) -> Dict[str, torch.Tensor]:
        source_task_name = consumer_task.data_source_task
        if not source_task_name:
            raise ValueError(
                f"DiT task '{consumer_task.name}' needs 'data_source_task' arg."
            )

        source_task = self.planner.get_task_by_name(source_task_name)
        global_source_rank = min(source_task.gpus)                   
        current_device = torch.device("cuda", self.rank % torch.cuda.device_count())
        consumer_gpus = set(consumer_task.gpus)
        producer_gpus = set(source_task.gpus)

        precomputed_bundle = None

        if consumer_gpus.issubset(producer_gpus):
                                                                        
                                                                        
                                                                                                         
                                                            

            self.logger.info(
                f"[Rank {self.rank}] Task '{consumer_task.name}': Using in-place data (Subset dependency)."
            )
            if source_task_name not in DATA_CACHE:
                raise RuntimeError(
                    f"Rank {self.rank} expected data for '{source_task_name}' "
                    f"in local cache for in-place dependency, but it was not found."
                )
            precomputed_bundle = DATA_CACHE[source_task_name]

        else:
            intersection = consumer_gpus.intersection(producer_gpus)
            if intersection:
  
                BROADCAST_TENSOR = True
                broadcast_src_rank = min(intersection)
                from my_utils import global_timer
                from t2v_flow.executor import TensorUtils
                self.logger.info(
                        f"[Rank {self.rank}] Task '{consumer_task.name}': Partial overlap dependency. "
                        f"Broadcasting from rank {broadcast_src_rank} within consumer group."
                    )
                global_timer.start("broadcast comm overhead")
                if self.rank == broadcast_src_rank:
                    data_to_send = [DATA_CACHE[source_task_name]]
                                                                                                                 
                else:
                    data_to_send = [None]

                received_package = None
                if BROADCAST_TENSOR:
                                       
                                                                                                                                             
                       
                    dist.barrier(group=consumer_group)
                    received_package = TensorUtils.broadcast_nested_tensor_package_gpu(
                        package=data_to_send[0],
                        src=broadcast_src_rank,
                        group=consumer_group,
                        device=f"cuda:{self.rank % torch.cuda.device_count()}",                
                    )

                else:
                    self.logger.info(
                        f"[Rank {self.rank}] Broadcasting data for task '{consumer_task.name}' from source rank {global_source_rank}..."
                    )
                    dist.broadcast_object_list(
                        data_to_send, src=global_source_rank, group=consumer_group
                    )
                    received_package = data_to_send[0]

                                                                                
                                                                                

                    if self.rank != global_source_rank and received_package is not None:
                        global_timer.start("move cuda device overhead")
                        
                        self.logger.info(
                            f"  > Rank {self.rank}: Moving received data to local device {current_device}..."
                        )

                        for key, value in received_package.items():
                            if isinstance(value, torch.Tensor):
                                received_package[key] = value.to(current_device)
                            elif isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, torch.Tensor):
                                        value[sub_key] = sub_value.to(current_device)
                                                                                    
                        global_timer.stop("move cuda device overhead")

                global_timer.stop("broadcast comm overhead")
                precomputed_bundle = received_package
            
            else:
                producer_leader = min(producer_gpus)
                consumer_leader = min(consumer_gpus)
                target_broker_rank = self._get_broker_rank_for_source_task(source_task_name)
                
                self.logger.info(
                    f"Task '{consumer_task.name}': Disjoint GPUs. Performing P2P "
                    f"({producer_leader} -> {consumer_leader}) + Broadcast via broker {target_broker_rank}."
                )

                metadata, packed_gpu_buffer = None, None
                from my_utils import global_timer
                                                  
                    
                if self.rank == consumer_leader:
                    global_timer.start("get_and_move_data_overhead") 
                    if consumer_leader == target_broker_rank:
                        self.logger.info(f"Rank {self.rank} (Broker as Consumer): Accessing local store for '{source_task_name}'.")
                        event = None
                        
                        try:
                            with self.broker_lock:
                                if source_task_name in self.central_data_store:
                                    self.logger.info(f"Rank {self.rank} (Broker as Consumer): Data '{source_task_name}' found immediately in store.")
                                    metadata, packed_cpu_buffer = self.central_data_store[source_task_name]
                                else:
                                    self.logger.info(f"Rank {self.rank} (Broker as Consumer): Data '{source_task_name}' not yet available. Preparing to wait.")
                                    from my_utils import DebuggingEvent
                                    event = self.data_events.setdefault(source_task_name, DebuggingEvent(logger=self.logger))
                            
                            if event:
                                is_ready = event.wait(timeout=self.broker_get_wait_timeout)
                                if not is_ready:
                                    raise RuntimeError(
                                        f"Rank {self.rank} (Broker) timed out after "
                                        f"{self.broker_get_wait_timeout:.1f}s waiting for local data "
                                        f"'{source_task_name}'."
                                    )

                                self.logger.info(f"Rank {self.rank} (Broker as Consumer): Event received. Retrieving '{source_task_name}' from store.")
                                with self.broker_lock:
                                    metadata, packed_cpu_buffer = self.central_data_store[source_task_name]
                                    del self.central_data_store[source_task_name]
                                    if source_task_name in self.data_events:
                                        del self.data_events[source_task_name]

                            self.logger.info(f"Rank {self.rank} (Broker as Consumer): Data '{source_task_name}' retrieved from local store.")
                        
                        except Exception as e:
                            self.logger.error(f"Rank {self.rank} (Broker as Consumer): Error getting local data for '{source_task_name}'", exc_info=True)
                            raise e
                    else:
                        prefetched = self._consume_prefetched_remote_get_if_any(
                            source_task_name
                        )
                        if prefetched is not None:
                            metadata, packed_cpu_buffer = prefetched
                        else:
                            self.logger.info(
                                f"Rank {self.rank} (Consumer Leader): Requesting '{source_task_name}' "
                                f"from Broker Rank {target_broker_rank} via tagged channel."
                            )
                            control_msg = {"action": "GET", "task_name": source_task_name}
                            self._send_control_msg(target_broker_rank, control_msg, tag=self.GET_CTRL_TAG)
                            metadata, packed_cpu_buffer = self._recv_prepacked_data(
                                target_broker_rank, tag=self.DATA_XFER_TAG
                            )
                                                                                
                
                    packed_gpu_buffer = packed_cpu_buffer.to(current_device, non_blocking=True)
                    self.logger.info(f"Rank {self.rank} (Consumer Leader): Data loaded to GPU.")
                    global_timer.stop("get_and_move_data_overhead")



                from t2v_flow.executor import TensorUtils
                metadata_list = [metadata] if self.rank == consumer_leader else [None]
                dist.broadcast_object_list(metadata_list, src=consumer_leader, group=consumer_group)
                final_metadata = metadata_list[0]
                
                if self.rank != consumer_leader:
                    packed_gpu_buffer = TensorUtils._create_empty_buffer_from_metadata(final_metadata, current_device)
                dist.broadcast(packed_gpu_buffer, src=consumer_leader, group=consumer_group)
                precomputed_bundle = TensorUtils._unpack_buffer_to_package(packed_gpu_buffer, final_metadata)
        if precomputed_bundle is None:
            raise RuntimeError(
                f"Rank {self.rank} failed to obtain precomputed data for task '{consumer_task.name}'."
            )

        return precomputed_bundle

    def _execute_task(
        self, task: Task, data_iterator, model, batch: dict = None
    ) -> List[torch.Tensor]:
        to_gb = lambda b: b / (1024 ** 3)
    
        allocated_start_gb = to_gb(torch.cuda.memory_allocated())
        reserved_start_gb = to_gb(torch.cuda.memory_reserved())
        self.logger.info(
            f"[TASK START] Rank {self.rank}: Starting task '{task.name}' (Type: {task.task_type}, GPUs: {task.gpus})"
        )
        self.logger.info(
        f"  [MEM START] Allocated: {allocated_start_gb:.3f}GB | "
        f"Reserved: {reserved_start_gb:.3f}GB"
    )
                                                      
                                                
        try:
            if task.task_type == "VAE":
                                         
                batch = self.get_mock_batch(data_iterator=data_iterator, current_task=task)
                torch.cuda.empty_cache()
                if mpu.get_context_parallel_world_size() > 1:
                    
                    dist.barrier(mpu.get_context_parallel_group())
                encoded_bundle = model(batch, mode="vae_only")
                                                          
                context_package = {"batch_dict": batch, "encoded_bundle": encoded_bundle}
                if task.has_overlapping_consumer:
                    DATA_CACHE[task.name] = context_package
                    self.logger.info(
                        f"Rank {self.rank}: Consumer overlaps. Storing '{task.name}' in local GPU cache."
                    )
                else:
                    if self.rank == min(task.gpus):
                        self.publish_queue.put((task.name, context_package))
                        self.logger.info(
                            f"Rank {self.rank}: Consumer is disjoint. Submitting '{task.name}' to publisher queue."
                        )
                                   
                                                                                   
                   

                output_tensor_list = [torch.tensor(0.0, device="cuda")]
            elif task.task_type in ["DiT", "DIT"]:
                consumer_group = mpu.get_context_parallel_group()
                context_package = self._get_dependency_data(
                    consumer_task=task, consumer_group=consumer_group
                )
                                     
                torch.cuda.empty_cache()
                if mpu.get_context_parallel_world_size() > 1:
                
                    dist.barrier(group=consumer_group)
                    self.logger.info(
                        f"[Rank {self.rank}] Barrier after getting dependency data for task '{task.name}'"
                    )
                                                                                                                                                                                                     
                self.logger.info(f"[after get dependency] latents shape is {context_package['encoded_bundle']['latents'].shape}")
                output_tensor_list = model(
                    context_package["batch_dict"],
                    mode="dit_only",
                    precomputed_encoded_bundle=context_package["encoded_bundle"],
                )

                                          
            elif task.task_type == "FULL":
                if self.stage == 0:
                    batch = self.get_mock_batch(data_iterator=data_iterator, current_task=task)
                                
                output_tensor_list = model(batch, mode="full")
        except Exception as e:
            if _is_distributed_fatal_error(e):
                self.logger.error(
                    f"[Rank {self.rank}] Distributed failure at task "
                    f"{task.name if task else 'unknown'}: {e}"
                )
                _record_resume_step(self.iter)
                                      
                _abort_distributed_hard()
                os._exit(100)
            raise
        allocated_end_gb = to_gb(torch.cuda.memory_allocated())
        reserved_end_gb = to_gb(torch.cuda.memory_reserved())
        
        self.logger.info(f"[TASK END]   Rank {self.rank}: Finished task '{task.name}'.")
        self.logger.info(
            f"  [MEM END]   Allocated: {allocated_end_gb:.3f}GB | "
            f"Reserved: {reserved_end_gb:.3f}GB  | "
            f"  [MEM DELTA] Allocated: {allocated_end_gb - allocated_start_gb:+.3f}GB | "
            f"Reserved: {reserved_end_gb - reserved_start_gb:+.3f}GB"
        )
        return output_tensor_list

    def get_batch(self, data_iterator):
                                                        
        batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
                                                               
        return batch

    def loss_func(self, output_tensor):
        """Loss function."""
        loss = output_tensor[0].mean()
        self.logger.info(f"[loss_func] rank {dist.get_rank()} loss = {loss}")
                                               
                                                                                     
                                 
                                                              

                                                    
                                            
                                
                                                                           
                                  
        loss = loss.unsqueeze(0)
                                                 
        return loss, {"loss": loss}

    def get_mock_batch_flex4211(self):
        global_rank = dist.get_rank()
        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        images = None
        group_ranks = None
        if global_rank in [0, 1, 2, 3]:
            bs = 1
            group_ranks = [6]
                                                                                               
            images = torch.zeros(
                (bs, 241, 3, 1936, 1072), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 112, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [
            4,
            5,
        ]:
            images = torch.zeros(
                (1, 221, 3, 1936, 1072), dtype=torch.float16, device=device
            )
                                                                                             
                                                                                            
            group_ranks = [4, 5]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [6]:
            bs = 1
            group_ranks = [7]
            images = torch.zeros(
                (bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device
            )
                                                                                                
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 120, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [7]:
            images = torch.zeros(
                (1, 97, 3, 1072, 1936), dtype=torch.float16, device=device
            )
            group_ranks = [0, 1, 2, 3]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 115, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        else:
            raise ValueError(f"Unsupported rank {rank}")
                                                                         
        from my_utils import global_timer

                                                  
                                                           
                                                 
                                             
                                        
                                            
                                                           
        return batch

    def get_mock_batch_8422(self):
        """Mock batches with per-rank-group frame counts (debug helper)."""
        global_rank = dist.get_rank()
        local_rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        height, width = 720, 1280
        batch = None

        if global_rank in [0, 1]:
            bs = 6
            num_frames = 13
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 112, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }

        elif global_rank in [2, 3]:
            bs = 4
            num_frames = 57
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 115, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }

        elif global_rank in [4, 5, 6, 7]:
            bs = 2
            num_frames = 89
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 120, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }

        elif global_rank in range(8, 16):
            bs = 2
            num_frames = 129
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }

        else:
            raise ValueError(
                f"Unsupported global_rank {global_rank} for mock batch generation."
            )

        return batch

    def get_mock_batch_dynamic_frames(self):
        """Mock batch whose size/frames vary with self.iter (debug helper)."""
                                                      

        global_rank = dist.get_rank()

        if not (0 <= global_rank < 8):
            raise ValueError(f"designed for ranks 0-7 only, got rank {global_rank}")

        local_rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        num_frames = 101 + 8 * (global_rank // mpu.get_context_parallel_world_size())
        
                                                                             
                                                                             
        if self.iter == 0:
            bs = 1
        else:
                                                            
                                                            
                                                        
            calculated_bs = 2 * self.iter
            os.environ["VAE_PROFILE_BATCH"] = str(calculated_bs)
            bs = calculated_bs
                                                                             

        height = 720
        width = 1280

        images = torch.zeros(
            (bs, num_frames, 3, width, height), dtype=torch.float16, device=device
        )

        batch = {
            "images": images,
            "prompt_embeds": torch.zeros(
                (bs, 112, 4096), dtype=torch.float16, device=device
            ),
            "clip_text_embed": torch.zeros(
                (bs, 768), dtype=torch.float16, device=device
            ),
            "first_ref_image": torch.zeros_like(images[:, :1]),
        }

        if global_rank == 0:                     
            print(
                f"[Iter {self.iter}] Dynamic BS={bs}. [Rank {global_rank}] Batch created. Images shape: {images.shape}"
            )

        return batch

    def get_mock_batch_flex4444(self):
        global_rank = dist.get_rank()
        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        images = None
        group_ranks = None
        if global_rank in [0, 1, 2, 3, ]:
                                     
            bs = 1
            group_ranks = [6]
            frames = 57 
            images = torch.zeros(
                (bs, frames, 3, 1280, 720), dtype=torch.float16, device=device
            )
                                   
                                                                            
               
                                                                                                
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 112, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
                                 
        elif global_rank in [4,  5, 6, 7]:
            bs = 1
                                                                                               
            frames = 129
            images = torch.zeros(
                (bs, frames, 3, 1280, 720), dtype=torch.float16, device=device
            )

                                                                                              
            group_ranks = [4, 5]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif global_rank in [8, 9, 10, 11]:
            bs = 1
            group_ranks = [7]
                                                                                                
            images = torch.zeros(
                (bs, 89, 3, 720, 1280), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 120, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif global_rank in [12, 13, 14, 15]:
            bs = 1
                                                                                               
            images = torch.zeros(
                (bs, 57, 3, 720, 1280), dtype=torch.float16, device=device
            )
            group_ranks = [0, 1, 2, 3]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 115, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        else:
            raise ValueError(f"Unsupported rank {rank}")
                                                                         
        from my_utils import global_timer

                                                  
                                                           
                                                 
                                             
                                        
                                            
                                                           
        return batch

    def get_mock_batch_new_plan(self):
        global_rank = dist.get_rank()
        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        images = None
        group_ranks = None
        if global_rank in [4, 5, 6, 7]:
            bs = 1
            group_ranks = [6]
                                                                                               
            images = torch.zeros(
                (bs, 241, 3, 1936, 1072), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 112, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [
            2,
            3,
        ]:
            images = torch.zeros(
                (1, 221, 3, 1936, 1072), dtype=torch.float16, device=device
            )
                                                                                             
                                                                                            
            group_ranks = [4, 5]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [1]:
            bs = 1
            group_ranks = [7]
            images = torch.zeros(
                (bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device
            )
                                                                                                
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 120, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif rank in [0]:
            images = torch.zeros(
                (1, 97, 3, 1072, 1936), dtype=torch.float16, device=device
            )
            group_ranks = [0, 1, 2, 3]
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (1, 115, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (1, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        else:
            raise ValueError(f"Unsupported rank {rank}")
                                                                         
        from my_utils import global_timer

                                                  
                                                           
                                                 
                                             
                                        
                                            
                                                           
        return batch

    def get_batch_with_mock(self):
                                 
                                              
                                               
                                               
                                               
                                              
                                               
                                                               
        fake = self.get_mock_dynamic_batch(self.get_data_list())
                                                     
                                           
                                                             
                                                         
                            
        return fake

    def get_mock_batch_all_129_frames(self):
        """Mock 129-frame batches for all ranks (debug helper)."""
        global_rank = dist.get_rank()
        local_rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        height, width = 1280, 720
        batch = None

        if global_rank in range(0, 12):
            bs = 1
            num_frames = 129 - self.iter * 4
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        elif global_rank in [12, 13, 14, 15]:
            bs = 1
            num_frames = 5
            images = torch.zeros(
                (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
            )
            batch = {
                "images": images,
                "prompt_embeds": torch.zeros(
                    (bs, 124, 4096), dtype=torch.float16, device=device
                ),
                "clip_text_embed": torch.zeros(
                    (bs, 768), dtype=torch.float16, device=device
                ),
                "first_ref_image": torch.zeros_like(images[:, :1]),
            }
        else:
            raise ValueError(
                f"Unsupported global_rank {global_rank} for mock batch generation."
            )

        return batch

    def get_bucket_data_list(self):
        """Compute the max feasible batch size per (frames, resolution) bucket
        and emit the data list."""
        width = 1280
        height = 720
        channels = 3
        
        ESTABLISHED_THRESHOLD = 31027200.0

        try:
            sp = mpu.get_context_parallel_world_size()
        except (NameError, ImportError):
            print("[warn] mpu unavailable; defaulting sp to 4.")
            sp = 4                   

        data_list = []
        import math
        for num_frames in range(5, 141 + 1, 4):
            
                                                                            
            
            denominator = num_frames * width * height
            if denominator == 0:
                max_bs = 0
            else:
                max_bs = math.floor((ESTABLISHED_THRESHOLD * sp) / denominator)
            
            if max_bs < 1:
                max_bs = 1
            
            data_list.append(
                (int(max_bs), num_frames, channels, width, height)
            )

        print(f"generated {len(data_list)} data combinations.")
        print("sample (max_bs, num_frames, C, H, W):")
        if len(data_list) > 6:
            for item in data_list[:3]:
                print(item)
            print("...")
            for item in data_list[-3:]:
                print(item)
        else:
            for item in data_list:
                print(item)

        return data_list
    

    def get_mock_fixed_batch(self, bs=4, frames=129, height=352, width=656):
        rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        images = torch.zeros(
            (bs, frames, 3, height, width), dtype=torch.float16, device=device
        )

        batch = {
            "images": images,
            "prompt_embeds": torch.zeros(
                (bs, 256, 4096), dtype=torch.float16, device=device
            ),
            "clip_text_embed": torch.zeros(
                (bs, 768), dtype=torch.float16, device=device
            ),
            "first_ref_image": torch.zeros_like(images[:, :1]),
        }
        return batch

    def get_data_list(self):
                                  
        bs = [1]
        h = 353 
        w = 656 
        shape = [
            [num_frames, 3, h, w] 
            for num_frames in [129]
        ]
        sp = mpu.get_context_parallel_world_size()


        data_list = [
            (b, *s) for b in bs for s in shape
                                                                
        ]

                      
                                                  
                                            
           
                         

        return data_list

    def get_mock_dynamic_batch(self, data_list):
        global_rank = dist.get_rank()

        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.cuda.current_device()


        if (self.iter + 1) % 40 == 0 :
            return self.get_mock_batch_flex4444()
                                               
   
        total_pool_size = len(data_list)
        sp = mpu.get_context_parallel_world_size()

        n = torch.distributed.get_world_size()

        need_to_get = n // sp

        start_index = ((self.iter+1) % 40 - 1 ) * need_to_get
        if global_rank == 0:
                    print(f"-------------------- Iteration {self.iter} --------------------")
                    print(f"📊 [Rank 0 Log] Data Pool Status: {total_pool_size} items total.")
                    print(f"⚙️  [Rank 0 Log] Trying to get {need_to_get} items starting from index {start_index}.")
        if start_index >= len(data_list):
            return self.get_mock_batch_flex4444()

        data_list = data_list[start_index : start_index + need_to_get]

        if len(data_list) < need_to_get:
            padding_size = need_to_get - len(data_list)
            for i in range(padding_size):
                data_list.append(data_list[-1])
                                 

        rank_data = data_list[global_rank // sp]

        bs = rank_data[0]
        num_frames = rank_data[1]
        height = rank_data[3]
        width = rank_data[4]

        images = torch.zeros(
            (bs, num_frames, 3, height, width), dtype=torch.float16, device=device
        )

        batch = {
            "images": images,
            "prompt_embeds": torch.zeros(
                (images.shape[0], 256, 4096), dtype=torch.float16, device=device
            ),
            "clip_text_embed": torch.zeros(
                (images.shape[0], 768), dtype=torch.float16, device=device
            ),
            "first_ref_image": torch.zeros_like(images[:, :1]),
        }
        return batch

    def get_mock_batch_uniform(self, images=None):
        global_rank = dist.get_rank()
        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        images = None

        bs = 1

        images = torch.zeros(
            (bs, 241, 3, 1936, 1072), dtype=torch.float16, device=device
        )

        batch = {
            "images": images,
            "prompt_embeds": torch.zeros(
                (bs, 112, 4096), dtype=torch.float16, device=device
            ),
            "clip_text_embed": torch.zeros(
                (bs, 768), dtype=torch.float16, device=device
            ),
            "first_ref_image": torch.zeros_like(images[:, :1]),
        }

        return batch

    def get_per_rank_schedules(self, world_size: Optional[int] = None):
        """Delegate to the planner: full per-rank task sequences.
        Returns Dict[rank, List[Task]]."""
        if self.task_list is None or len(self.task_list) == 0:
            self.logger.info(
                f"Handler: No task list found, calling planner to get schedules for world size {world_size}"
            )
            self.task_list = self.planner.get_per_rank_schedules(world_size)[self.rank]
        else:
            return self.task_list
        

    def create_tensors_from_rank_shape(self,
    task_name: str, 
    dtype: torch.dtype = torch.float16
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.iter  == -1:
            return self.get_mock_batch_flex4444()
        
        global_rank = dist.get_rank()
        rank = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        from simulation_data import IterationLogParser
        iter_parser = IterationLogParser()
        iter_data = iter_parser.get_iteration(iteration_number=self.iter)
                                                                   
                                                                                                       
                                                    
                                                        
                                                      
               
        parts = task_name.split('_')
        first_five_parts = parts[:5]
        result = "_".join(first_five_parts)
        rank_data = iter_data[result]
        if not rank_data or not isinstance(rank_data, dict):
            print("error: invalid rank_data input.")
            return None

        image_shape = rank_data.get("images")
        if not image_shape or len(image_shape) < 5:
            print(f"error: 'images' shape missing/incomplete: {image_shape}")
            return None
            
                                                   
        bs = image_shape[0]
        num_frames = image_shape[1]
        channels = image_shape[2]
        height = image_shape[3]
        width = image_shape[4]
            
        images = torch.zeros(
            (bs, num_frames, channels, height, width), 
            dtype=dtype, 
            device=device
        )

        prompt_embeds = torch.zeros(
            (bs, 226, 4096), 
            dtype=dtype, 
            device=device
        )

        clip_text_embed = torch.zeros(
            (bs, 768), 
            dtype=dtype, 
            device=device
        )

        first_ref_image = torch.zeros_like(images[:, :1])

        batch = {
            "images": images,
            "prompt_embeds": prompt_embeds,
            "clip_text_embed": clip_text_embed,
            "first_ref_image": first_ref_image,
        }

        return batch
