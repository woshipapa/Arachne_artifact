# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import logging
from contextlib import contextmanager
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from .pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from .utils import StatelessProcessGroup, get_current_device_stream_fast
# from sglang.srt.utils.common import get_current_device_stream_fast

logger = logging.getLogger(__name__)

class PyNcclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
        use_current_stream: bool = False,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.NCCL
            ), "PyNcclCommunicator should be attached to a non-NCCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
            if self.rank == 0:
                print(
                    f"[PyNccl] init: world_size={self.world_size} backend={dist.get_backend(group)}"
                )
        else:
            self.rank = group.rank
            self.world_size = group.world_size
            if self.rank == 0:
                print(f"[PyNccl] init: world_size={self.world_size} stateless_group=True")

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            self.stream = None
            if self.rank == 0:
                print("[PyNccl] skipped: world_size=1")
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception as e:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            self.stream = None
            if self.rank == 0:
                print(f"[PyNccl] disabled: NCCL library load failed e = {e}")
            return

        self.available = True
        self.disabled = False
        self.use_current_stream = use_current_stream

        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            nccl_version = self.nccl.ncclGetVersion()
            print(f"[PyNccl] nccl version={nccl_version}")

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
            print("[PyNccl] got ncclUniqueId on rank 0")
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            if self.rank == 0:
                print(
                    f"[PyNccl] broadcast unique_id via gloo group src={ranks[0]} "
                    f"ranks={ranks}"
                )
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
            if self.rank == 0:
                print("[PyNccl] broadcast unique_id via stateless group")
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        print(
            f"[PyNccl] rank={self.rank} init comm on device={device} "
            f"(world_size={self.world_size})"
        )
        with torch.cuda.device(device):
            print(f"[PyNccl] rank={self.rank} calling ncclCommInitRank")
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )
            print(f"[PyNccl] rank={self.rank} ncclCommInitRank done")
            self.stream = torch.cuda.Stream()
            print(
                f"[PyNccl] rank={self.rank} stream created "
                f"(use_current_stream={self.use_current_stream})"
            )

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            print(f"[PyNccl] rank={self.rank} warmup all_reduce start")
            self.all_reduce(data)
            self.stream.synchronize()
            print(f"[PyNccl] rank={self.rank} warmup all_reduce done")
            del data
            if self.rank == 0:
                print("[PyNccl] warmup all_reduce done")

        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        self.disabled = True

    def _resolve_stream(self, stream: Optional[torch.cuda.Stream]):
        """Return the stream to use for NCCL calls.

        Behavior mirrors the previous inline logic:
        - if an explicit stream is provided, return it
        - if stream is None and self.use_current_stream is True, return
          torch.cuda.current_stream()
        - otherwise return the communicator's default stream (self.stream)
        """
        if stream is not None:
            return stream
        if self.use_current_stream:
            return get_current_device_stream_fast()
        return self.stream

    def all_reduce(
        self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream(stream)
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream=None,
        sizes: Optional[list[int]] = None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        stream = self._resolve_stream(stream)

        if sizes is not None:
            split_offset = 0

            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                dst_slice = output_tensor[split_offset : split_offset + split_size]
                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            self.nccl.ncclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

    def all_toall(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream=None,
    ) -> None:
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        assert output_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the output tensor is on {output_tensor.device}"
        )
        if output_tensor.numel() != input_tensor.numel():
            raise ValueError(
                "all_toall requires input/output tensors with the same numel"
            )
        if output_tensor.numel() % self.world_size != 0:
            raise ValueError(
                "all_toall requires numel divisible by world_size"
            )
        stream = self._resolve_stream(stream)
        if "ncclAllToAll" in self.nccl._funcs:
            count = output_tensor.numel() // self.world_size
            self.nccl.ncclAllToAll(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                count,
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            return

        if not output_tensor.is_contiguous():
            raise ValueError("all_toall fallback requires contiguous output tensor")
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        inp = input_tensor.view(-1)
        out = output_tensor.view(-1)
        chunk = out.numel() // self.world_size
        out[self.rank * chunk : (self.rank + 1) * chunk].copy_(
            inp[self.rank * chunk : (self.rank + 1) * chunk]
        )
        self.group_start()
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            send_slice = inp[peer * chunk : (peer + 1) * chunk]
            recv_slice = out[peer * chunk : (peer + 1) * chunk]
            self.send(send_slice, peer, stream=stream)
            self.recv(recv_slice, peer, stream=stream)
        self.group_end()

    def cp_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream=None,
        sizes: Optional[list[int]] = None,
    ):
        """
        Currently, it is mainly used in context parallelism,
        primarily leveraging pynccl to implement non-blocking allgather communication.
        """
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        stream = self._resolve_stream(stream)
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
        sizes: Optional[list[int]] = None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        stream = self._resolve_stream(stream)

        if sizes is not None:
            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                chunk = input_tensor[split_offset : split_offset + split_size, ...]

                self.nccl.ncclReduce(
                    buffer_type(chunk.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    chunk.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    ncclRedOpTypeEnum.from_torch(op),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            self.nccl.ncclReduceScatter(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                output_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream(stream)
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream(stream)
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream(stream)

        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def register_comm_window_raw(self, ptr: int, size: int):
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    def _require_amem(self, func_name: str) -> None:
        if func_name not in self.nccl._funcs:
            raise RuntimeError(
                f"{func_name} is not available in the loaded NCCL library."
            )

    def pause(self, comm: Optional[ncclComm_t] = None) -> None:
        self._require_amem("ncclPause")
        self.nccl.ncclPause(comm)

    def resume(self, comm: Optional[ncclComm_t] = None) -> None:
        self._require_amem("ncclResume")
        self.nccl.ncclResume(comm)

    def mem_stats(self) -> None:
        self._require_amem("ncclMemStats")
        self.nccl.ncclMemStats()

    def set_group_id(self, gid: int) -> None:
        self._require_amem("ncclSetGroupID")
        self.nccl.ncclSetGroupID(gid)

    def get_group_id(self) -> int:
        self._require_amem("ncclGetGroupID")
        return self.nccl.ncclGetGroupID()

    @contextmanager
    def change_state(
        self, enable: Optional[bool] = None, stream: Optional[torch.cuda.Stream] = None
    ):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream
