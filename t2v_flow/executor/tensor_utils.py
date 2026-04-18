import pickle
import time
from typing import Dict, Any

import torch
import torch.distributed as dist
from loguru import logger


class TensorUtils:
    """
    一个用于在 PyTorch 分布式环境中高效传输嵌套 Tensor 字典的工具类。
    支持一对多 (broadcast) 和一对一 (send/recv) 两种通信模式。
    """

    # --------------------------------------------------------------------------
    # 内部辅助方法 (Internal Helper Methods)
    # --------------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(package: Dict[str, Dict[str, torch.Tensor]]) -> list:
        """从数据包中提取元数据（结构、形状、类型等）。"""
        metadata = []
        for outer_key, inner_dict in package.items():
            for inner_key, tensor in inner_dict.items():
                assert isinstance(tensor, torch.Tensor), "Package values must be Tensors"
                metadata.append({
                    "outer_key": outer_key,
                    "inner_key": inner_key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": tensor.numel(),
                })
        return metadata

    @staticmethod
    def _pack_tensors_to_buffer(
        package: dict, metadata: list, device: str
    ) -> torch.Tensor:
        """根据元数据，将数据包中的所有 Tensors 打包成一个连续的 GPU ByteTensor。"""
        buffer_list = []
        for meta in metadata:
            tensor = package[meta["outer_key"]][meta["inner_key"]]
            assert tensor.is_cuda, f"Tensor {meta['outer_key']}.{meta['inner_key']} must be on GPU for packing."
            # 转换为连续内存并视为 byte 类型
            contiguous_tensor = tensor.contiguous()
            num_bytes = contiguous_tensor.numel() * contiguous_tensor.element_size()
            byte_view_1d = contiguous_tensor.view(torch.uint8).flatten()
        
            # 从一维字节向量中准确切出所需部分，以防内存对齐等问题导致尾部有多余字节
            buffer_list.append(byte_view_1d.narrow(0, 0, num_bytes))
        
        if not buffer_list:
            return torch.empty(0, dtype=torch.uint8, device=device)
            
        return torch.cat(buffer_list).to(device)

    @staticmethod
    def _create_empty_buffer_from_metadata(metadata: list, device: str) -> torch.Tensor:
        """根据元数据，在接收端创建一个正确大小的空 GPU ByteTensor 以接收数据。"""
        total_bytes = 0
        for meta in metadata:
            dtype = getattr(torch, meta["dtype"])
            # 创建一个0维的临时 tensor 来获取 element_size，避免创建不必要的存储
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_bytes += meta["numel"] * element_size
        return torch.empty(total_bytes, dtype=torch.uint8, device=device)

    @staticmethod
    def _unpack_buffer_to_package(
        packed_buffer: torch.Tensor, metadata: list
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """将接收到的 ByteTensor 根据元数据解包，还原成原始的嵌套字典结构。"""
        reshaped = {}
        offset = 0
        for meta in metadata:
            dtype = getattr(torch, meta["dtype"])
            element_size = torch.tensor([], dtype=dtype).element_size()
            num_bytes = meta["numel"] * element_size

            # 从大 buffer 中切片，并确保不会意外修改原始 buffer
            byte_slice = packed_buffer.narrow(0, offset, num_bytes)
            
            # 将 byte 切片还原为原始 Tensor
            tensor = byte_slice.view(dtype).view(meta["shape"]).clone()

            if meta["outer_key"] not in reshaped:
                reshaped[meta["outer_key"]] = {}
            reshaped[meta["outer_key"]][meta["inner_key"]] = tensor
            offset += num_bytes
        return reshaped

    # --------------------------------------------------------------------------
    # 公共 API (Public APIs)
    # --------------------------------------------------------------------------

    @staticmethod
    def broadcast_nested_tensor_package_gpu(
        package: Dict[str, Dict[str, torch.Tensor]] | None,
        src: int,
        group: Any,
        device: str,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        🚀 [一对多] 高效广播一个嵌套的 Tensor 字典。
        发送方提供 package, 接收方提供 None。所有 rank 都将返回还原后的 package。
        """
        global_rank = dist.get_rank()
        is_sender = (global_rank == src)

        # 1. 广播元数据
        metadata_list = [TensorUtils._extract_metadata(package) if is_sender else None]
        dist.broadcast_object_list(metadata_list, src=src, group=group)
        metadata = metadata_list[0]

        # 2. 准备发送/接收的 buffer
        if is_sender:
            packed_buffer = TensorUtils._pack_tensors_to_buffer(package, metadata, device)
        else:
            packed_buffer = TensorUtils._create_empty_buffer_from_metadata(metadata, device)

        # 3. 通过高速后端 (NCCL) 广播核心数据
        dist.broadcast(packed_buffer, src=src, group=group)

        # 4. 所有 rank 解包数据并返回
        return TensorUtils._unpack_buffer_to_package(packed_buffer, metadata)

    @staticmethod
    def send_recv_nested_tensor_package_gpu(
        package: Dict[str, Dict[str, torch.Tensor]] | None,
        src: int,
        dst: int,
        device: str,
    ) -> Dict[str, Dict[str, torch.Tensor]] | None:
        """
        🛰️ [一对一] 高效点对点发送一个嵌套的 Tensor 字典。
        只有 src rank 提供 package，只有 dst rank 会返回还原后的 package，其他 rank 返回 None。
        """
        rank = dist.get_rank()

        if rank == src:
            # 1. 准备并发送元数据
            metadata = TensorUtils._extract_metadata(package)
            metadata_bytes = pickle.dumps(metadata)
            
            # a) 发送元数据的大小 (CPU Tensor)
            metadata_size = torch.tensor([len(metadata_bytes)], dtype=torch.long, device="cpu")
            dist.send(tensor=metadata_size, dst=dst)
            
            # b) 发送元数据本身 (CPU Tensor)
            metadata_tensor = torch.frombuffer(metadata_bytes, dtype=torch.uint8).to("cpu")
            dist.send(tensor=metadata_tensor, dst=dst)

            # 2. 准备并发送打包好的 Tensor 数据 (GPU Tensor)
            packed_buffer = TensorUtils._pack_tensors_to_buffer(package, metadata, device)
            dist.send(tensor=packed_buffer, dst=dst)
            return None # 发送方不返回数据

        elif rank == dst:
            # 1. 接收并解析元数据
            # a) 接收元数据大小
            metadata_size = torch.empty(1, dtype=torch.long, device="cpu")
            dist.recv(tensor=metadata_size, src=src)
            
            # b) 根据大小接收元数据本身
            metadata_tensor = torch.empty(metadata_size.item(), dtype=torch.uint8, device="cpu")
            dist.recv(tensor=metadata_tensor, src=src)
            metadata = pickle.loads(metadata_tensor.numpy().tobytes())

            # 2. 根据元数据准备空 buffer 并接收打包好的 Tensor 数据
            packed_buffer = TensorUtils._create_empty_buffer_from_metadata(metadata, device)
            dist.recv(tensor=packed_buffer, src=src)
            
            # 3. 解包并返回
            return TensorUtils._unpack_buffer_to_package(packed_buffer, metadata)
        
        else:
            # 其他无关 rank 直接返回
            return None