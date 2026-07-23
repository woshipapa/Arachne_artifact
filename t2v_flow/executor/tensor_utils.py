import pickle
import time
from typing import Dict, Any

import torch
import torch.distributed as dist
from loguru import logger


class TensorUtils:

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(package: Dict[str, Dict[str, torch.Tensor]]) -> list:
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
        buffer_list = []
        for meta in metadata:
            tensor = package[meta["outer_key"]][meta["inner_key"]]
            assert tensor.is_cuda, f"Tensor {meta['outer_key']}.{meta['inner_key']} must be on GPU for packing."
            contiguous_tensor = tensor.contiguous()
            num_bytes = contiguous_tensor.numel() * contiguous_tensor.element_size()
            byte_view_1d = contiguous_tensor.view(torch.uint8).flatten()
        
            buffer_list.append(byte_view_1d.narrow(0, 0, num_bytes))
        
        if not buffer_list:
            return torch.empty(0, dtype=torch.uint8, device=device)
            
        return torch.cat(buffer_list).to(device)

    @staticmethod
    def _create_empty_buffer_from_metadata(metadata: list, device: str) -> torch.Tensor:
        total_bytes = 0
        for meta in metadata:
            dtype = getattr(torch, meta["dtype"])
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_bytes += meta["numel"] * element_size
        return torch.empty(total_bytes, dtype=torch.uint8, device=device)

    @staticmethod
    def _unpack_buffer_to_package(
        packed_buffer: torch.Tensor, metadata: list
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        reshaped = {}
        offset = 0
        for meta in metadata:
            dtype = getattr(torch, meta["dtype"])
            element_size = torch.tensor([], dtype=dtype).element_size()
            num_bytes = meta["numel"] * element_size

            byte_slice = packed_buffer.narrow(0, offset, num_bytes)
            
            tensor = byte_slice.view(dtype).view(meta["shape"]).clone()

            if meta["outer_key"] not in reshaped:
                reshaped[meta["outer_key"]] = {}
            reshaped[meta["outer_key"]][meta["inner_key"]] = tensor
            offset += num_bytes
        return reshaped

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @staticmethod
    def broadcast_nested_tensor_package_gpu(
        package: Dict[str, Dict[str, torch.Tensor]] | None,
        src: int,
        group: Any,
        device: str,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        global_rank = dist.get_rank()
        is_sender = (global_rank == src)

        metadata_list = [TensorUtils._extract_metadata(package) if is_sender else None]
        dist.broadcast_object_list(metadata_list, src=src, group=group)
        metadata = metadata_list[0]

        if is_sender:
            packed_buffer = TensorUtils._pack_tensors_to_buffer(package, metadata, device)
        else:
            packed_buffer = TensorUtils._create_empty_buffer_from_metadata(metadata, device)

        dist.broadcast(packed_buffer, src=src, group=group)

        return TensorUtils._unpack_buffer_to_package(packed_buffer, metadata)

    @staticmethod
    def send_recv_nested_tensor_package_gpu(
        package: Dict[str, Dict[str, torch.Tensor]] | None,
        src: int,
        dst: int,
        device: str,
    ) -> Dict[str, Dict[str, torch.Tensor]] | None:
        rank = dist.get_rank()

        if rank == src:
            metadata = TensorUtils._extract_metadata(package)
            metadata_bytes = pickle.dumps(metadata)
            
            metadata_size = torch.tensor([len(metadata_bytes)], dtype=torch.long, device="cpu")
            dist.send(tensor=metadata_size, dst=dst)
            
            metadata_tensor = torch.frombuffer(metadata_bytes, dtype=torch.uint8).to("cpu")
            dist.send(tensor=metadata_tensor, dst=dst)

            packed_buffer = TensorUtils._pack_tensors_to_buffer(package, metadata, device)
            dist.send(tensor=packed_buffer, dst=dst)
            return None

        elif rank == dst:
            metadata_size = torch.empty(1, dtype=torch.long, device="cpu")
            dist.recv(tensor=metadata_size, src=src)
            
            metadata_tensor = torch.empty(metadata_size.item(), dtype=torch.uint8, device="cpu")
            dist.recv(tensor=metadata_tensor, src=src)
            metadata = pickle.loads(metadata_tensor.numpy().tobytes())

            packed_buffer = TensorUtils._create_empty_buffer_from_metadata(metadata, device)
            dist.recv(tensor=packed_buffer, src=src)
            
            return TensorUtils._unpack_buffer_to_package(packed_buffer, metadata)
        
        else:
            return None