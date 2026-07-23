from typing import Dict, Tuple

import torch
import torch.distributed as dist


def _all_gather_variable_1d(
    local_tensor: torch.Tensor,
    group,
) -> Tuple[list, list]:
    device = local_tensor.device
    world_size = dist.get_world_size(group=group)

    local_size = torch.tensor([local_tensor.numel()], dtype=torch.long, device=device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size, group=group)
    sizes = [int(item.item()) for item in gathered_sizes]
    max_size = max(sizes) if sizes else 0

    if max_size == 0:
        return [local_tensor.new_empty((0,)) for _ in range(world_size)], sizes

    if local_tensor.numel() < max_size:
        pad = torch.zeros(
            max_size - local_tensor.numel(),
            dtype=local_tensor.dtype,
            device=device,
        )
        padded = torch.cat([local_tensor, pad], dim=0)
    else:
        padded = local_tensor

    gathered = [torch.empty(max_size, dtype=local_tensor.dtype, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    sliced = [
        gathered[idx][: sizes[idx]] if sizes[idx] > 0 else local_tensor.new_empty((0,))
        for idx in range(world_size)
    ]
    return sliced, sizes


def gather_tiled_tensors_with_all_gather(
    local_tiles: Dict[Tuple[int, int], torch.Tensor],
    group,
    device: torch.device,
    fallback_dtype: torch.dtype,
    max_ndim: int = 8,
) -> Dict[Tuple[int, int], torch.Tensor]:
    if not dist.is_initialized():
        return {
            (int(i), int(j)): tensor.to(device)
            for (i, j), tensor in local_tiles.items()
        }

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return {
            (int(i), int(j)): tensor.to(device)
            for (i, j), tensor in local_tiles.items()
        }

    meta_width = 3 + max_ndim + 2
    sorted_items = sorted(local_tiles.items(), key=lambda item: (item[0][0], item[0][1]))

    local_rows = []
    flat_chunks = []
    running_offset = 0
    data_dtype = fallback_dtype
    if sorted_items:
        data_dtype = sorted_items[0][1].dtype

    for (coord_i, coord_j), tile in sorted_items:
        if tile.device != device:
            tile = tile.to(device, non_blocking=True)
        tile = tile.contiguous()
        tile_numel = int(tile.numel())
        tile_ndim = int(tile.dim())
        if tile_ndim > max_ndim:
            raise RuntimeError(
                f"Tile ndim {tile_ndim} exceeds max_ndim {max_ndim}."
            )

        row = [int(coord_i), int(coord_j), tile_ndim]
        row.extend(int(dim) for dim in tile.shape)
        row.extend([0] * (max_ndim - tile_ndim))
        row.extend([running_offset, tile_numel])
        local_rows.append(row)
        flat_chunks.append(tile.view(-1))
        running_offset += tile_numel

    if local_rows:
        local_meta = torch.tensor(local_rows, dtype=torch.long, device=device)
        local_data = torch.cat(flat_chunks, dim=0)
    else:
        local_meta = torch.empty((0, meta_width), dtype=torch.long, device=device)
        local_data = torch.empty((0,), dtype=data_dtype, device=device)

    gathered_meta_flat, _ = _all_gather_variable_1d(local_meta.view(-1), group=group)
    gathered_data_flat, _ = _all_gather_variable_1d(local_data, group=group)

    all_tiles: Dict[Tuple[int, int], torch.Tensor] = {}
    for rank_meta_flat, rank_data_flat in zip(gathered_meta_flat, gathered_data_flat):
        if rank_meta_flat.numel() == 0:
            continue
        if rank_meta_flat.numel() % meta_width != 0:
            raise RuntimeError(
                f"Invalid gathered tile metadata length: {rank_meta_flat.numel()} "
                f"(meta_width={meta_width})."
            )
        rank_meta = rank_meta_flat.view(-1, meta_width)
        for row in rank_meta:
            coord_i = int(row[0].item())
            coord_j = int(row[1].item())
            tile_ndim = int(row[2].item())
            shape = [int(row[3 + dim_idx].item()) for dim_idx in range(tile_ndim)]
            offset = int(row[3 + max_ndim].item())
            numel = int(row[4 + max_ndim].item())
            tile_flat = rank_data_flat[offset : offset + numel]
            all_tiles[(coord_i, coord_j)] = tile_flat.view(shape)

    return all_tiles
