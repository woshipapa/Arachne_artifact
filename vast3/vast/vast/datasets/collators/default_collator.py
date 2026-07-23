import numpy as np
import torch

from .. import utils

import torch
from collections import defaultdict

def create_tensors_from_shape_dict(
    shape_dict: dict, 
    dtype: torch.dtype = torch.float16, 
    device: str = "cuda"
) -> dict:
    """
    根据单个样本的形状字典，创建一组完整的 PyTorch Tensors。
    这是一个纯函数，不依赖任何全局状态。
    """
    if not shape_dict or not isinstance(shape_dict, dict):
        print("错误: 输入的 shape_dict 无效。")
        return None

    image_shape = shape_dict.get("images")
    if not image_shape or len(image_shape) < 5:
        print(f"错误: 'images' 形状数据缺失或不完整: {image_shape}")
        return None
        
    bs, num_frames, c, h, w = image_shape
    images = torch.zeros((bs, num_frames, c, h, w), dtype=dtype, device=device)
    prompt_embeds = torch.zeros((bs, 226, 4096), dtype=dtype, device=device)
    clip_text_embed = torch.zeros((bs, 768), dtype=dtype, device=device)
    first_ref_image = torch.zeros_like(images[:, :1])

    batch_part = {
        "images": images,
        "prompt_embeds": prompt_embeds,
        "clip_text_embed": clip_text_embed,
        "first_ref_image": first_ref_image,
    }
    return batch_part
class DefaultCollator:
    def __init__(self, is_equal=False, tensor_dtype = torch.float16):
        self.is_equal = is_equal
        self.tensor_dtype = tensor_dtype

    def __call__(self, batch):
        is_shape_batch = (
            isinstance(batch, list) and len(batch) > 0 and 
            isinstance(batch[0], dict) and 
            "shapes" in batch[0] and "metadata" in batch[0]
        )

        if is_shape_batch:
            # 如果是，执行我们为 ExpDataset 定制的 Tensor 创建逻辑
            # print(f'ShapeCollator logic triggered for batch of size {len(batch)}') # for debugging
                       # 根据你的澄清，这里的 batch 列表长度应该是 1
            # 为安全起见，可以加一个断言或警告
            if len(batch) != 1:
                print(
                    f"WARNING: Shape-based collator received a batch of size {len(batch)}, "
                    f"but was expected to handle size 1. Processing only the first element."
                )
            
            # 1. 从批次列表中获取唯一的那个样本
            sample = batch[0]
            
            # 2. 提取形状信息字典
            shape_dict = sample["shapes"]
            
 
            print(f"rank {torch.distributed.get_rank()} in DefaultCollator")
            tensor_batch = create_tensors_from_shape_dict(
                shape_dict, dtype=self.tensor_dtype, device=torch.cuda.current_device()
            )
            
            # 4. (可选) 将元数据也添加到最终的批次中
            if tensor_batch is not None:
                tensor_batch['metadata'] = sample['metadata']
            
            return tensor_batch

        
        # ==========================================================
        # 3. 如果不是特殊格式，则执行原来的默认逻辑
        # ==========================================================
        # print(f'DefaultCollator logic triggered for batch of size {len(batch)}') # for debugging
        batch_dict = dict()
        if isinstance(batch, list) and len(batch) > 0:
            for key in batch[0]:
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
             for key in batch:
                 batch_dict[key] = self._collate(batch[key])
        else:
            # 如果 batch 为空列表，直接返回空字典
            if isinstance(batch, list) and len(batch) == 0:
                return {}
            assert False, f"Unsupported batch type: {type(batch)}"
        return batch_dict

    def _collate(self, batch):
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                batch = utils.stack_data(batch, is_equal=self.is_equal)
            elif isinstance(batch[0], np.ndarray):
                batch = utils.stack_data(batch, is_equal=self.is_equal)
                batch = torch.from_numpy(batch)
            elif isinstance(batch[0], (np.bool_, np.number, np.object_)):
                batch = torch.as_tensor(batch)
            elif isinstance(batch[0], dict):
                batch = {
                    key: self._collate([d[key] for d in batch]) for key in batch[0]
                }
            elif isinstance(batch[0], (list, tuple)):
                batch = type(batch[0])([self._collate(d) for d in zip(*batch)])
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch
