import torch
from BatchLogger import BatchLogger
class SimpleCollator:

    def __call__(self, batch):
        """
        batch: List[torch.Tensor]，每个 Tensor shape 相同，例如 (T, C, H, W)
        返回: 一个打包后的 batch Tensor,shape = (B, T, C, H, W)
        """
        # logger = BatchLogger("debug_batch.txt")
        # logger.log(batch=batch)
        # 
        ret = torch.utils.data.default_collate(batch)
        # print(f'ret is {len(ret["prompt"])}')
        return ret