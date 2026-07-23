import os
import torch

class BatchLogger:
    def __init__(self, log_path="batch_log.txt", overwrite=True):
        self.log_path = log_path
        if overwrite and os.path.exists(log_path):
            open(log_path, "w").close()  # 清空原文件内容

    def log(self, batch, prefix="Batch"):
        with open(self.log_path, "a") as f:
            f.write(f"{prefix}:\n")
            if isinstance(batch, torch.Tensor):
                f.write(str(batch.shape) + "\n")
                f.write(str(batch) + "\n")
            elif isinstance(batch, dict):
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        f.write(f"{key}: shape={val.shape}\n")
                        f.write(str(val) + "\n")
                    else:
                        f.write(f"{key}: {val}\n")
            elif isinstance(batch, list):
                for i, item in enumerate(batch):
                    f.write(f"Item {i}: {item}\n")
            else:
                f.write(str(batch) + "\n")
            f.write("=" * 50 + "\n")