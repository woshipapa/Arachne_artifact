import torch
import torch.cuda.nvtx as nvtx

nvtx.range_push("test_matmul")
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
nvtx.range_pop()
