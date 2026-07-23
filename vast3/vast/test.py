# test_cuda.py
import torch
a = torch.rand((1000, 1000), device='cuda')
b = torch.matmul(a, a)
torch.cuda.synchronize()