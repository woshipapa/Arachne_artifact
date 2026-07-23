import torch
from .pad import set_origin_length, set_target_length, pad_for_context_parallel, remove_pad_for_context_parallel

set_origin_length(5)
set_target_length(8)

x = torch.randn(2, 5)
x_padded = pad_for_context_parallel(x, dim=1)
print(x_padded.shape)  # torch.Size([2, 8])

x_restored = remove_pad_for_context_parallel(x_padded, dim=1)
print(x_restored.shape)  # torch.Size([2, 5])
