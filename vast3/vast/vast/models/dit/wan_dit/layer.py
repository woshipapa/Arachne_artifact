
import torch
# from megatron.core import mpu
from accelerate.sp_utils.comm import *

class GateWithGradReduce(torch.autograd.Function ):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual
    
    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate 
        gate_grad = torch.sum((x_grad * residual), dim=1, keepdim=True)
        parallel_manager = ParallelManagerFactory.get_instance()
        torch.distributed.all_reduce(gate_grad, group=parallel_manager.get_seq_parallel_group())
        return x_grad, gate_grad, r_grad


class ModulateWithCPGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale):
        ctx.save_for_backward(x, scale)
        return (x * (1 + scale) + shift)
    
    @staticmethod 
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_grad = grad_output * (1 + scale) 
        scale_grad = torch.sum((grad_output * x), dim=1, keepdim=True)
        parallel_manager = ParallelManagerFactory.get_instance()
        torch.distributed.all_reduce(scale_grad, group=parallel_manager.get_seq_parallel_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=parallel_manager.get_seq_parallel_group())
        return x_grad, shift_grad, scale_grad

