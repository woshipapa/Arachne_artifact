import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from utils import log, set_seed
import torch.cuda.nvtx as nvtx

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    from my_utils import GlobalLogger, global_timer

    global_logger = GlobalLogger()
    global_logger.setup("debug_logs/", rank=rank, world_size=world_size)

    logger = global_logger.get_logger()

    global_timer.set_logger(logger_instance=logger)
    batch_size = 1
    seqlen = 12000
    nheads = 24
    d = 128
    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    logger.info("==== normal flash_attn forward ====")
    with nvtx.range("flash_attn_forward"):
        out, lse, _ = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    fn = ring_flash_attn_qkvpacked_func
    logger.info("==== ring flash attn ====")
    with nvtx.range("ring_flash_attn_forward"):
        ring_out, ring_lse, _ = fn(
            local_qkv,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)
    logger.info("==== normal flash_attn backward ====")
    with nvtx.range("flash_attn_backward"):
        out.backward(dout)
    dqkv = qkv.grad
    local_dqkv = dqkv.chunk(world_size, dim=1)[rank]


    logger.info("==== ring_flash_attn backward ====")
    with nvtx.range("ring_flash_attn_backward"):
        ring_out.backward(local_dout)
    ring_dqkv = local_qkv.grad

    log("local_dqkv", local_dqkv)
    log("dq diff", local_dqkv[:, 0] - ring_dqkv[:, 0])
    log("dk diff", local_dqkv[:, 1] - ring_dqkv[:, 1])
    log("dv diff", local_dqkv[:, 2] - ring_dqkv[:, 2])

    global_timer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        flash_attn_qkvpacked_func = torch.compile(flash_attn_qkvpacked_func)
        ring_flash_attn_qkvpacked_func = torch.compile(ring_flash_attn_qkvpacked_func)
    main()
