import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse, get_default_args
# import nvtx

from my_utils import global_timer
COMPUTE_DOMAIN = "Compute"
COMM_DOMAIN = "Communication"

# # 为 ring_flash_attn_forward 预注册所有静态 stage
global_timer.register_stage("send_recv_kv",          color="red",       domain_name=COMM_DOMAIN)
global_timer.register_stage("send_recv_kv_blocking",          color="red",       domain_name=COMM_DOMAIN)
global_timer.register_stage("flash_attn",            color="green",     domain_name=COMPUTE_DOMAIN)
global_timer.register_stage("update_out_lse",        color="darkgreen", domain_name=COMPUTE_DOMAIN)
global_timer.register_stage("wait_kv",               color="darkred",   domain_name=COMM_DOMAIN)
global_timer.register_stage("Finalize",              color="purple",    domain_name=COMPUTE_DOMAIN)
global_timer.register_stage("send_k",                  color="orange",    domain_name=COMM_DOMAIN)
global_timer.register_stage("recv_k",                  color="darkorange",domain_name=COMM_DOMAIN)
global_timer.register_stage("send_v",                  color="lightcoral",domain_name=COMM_DOMAIN)
global_timer.register_stage("recv_v",                  color="indianred", domain_name=COMM_DOMAIN)
global_timer.register_stage("send_dk_dv", color="blue", domain_name=COMM_DOMAIN)
global_timer.register_stage("wait_dk_dv", color="darkblue", domain_name=COMM_DOMAIN)

# # global_timer.disable_cuda_time()

# def ring_flash_attn_forward(
#     process_group,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     softmax_scale,
#     dropout_p=0,
#     causal=True,
#     window_size=(-1, -1),
#     alibi_slopes=None,
#     deterministic=False,
# ):
#     global_timer.start("ring_flash_attn_forward")
#     try:
#         comm = RingComm(process_group)
#         out, lse = None, None

#         for step in range(comm.world_size):
#             step_label = f"Step {step}/{comm.world_size} (Rank {comm.rank})"
#             global_timer.start(step_label, color="yellow")
            
#             try:
#                 # 1. 首先，使用当前的 k 和 v 进行计算
#                 if not causal or step <= comm.rank:
#                     global_timer.start("flash_attn")
#                     # ... (原有的 flash_attn 计算逻辑保持不变)
#                     params = get_default_args(_flash_attn_forward).copy()
#                     params.update({ "q": q, "k": k, "v": v, "dropout_p": dropout_p, "softmax_scale": softmax_scale, "causal": causal and step == 0, "alibi_slopes": alibi_slopes, "return_softmax": True and dropout_p > 0, "window_size": window_size, })
#                     outputs = _flash_attn_forward(**params)
#                     if len(outputs) == 8:
#                         block_out, _, _, _, _, block_lse, _, _ = outputs
#                     else:
#                         block_out, block_lse, _, _ = outputs
#                     global_timer.stop("flash_attn")
                    
#                     global_timer.start("update_out_lse")
#                     out, lse = update_out_and_lse(out, lse, block_out, block_lse)
#                     global_timer.stop("update_out_lse")
                
#                 # 2. 然后，执行无死锁的阻塞式 send 和 recv
#                 if step + 1 != comm.world_size:
#                     global_timer.start("send_recv_kv_blocking")
                    
#                     next_k = torch.empty_like(k)
#                     next_v = torch.empty_like(v)

#                     # === 修改开始: 为每个 send/recv 添加 timer ===
#                     if comm.rank == 0:
#                         # Rank 0: Send then Recv
#                         global_timer.start("send_k")
#                         dist.send(k, comm.send_rank)
#                         global_timer.stop("send_k")

#                         global_timer.start("recv_k")
#                         dist.recv(next_k, comm.recv_rank)
#                         global_timer.stop("recv_k")

#                         global_timer.start("send_v")
#                         dist.send(v, comm.send_rank)
#                         global_timer.stop("send_v")

#                         global_timer.start("recv_v")
#                         dist.recv(next_v, comm.recv_rank)
#                         global_timer.stop("recv_v")
#                     else:
#                         # Other Ranks: Recv then Send
#                         global_timer.start("recv_k")
#                         dist.recv(next_k, comm.recv_rank)
#                         global_timer.stop("recv_k")

#                         global_timer.start("send_k")
#                         dist.send(k, comm.send_rank)
#                         global_timer.stop("send_k")

#                         global_timer.start("recv_v")
#                         dist.recv(next_v, comm.recv_rank)
#                         global_timer.stop("recv_v")

#                         global_timer.start("send_v")
#                         dist.send(v, comm.send_rank)
#                         global_timer.stop("send_v")
#                     # === 修改结束 ===
                    
#                     # 更新 k, v 以便下一次循环使用
#                     k, v = next_k, next_v
                    
#                     global_timer.stop("send_recv_kv_blocking")

#             finally:
#                 global_timer.stop(step_label)
        
#         # ... (函数剩余部分保持不变)
#         global_timer.start("Finalize")
#         out = out.to(q.dtype)
#         lse = lse.squeeze(dim=-1).transpose(1, 2)
#         global_timer.stop("Finalize")
        
#         return out, lse
#     finally:
#         global_timer.stop("ring_flash_attn_forward")

# async
def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # 直接调用全局 timer 实例
    global_timer.start("ring_flash_attn_forward")
    try:
        comm = RingComm(process_group)
        out, lse = None, None
        next_k, next_v = None, None

        for step in range(comm.world_size):
            step_label = f"Step {step}/{comm.world_size} (Rank {comm.rank})"
            
            # 由于 step_label 是动态的，它会走探索式路径
            global_timer.start(step_label, color="yellow")
            try:
                if step + 1 != comm.world_size:
                    # 调用预注册的 stage
                    global_timer.start("send_recv_kv")
                    next_k, next_v = comm.send_recv_kv(k, v)
                    global_timer.stop("send_recv_kv")

                if not causal or step <= comm.rank:
                    global_timer.start("flash_attn")
                    # ... (原有的 flash_attn 计算逻辑)
                    params = get_default_args(_flash_attn_forward).copy()
                    params.update({ "q": q, "k": k, "v": v, "dropout_p": dropout_p, "softmax_scale": softmax_scale, "causal": causal and step == 0, "alibi_slopes": alibi_slopes, "return_softmax": True and dropout_p > 0, "window_size": window_size, })
                    outputs = _flash_attn_forward(**params)
                    if len(outputs) == 8:
                        block_out, _, _, _, _, block_lse, _, _ = outputs
                    else:
                        block_out, block_lse, _, _ = outputs
                    global_timer.stop("flash_attn")
                    
                    global_timer.start("update_out_lse")
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                    global_timer.stop("update_out_lse")
                
                if step + 1 != comm.world_size:
                    global_timer.start("wait_kv")
                    comm.wait()
                    k, v = next_k, next_v
                    global_timer.stop("wait_kv")
            finally:
                global_timer.stop(step_label)
        
        global_timer.start("Finalize")
        out = out.to(q.dtype)
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        global_timer.stop("Finalize")
        
        return out, lse
    finally:
        global_timer.stop("ring_flash_attn_forward")



# def ring_flash_attn_forward(
#     process_group,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     softmax_scale,
#     dropout_p=0,
#     causal=True,
#     window_size=(-1, -1),
#     alibi_slopes=None,
#     deterministic=False,
# ):
#     comm = RingComm(process_group)

    # out = None
    # lse = None

    # next_k, next_v = None, None

    # for step in range(comm.world_size):
    #     if step + 1 != comm.world_size:
    #         next_k, next_v = comm.send_recv_kv(k, v)

    #     if not causal or step <= comm.rank:
    #         params = get_default_args(_flash_attn_forward).copy()
    #         params.update(
    #             {
    #                 "q": q,
    #                 "k": k,
    #                 "v": v,
    #                 "dropout_p": dropout_p,
    #                 "softmax_scale": softmax_scale,
    #                 "causal": causal and step == 0,
    #                 "alibi_slopes": alibi_slopes,
    #                 "return_softmax": True and dropout_p > 0,
    #             }
    #         )
    #         if "window_size" in params:
    #             params.update({"window_size": window_size})
    #         else:
    #             params.update(
    #                 {
    #                     "window_size_left": window_size[0],
    #                     "window_size_right": window_size[1],
    #                 }
    #             )
    #         outputs = _flash_attn_forward(**params)
    #         if len(outputs) == 8:
    #             block_out, _, _, _, _, block_lse, _, _ = outputs
    #         else:
    #             assert len(outputs) == 4
    #             block_out, block_lse, _, _ = outputs

    #         from my_utils import get_global_logger
    #         logger = get_global_logger()
    #         # logger.info(f"outputs shape is {block_out.shape}")
    #         out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    #     if step + 1 != comm.world_size:
    #         comm.wait()
    #         k, v = next_k, next_v

    # out = out.to(q.dtype)
    # lse = lse.squeeze(dim=-1).transpose(1, 2)
    # return out, lse



def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_k, next_v = None, None

    global_timer.start("ring_flash_attn_backward")
    try:
        for step in range(kv_comm.world_size):
            step_label = f"Step {step}/{kv_comm.world_size} (Rank {kv_comm.rank})"
            global_timer.start(step_label, color="yellow")
            try:
                # --------------------------------------------------
                # KV 通信
                # --------------------------------------------------
                if step + 1 != kv_comm.world_size:
                    global_timer.start("send_recv_kv")
                    next_k, next_v = kv_comm.send_recv_kv(k, v)
                    global_timer.stop("send_recv_kv")

                # --------------------------------------------------
                # 计算 dq/dk/dv
                # --------------------------------------------------
                if step <= kv_comm.rank or not causal:
                    bwd_causal = causal and step == 0
                    params = get_default_args(_flash_attn_backward).copy()
                    params.update(
                        {
                            "dout": dout,
                            "q": q,
                            "k": k,
                            "v": v,
                            "out": out,
                            "softmax_lse": softmax_lse,
                            "dq": block_dq_buffer,
                            "dk": block_dk_buffer,
                            "dv": block_dv_buffer,
                            "dropout_p": dropout_p,
                            "softmax_scale": softmax_scale,
                            "causal": bwd_causal,
                            "alibi_slopes": alibi_slopes,
                            "deterministic": deterministic,
                        }
                    )
                    if "window_size" in params:
                        params.update({"window_size": window_size})
                    else:
                        params.update(
                            {"window_size_left": window_size[0], "window_size_right": window_size[1]}
                        )

                    global_timer.start("flash_attn")
                    _flash_attn_backward(**params)
                    global_timer.stop("flash_attn")

                    # 累加梯度
                    if dq is None:
                        dq = block_dq_buffer.to(torch.float32)
                        dk = block_dk_buffer.to(torch.float32)
                        dv = block_dv_buffer.to(torch.float32)
                    else:
                        dq += block_dq_buffer

                        global_timer.start("wait_kv")
                        d_kv_comm.wait()
                        global_timer.stop("wait_kv")

                        dk = block_dk_buffer + next_dk
                        dv = block_dv_buffer + next_dv
                elif step != 0:
                    global_timer.start("wait_dk_dv")
                    d_kv_comm.wait()
                    global_timer.stop("wait_dk_dv")
                    dk, dv = next_dk, next_dv

                # --------------------------------------------------
                # 更新 k/v
                # --------------------------------------------------
                if step + 1 != kv_comm.world_size:
                    global_timer.start("wait_kv")
                    kv_comm.wait()
                    k, v = next_k, next_v
                    global_timer.stop("wait_kv")

                # --------------------------------------------------
                # 发送 dk/dv 给下一个 rank
                # --------------------------------------------------
                global_timer.start("send_dk_dv")
                next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
                global_timer.stop("send_dk_dv")


            finally:
                global_timer.stop(step_label)

        global_timer.start("Finalize")
        dq = dq.to(torch.bfloat16)
        dk = next_dk.to(q.dtype)
        dv = next_dv.to(q.dtype)
        global_timer.stop("Finalize")

        return dq, dk, dv
    finally:
        global_timer.stop("ring_flash_attn_backward")


# def ring_flash_attn_backward(
#     process_group,
#     dout,
#     q,
#     k,
#     v,
#     out,
#     softmax_lse,
#     softmax_scale,
#     dropout_p=0,
#     causal=True,
#     window_size=(-1, -1),
#     alibi_slopes=None,
#     deterministic=False,
# ):
#     kv_comm = RingComm(process_group)
#     d_kv_comm = RingComm(process_group)
#     dq, dk, dv = None, None, None
#     next_dk, next_dv = None, None

#     block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
#     block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
#     block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

#     next_dk, next_dv = None, None
#     next_k, next_v = None, None

#     for step in range(kv_comm.world_size):
#         if step + 1 != kv_comm.world_size:
#             next_k, next_v = kv_comm.send_recv_kv(k, v)

#         if step <= kv_comm.rank or not causal:
#             bwd_causal = causal and step == 0
#             params = get_default_args(_flash_attn_backward).copy()
#             params.update(
#                 {
#                     "dout": dout,
#                     "q": q,
#                     "k": k,
#                     "v": v,
#                     "out": out,
#                     "softmax_lse": softmax_lse,
#                     "dq": block_dq_buffer,
#                     "dk": block_dk_buffer,
#                     "dv": block_dv_buffer,
#                     "dropout_p": dropout_p,
#                     "softmax_scale": softmax_scale,
#                     "causal": bwd_causal,
#                     "alibi_slopes": alibi_slopes,
#                     "deterministic": deterministic,
#                 }
#             )
#             if "window_size" in params:
#                 params.update({"window_size": window_size})
#             else:
#                 params.update(
#                     {
#                         "window_size_left": window_size[0],
#                         "window_size_right": window_size[1],
#                     }
#                 )
#             _flash_attn_backward(**params)

#             if dq is None:
#                 dq = block_dq_buffer.to(torch.float32)
#                 dk = block_dk_buffer.to(torch.float32)
#                 dv = block_dv_buffer.to(torch.float32)
#             else:
#                 dq += block_dq_buffer
#                 d_kv_comm.wait()
#                 dk = block_dk_buffer + next_dk
#                 dv = block_dv_buffer + next_dv
#         elif step != 0:
#             d_kv_comm.wait()
#             dk, dv = next_dk, next_dv

#         if step + 1 != kv_comm.world_size:
#             kv_comm.wait()
#             k, v = next_k, next_v

#         next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

#     d_kv_comm.wait()

#     return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
