import os
import torch
import torch.distributed as dist
from my_utils import global_timer

# ----------------------------
# 预注册要 profile 的 stage
# ----------------------------
COMM_DOMAIN = "Communication"
global_timer.register_stage("send", color="red", domain_name=COMM_DOMAIN)
global_timer.register_stage("recv", color="blue", domain_name=COMM_DOMAIN)

# ----------------------------
# 初始化分布式
# ----------------------------
def init_process():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size

# ----------------------------
# 简单 P2P 测试
# ----------------------------
def simple_p2p_test(rank, world_size):
    device = torch.device("cuda", rank)
    tensor = torch.ones(4, device=device) * (rank + 1)

    # 两两 rank 配对: 0-1, 2-3, 4-5, 6-7
    pair_rank = rank ^ 1  # XOR 1 得到配对 rank
    if rank < pair_rank:
        # 小 rank 先 send 后 recv
        global_timer.start("send")
        dist.send(tensor, dst=pair_rank)
        global_timer.stop("send")

        recv_tensor = torch.zeros_like(tensor)
        global_timer.start("recv")
        dist.recv(recv_tensor, src=pair_rank)
        global_timer.stop("recv")
    else:
        # 大 rank 先 recv 后 send
        recv_tensor = torch.zeros_like(tensor)
        global_timer.start("recv")
        dist.recv(recv_tensor, src=pair_rank)
        global_timer.stop("recv")

        global_timer.start("send")
        dist.send(tensor, dst=pair_rank)
        global_timer.stop("send")

    print(f"[Rank {rank}] received tensor from Rank {pair_rank}: {recv_tensor}")

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    rank, world_size = init_process()
    simple_p2p_test(rank, world_size)
    dist.destroy_process_group()
