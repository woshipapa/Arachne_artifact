import os
import torch
import torch.distributed as dist
from my_utils import global_timer

# ----------------------------
# ----------------------------
COMM_DOMAIN = "Communication"
global_timer.register_stage("send", color="red", domain_name=COMM_DOMAIN)
global_timer.register_stage("recv", color="blue", domain_name=COMM_DOMAIN)

# ----------------------------
# ----------------------------
def init_process():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size

# ----------------------------
# ----------------------------
def simple_p2p_test(rank, world_size):
    device = torch.device("cuda", rank)
    tensor = torch.ones(4, device=device) * (rank + 1)

    pair_rank = rank ^ 1
    if rank < pair_rank:
        global_timer.start("send")
        dist.send(tensor, dst=pair_rank)
        global_timer.stop("send")

        recv_tensor = torch.zeros_like(tensor)
        global_timer.start("recv")
        dist.recv(recv_tensor, src=pair_rank)
        global_timer.stop("recv")
    else:
        recv_tensor = torch.zeros_like(tensor)
        global_timer.start("recv")
        dist.recv(recv_tensor, src=pair_rank)
        global_timer.stop("recv")

        global_timer.start("send")
        dist.send(tensor, dst=pair_rank)
        global_timer.stop("send")

    print(f"[Rank {rank}] received tensor from Rank {pair_rank}: {recv_tensor}")

# ----------------------------
# ----------------------------
if __name__ == "__main__":
    rank, world_size = init_process()
    simple_p2p_test(rank, world_size)
    dist.destroy_process_group()
