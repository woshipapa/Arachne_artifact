import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh,init_device_mesh

def init_process_group():
    # 初始化默认的分布式通信
    dist.init_process_group(
        backend='nccl',       # 单机多卡常用nccl
        init_method='env://', # torchrun会传环境变量
    )

def create_device_mesh():
    # 设备号列表 [0,1,2,3,4,5,6,7]
    devices = list(range(torch.cuda.device_count()))
    mesh_shape = (2, 4)  # 2行4列
    device_mesh = init_device_mesh(device_type="cuda",mesh_shape=mesh_shape,mesh_dim_names=["dp","sp"])
    return device_mesh

def main():
    # 本地rank
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    init_process_group()

    device_mesh = create_device_mesh()

    # 打印DeviceMesh的基本信息
    if dist.get_rank() == 0:
        # print(f"DeviceMesh Shape: {device_mesh.mesh_shape}")
        print(f"DeviceMesh Devices: {device_mesh.mesh}")

    # 每个进程打印自己在mesh里的坐标
    my_coords = [device_mesh.get_local_rank(mesh_dim=0),device_mesh.get_local_rank(mesh_dim=1)]
    print(f"Global Rank: {dist.get_rank()}, Local Coords in Mesh: {my_coords}, {device_mesh.get_coordinate()}")

    # # 测试dim group，举个例子：在第0维group上做个all_reduce
    # dim0_groups = device_mesh.get_dim_groups()[0]  # 第0维（列方向）

    # # 每个rank创建一个tensor，内容是rank id
    # x = torch.tensor([dist.get_rank()], dtype=torch.float32, device=f"cuda:{local_rank}")

    # # 做all_reduce，只在第0维的小组里
    # dist.all_reduce(x, group=dim0_groups)

    # print(f"Rank {dist.get_rank()} after all_reduce on dim-0 group, tensor: {x.item()}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
