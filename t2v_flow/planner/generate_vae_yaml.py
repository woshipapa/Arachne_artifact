import yaml
import argparse
import random
import sys

def generate_vae_tasks(total_gpus: int, sp_size: int, tag: str) -> list:
    """
    根据给定的GPU总数和序列并行大小，生成VAE任务列表。

    Args:
        total_gpus (int): GPU 的总数量，通常是 8。
        sp_size (int): 每个任务使用的 GPU 数量 (Sequence Parallelism size)。
        tag (str): 用于任务命名的标签, e.g., '720p'。

    Returns:
        list: 生成的任务字典列表。
    """
    if total_gpus % sp_size != 0:
        print(f"错误: GPU总数 ({total_gpus}) 必须能够被 sp_size ({sp_size}) 整除。", file=sys.stderr)
        sys.exit(1)

    tasks = []
    num_tasks = total_gpus // sp_size
    
    print(f"将在 {total_gpus} 个GPU上为 sp={sp_size} 生成 {num_tasks} 个 VAE 任务...")

    for i in range(num_tasks):
        # 1. 计算当前任务使用的 GPU 列表
        start_gpu = i * sp_size
        gpus = list(range(start_gpu, start_gpu + sp_size))

        # 2. 生成任务名称
        task_id = random.randint(10, 200)  # 仿照示例生成一个随机ID
        gpu_str = "g" + "".join(map(str, gpus))
        task_name = f"D_{task_id}_{tag}_VAE_{gpu_str}"

        # 3. 构建任务字典
        task = {
            'name': task_name,
            'gpus': gpus,
            'dependencies': [],  # VAE 任务没有依赖
            'args': {
                'task_type': 'VAE',
                'sp': sp_size
            }
        }
        tasks.append(task)
        
    return tasks

def main():
    """主函数，用于解析参数并生成YAML文件。"""
    parser = argparse.ArgumentParser(
        description="为 VAE 任务生成 YAML 配置文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sp",
        type=int,
        required=True,
        choices=[1, 2, 4, 8],
        help="序列并行大小 (Sequence Parallelism size)。每个任务将使用多少个GPU。"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=8,
        help="可用的GPU总数。默认为 8。"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="720p",
        help="用于任务命名的标签。默认为 '720p'。"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vae_tasks.yaml",
        help="输出的YAML文件名。默认为 'vae_tasks.yaml'。"
    )

    args = parser.parse_args()

    # 生成任务列表
    vae_tasks = generate_vae_tasks(args.gpus, args.sp, args.tag)
    
    # 将任务列表包装在顶层 'tasks' 键下
    output_data = {'tasks': vae_tasks}

    # 写入YAML文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            # sort_keys=False 保持字典原有顺序
            # default_flow_style=False 使用块状格式，更美观
            yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, indent=2)
        print(f"\n成功！配置文件已保存至: {args.output}")
    except Exception as e:
        print(f"\n错误: 无法写入文件 {args.output}。原因: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()