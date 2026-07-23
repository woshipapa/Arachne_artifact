import yaml
import argparse
import random
import sys

def generate_vae_tasks(total_gpus: int, sp_size: int, tag: str) -> list:
    """

    Args:

    Returns:
    """
    if total_gpus % sp_size != 0:
        print(f"Error: the total GPU count ({total_gpus}) must be divisible by sp_size ({sp_size}).", file=sys.stderr)
        sys.exit(1)

    tasks = []
    num_tasks = total_gpus // sp_size
    
    print(f"Generating {num_tasks} VAE tasks for sp={sp_size} across {total_gpus} GPUs...")

    for i in range(num_tasks):
        start_gpu = i * sp_size
        gpus = list(range(start_gpu, start_gpu + sp_size))

        task_id = random.randint(10, 200)
        gpu_str = "g" + "".join(map(str, gpus))
        task_name = f"D_{task_id}_{tag}_VAE_{gpu_str}"

        task = {
            'name': task_name,
            'gpus': gpus,
            'dependencies': [],
            'args': {
                'task_type': 'VAE',
                'sp': sp_size
            }
        }
        tasks.append(task)
        
    return tasks

def main():
    parser = argparse.ArgumentParser(
        description="Generate the YAML configuration for VAE tasks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sp",
        type=int,
        required=True,
        choices=[1, 2, 4, 8],
        help="Sequence-parallel size: how many GPUs each task uses."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=8,
        help="Total number of available GPUs. Default 8."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="720p",
        help="Label used to name the tasks. Default '720p'."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vae_tasks.yaml",
        help="Output YAML filename. Default 'vae_tasks.yaml'."
    )

    args = parser.parse_args()

    vae_tasks = generate_vae_tasks(args.gpus, args.sp, args.tag)
    
    output_data = {'tasks': vae_tasks}

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, indent=2)
        print(f"\nDone. Configuration written to: {args.output}")
    except Exception as e:
        print(f"\nError: could not write {args.output}. Reason: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()