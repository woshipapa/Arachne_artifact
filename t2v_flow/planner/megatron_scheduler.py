def megatron_schedule(tasks, total_gpus):
    """
    最朴素 Megatron-LM 调度策略：

    - 所有 task 使用相同的 SP
    - SP = total_gpus // num_tasks
    - 每个 task 独占一组连续 GPU
    - VAE -> DiT 顺序执行
    - Gang scheduling, Start=0
    """

    num_tasks = len(tasks)
    if num_tasks == 0:
        return [], 0

    # --- 1. 计算 SP ---
    if total_gpus % num_tasks != 0:
        raise ValueError(
            f"Naive Megatron requires total_gpus % num_tasks == 0, "
            f"but got {total_gpus} GPUs and {num_tasks} tasks."
        )

    sp = total_gpus // num_tasks

    trace = []
    gpu_cursor = 0
    max_makespan = 0.0
    base_start_time = 0.0

    # --- 2. 逐 task 分配 ---
    for task in tasks:
        t_id = task["task_id"]
        dataset_id = task["dataset_id"]

        # --- 3. 查找该 task 在该 SP 下的 VAE / DiT 时间 ---
        vae_time = None
        dit_time = None

        for s, t in task["vae"]:
            if s == sp:
                vae_time = t
                break

        for s, t in task["dit"]:
            if s == sp:
                dit_time = t
                break

        if vae_time is None or dit_time is None:
            raise ValueError(
                f"Task {t_id} does not support SP={sp}. "
                f"VAE_time={vae_time}, DiT_time={dit_time}"
            )

        total_time = vae_time + dit_time
        max_makespan = max(max_makespan, total_time)

        assigned_gpus = list(range(gpu_cursor, gpu_cursor + sp))
        gpu_cursor += sp

        # --- 4. VAE ---
        trace.append([
            t_id,                    # Task_ID
            "VAE",                   # Stage
            sp,                      # GPUs_Count
            vae_time,                # Time
            base_start_time,         # Start
            base_start_time + vae_time,  # End
            dataset_id,              # DatasetID
            assigned_gpus            # GPU_List
        ])

        # --- 5. DiT ---
        trace.append([
            t_id,
            "DIT",
            sp,
            dit_time,
            base_start_time + vae_time,
            base_start_time + total_time,
            dataset_id,
            assigned_gpus
        ])

    return trace, max_makespan
