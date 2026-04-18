import math

# =============================================================================
# 1. 数据和配置
# =============================================================================
# -- A. 资源静态划分 --
gpu_partitions = {
    'D_241': 4,
    'D_221': 2,
    'D_121': 1,
    'D_97': 1,
}
# 确认总数
total_gpus_in_partitions = sum(gpu_partitions.values())

# -- B. 任务数据 --
data_items = ['D_97', 'D_121', 'D_221', 'D_241']
modules = ['VAE', 'DiT']
proc_times = {
    'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}},
    'D_121': {'VAE': {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}},
    'D_221': {'VAE': {2: 52.9, 4: 26.64, 8: 16.5}, 'DiT': {2: 76.8, 4: 45.48, 8: 43.6}},
    'D_241': {'VAE': {2: 55.47, 4: 30.01, 8: 17.29}, 'DiT': {2: 95.63, 4: 55.54, 8: 49.28}}
}

# =============================================================================
# 2. 针对“静态划分”修改的模拟器
# =============================================================================
def simulate_partitioned_plan(plan):
    """
    这个模拟器理解GPU是被静态划分的。
    """
    # 每个分区的可用GPU是独立的
    available_gpus = dict(gpu_partitions)
    
    # 每个分区的调度也是独立的，我们为每个分区模拟一条时间线
    partition_schedules = {item: [] for item in data_items}
    partition_final_times = {item: 0.0 for item in data_items}

    for item in data_items:
        # 获取该分区的任务和SP选择
        vae_task = (item, 'VAE')
        dit_task = (item, 'DiT')
        
        vae_sp = plan.get(vae_task)
        dit_sp = plan.get(dit_task)
        
        partition_gpu_limit = gpu_partitions[item]
        
        # 检查SP选择是否合法
        if vae_sp is None or dit_sp is None:
            print(f"❌ 错误: 方案中未定义 {item} 的SP。")
            return float('inf'), {}
        if vae_sp > partition_gpu_limit or dit_sp > partition_gpu_limit:
            print(f"❌ 错误: {item} 的SP选择 ({vae_sp}, {dit_sp}) 超过其分区限制 {partition_gpu_limit}。")
            return float('inf'), {}
            
        # --- 在这个分区内，任务是串行的 (VAE -> DiT) ---
        # 1. VAE 任务
        vae_start_time = 0.0
        vae_duration = proc_times[item]['VAE'][vae_sp]
        vae_end_time = vae_start_time + vae_duration
        partition_schedules[item].append({
            'name': vae_task, 'sp': vae_sp, 'start': vae_start_time, 'end': vae_end_time
        })
        
        # 2. DiT 任务
        dit_start_time = vae_end_time # VAE一结束，DiT立刻开始
        dit_duration = proc_times[item]['DiT'][dit_sp]
        dit_end_time = dit_start_time + dit_duration
        partition_schedules[item].append({
            'name': dit_task, 'sp': dit_sp, 'start': dit_start_time, 'end': dit_end_time
        })
        
        # 记录这个分区的最终完成时间
        partition_final_times[item] = dit_end_time

    # 总的Makespan由最慢的那个分区决定
    overall_makespan = max(partition_final_times.values())
    
    # 整合所有分区的调度详情用于显示
    full_schedule = []
    for partition_schedule in partition_schedules.values():
        full_schedule.extend(partition_schedule)
        
    return overall_makespan, full_schedule

# =============================================================================
# 3. 主程序: 定义并查询您的静态划分方案
# =============================================================================
if __name__ == "__main__":
    
    # --- !! 在这里定义您想查询的静态划分方案 !! ---
    # 注意：每个任务的SP不能超过其所在分区的GPU数量
    my_partitioned_plan = {
        # D_241 分区 (4卡): VAE和DiT最多可用SP=4
        ('D_241', 'VAE'): 4,
        ('D_241', 'DiT'): 4,
        
        # D_221 分区 (2卡): VAE和DiT最多可用SP=2
        ('D_221', 'VAE'): 2,
        ('D_221', 'DiT'): 2,

        # D_121 分区 (1卡): VAE和DiT只能用SP=1
        ('D_121', 'VAE'): 1,
        ('D_121', 'DiT'): 1,
        
        # D_97 分区 (1卡): VAE和DiT只能用SP=1
        ('D_97', 'VAE'): 1,
        ('D_97', 'DiT'): 1
    }

    print(f"--- Querying for Static Partitioning Plan ---")
    print(f"Total GPUs allocated: {total_gpus_in_partitions}")
    
    # --- 调用模拟器并获取结果 ---
    makespan, schedule = simulate_partitioned_plan(my_partitioned_plan)
    
    # --- 打印结果 ---
    if makespan == float('inf'):
        print("\n❌ This plan is invalid for the given partitioning.")
    else:
        print(f"\n✅ Plan analysis complete!")
        print(f"   Calculated Makespan: {makespan:.2f} seconds")
        print("\n--- Detailed Schedule by Partition ---")
        schedule.sort(key=lambda x: (x['name'][0], x['start']))
        for task in schedule:
            task_name_str = f"({task['name'][0]}, {task['name'][1]})"
            print(f"Task {task_name_str:<18}: Use SP={task['sp']:<2} | Starts at {task['start']:>6.2f} | Ends at {task['end']:>6.2f}")