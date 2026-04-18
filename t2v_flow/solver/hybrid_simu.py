import math

# =============================================================================
# 1. 数据和配置
# =============================================================================
# 整个系统的总GPU数
N_gpus_total = 8

# 任务数据
data_items = ['D_97', 'D_121', 'D_221', 'D_241']
modules = ['VAE', 'DiT']
proc_times = {
    'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}},
    'D_121': {'VAE': {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}},
    'D_221': {'VAE': {2: 52.9, 4: 26.64, 8: 16.5}, 'DiT': {2: 76.8, 4: 45.48, 8: 43.6}},
    'D_241': {'VAE': {2: 55.47, 4: 30.01, 8: 17.29}, 'DiT': {2: 95.63, 4: 55.54, 8: 49.28}}
}

# =============================================================================
# 2. 针对“混合调度”定制的模拟器
# =============================================================================
def simulate_hybrid_plan(plan):
    """
    这个模拟器专门用于您设想的混合调度策略。
    plan: 定义了每个任务具体使用的SP值。
    """
    schedule_details = []
    
    # --- 1. 模拟独立的 1-GPU 分区 ---
    
    # D_97 的流水线
    sp_97_vae = plan[('D_97', 'VAE')]
    sp_97_dit = plan[('D_97', 'DiT')]
    if sp_97_vae > 1 or sp_97_dit > 1:
        raise ValueError("D_97 的SP不能超过1")
    
    time_97_vae = proc_times['D_97']['VAE'][sp_97_vae]
    time_97_dit = proc_times['D_97']['DiT'][sp_97_dit]
    makespan_97 = time_97_vae + time_97_dit
    schedule_details.append({'name': ('D_97', 'VAE'), 'sp': sp_97_vae, 'start': 0.0, 'end': time_97_vae})
    schedule_details.append({'name': ('D_97', 'DiT'), 'sp': sp_97_dit, 'start': time_97_vae, 'end': makespan_97})

    # D_121 的流水线
    sp_121_vae = plan[('D_121', 'VAE')]
    sp_121_dit = plan[('D_121', 'DiT')]
    if sp_121_vae > 1 or sp_121_dit > 1:
        raise ValueError("D_121 的SP不能超过1")

    time_121_vae = proc_times['D_121']['VAE'][sp_121_vae]
    time_121_dit = proc_times['D_121']['DiT'][sp_121_dit]
    makespan_121 = time_121_vae + time_121_dit
    schedule_details.append({'name': ('D_121', 'VAE'), 'sp': sp_121_vae, 'start': 0.0, 'end': time_121_vae})
    schedule_details.append({'name': ('D_121', 'DiT'), 'sp': sp_121_dit, 'start': time_121_vae, 'end': makespan_121})
    
    # --- 2. 模拟 6-GPU 动态共享池 ---
    
    # VAE 阶段的SP分配
    sp_241_vae = plan[('D_241', 'VAE')]
    sp_221_vae = plan[('D_221', 'VAE')]
    if sp_241_vae + sp_221_vae > 6:
        raise ValueError("241和221的VAE阶段SP总和不能超过6")

    # DiT 阶段的SP分配
    sp_241_dit = plan[('D_241', 'DiT')]
    sp_221_dit = plan[('D_221', 'DiT')]
    if sp_241_dit + sp_221_dit > 6:
        raise ValueError("241和221的DiT阶段SP总和不能超过6")

    # VAE 阶段并行开始
    time_241_vae = proc_times['D_241']['VAE'][sp_241_vae]
    time_221_vae = proc_times['D_221']['VAE'][sp_221_vae]
    
    end_241_vae = 0.0 + time_241_vae
    end_221_vae = 0.0 + time_221_vae

    schedule_details.append({'name': ('D_241', 'VAE'), 'sp': sp_241_vae, 'start': 0.0, 'end': end_241_vae})
    schedule_details.append({'name': ('D_221', 'VAE'), 'sp': sp_221_vae, 'start': 0.0, 'end': end_221_vae})
    
    # DiT 阶段的调度依赖于VAE的完成时间
    # D_241_DiT 可以在 D_241_VAE 完成后立即开始
    start_241_dit = end_241_vae
    time_241_dit = proc_times['D_241']['DiT'][sp_241_dit]
    end_241_dit = start_241_dit + time_241_dit
    
    # D_221_DiT 可以在 D_221_VAE 完成后立即开始
    start_221_dit = end_221_vae
    time_221_dit = proc_times['D_221']['DiT'][sp_221_dit]
    end_221_dit = start_221_dit + time_221_dit
    
    # **关键**：DiT阶段也需要GPU，它们之间可能也存在资源竞争
    # 我们需要模拟一个简单的事件队列
    pool_schedule = []
    # 如果两个DiT任务的时间段重叠，检查它们的SP总和
    overlap_start = max(start_241_dit, start_221_dit)
    overlap_end = min(end_241_dit, end_221_dit)

    if overlap_start < overlap_end: # 确认有重叠
        if sp_241_dit + sp_221_dit > 6:
            # 有冲突，需要串行化。让先就绪的先开始
            if start_241_dit < start_221_dit:
                # 241_DiT 先跑，221_DiT 必须等241_DiT跑完才能开始
                start_221_dit = end_241_dit
                end_221_dit = start_221_dit + time_221_dit
            else:
                # 221_DiT 先跑
                start_241_dit = end_221_dit
                end_241_dit = start_241_dit + time_241_dit

    schedule_details.append({'name': ('D_241', 'DiT'), 'sp': sp_241_dit, 'start': start_241_dit, 'end': end_241_dit})
    schedule_details.append({'name': ('D_221', 'DiT'), 'sp': sp_221_dit, 'start': start_221_dit, 'end': end_221_dit})

    makespan_pool = max(end_241_dit, end_221_dit)

    # --- 3. 计算总耗时 ---
    overall_makespan = max(makespan_97, makespan_121, makespan_pool)
    
    return overall_makespan, schedule_details

# =============================================================================
# 3. 主程序: 定义并查询您的混合调度方案
# =============================================================================
if __name__ == "__main__":
    
    # --- !! 定义您设想的混合调度方案的SP值 !! ---
    my_hybrid_plan = {
        # 独立分区
        ('D_97', 'VAE'): 1,
        ('D_97', 'DiT'): 1,
        ('D_121', 'VAE'): 1,
        ('D_121', 'DiT'): 1,
        
        # 动态共享池
        # VAE 阶段: 4 + 2 = 6 GPUs
        ('D_241', 'VAE'): 4,
        ('D_221', 'VAE'): 2,
        # DiT 阶段: 2 + 4 = 6 GPUs
        ('D_241', 'DiT'): 2,
        ('D_221', 'DiT'): 4,
    }

    print(f"--- Querying for Hybrid Dynamic Plan ---")
    
    # --- 调用模拟器并获取结果 ---
    makespan, schedule = simulate_hybrid_plan(my_hybrid_plan)
    
    # --- 打印结果 ---
    if makespan == float('inf'):
        print("\n❌ This plan is invalid.")
    else:
        print(f"\n✅ Plan analysis complete!")
        print(f"   Calculated Makespan: {makespan:.2f} seconds")
        print("\n--- Detailed Schedule ---")
        schedule.sort(key=lambda x: (x['start'], x['name'][0], x['name'][1]))
        for task in schedule:
            task_name_str = f"({task['name'][0]}, {task['name'][1]})"
            print(f"Task {task_name_str:<18}: Use SP={task['sp']:<2} | Starts at {task['start']:>6.2f} | Ends at {task['end']:>6.2f}")