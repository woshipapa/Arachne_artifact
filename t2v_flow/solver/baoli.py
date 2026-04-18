import itertools
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# 1. 数据和配置
# =============================================================================
N_gpus = 8
data_items = ['D_97', 'D_121', 'D_221', 'D_241']
modules = ['VAE', 'DiT']

proc_times = {
    'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}},
    'D_121': {'VAE': {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}},
    'D_221': {'VAE': {2: 52.9, 4: 26.64, 8: 16.5}, 'DiT': {2: 76.8, 4: 45.48, 8: 43.6}},
    'D_241': {'VAE': {2: 55.47, 4: 30.01, 8: 17.29}, 'DiT': {2: 95.63, 4: 55.54, 8: 49.28}}
}
sp_choices = {
    ('D_97', 'VAE'): [1, 2, 4, 8], ('D_97', 'DiT'): [1, 2, 4, 8],
    ('D_121', 'VAE'): [1, 2, 4, 8], ('D_121', 'DiT'): [1, 2, 4, 8],
    ('D_221', 'VAE'): [2, 4, 8],    ('D_221', 'DiT'): [2, 4, 8],
    ('D_241', 'VAE'): [2, 4, 8],    ('D_241', 'DiT'): [2, 4, 8]
}

# =============================================================================
# 2. 模拟器与绘图函数 (保持不变)
# =============================================================================

def simulate_plan_and_get_schedule(plan):
    current_time = 0.0
    available_gpus = N_gpus
    tasks_running = {}
    tasks_to_do = set()
    tasks_done = set()
    schedule_details = []
    
    # todo task 
    for item in data_items:
        tasks_to_do.add((item, 'VAE'))
        
    total_tasks = len(data_items) * len(modules)

    while len(tasks_done) < total_tasks:
        scheduled_in_this_step = False
        tasks_to_consider = sorted(list(tasks_to_do))
        
        for task_key in tasks_to_consider:
            if task_key in tasks_to_do:
                sp_needed = plan[task_key]
                if sp_needed <= available_gpus:
                    tasks_to_do.remove(task_key)
                    available_gpus -= sp_needed
                    duration = proc_times[task_key[0]][task_key[1]][sp_needed]
                    finish_time = current_time + duration
                    tasks_running[task_key] = finish_time
                    schedule_details.append({'name': task_key, 'sp': sp_needed, 'start': current_time, 'end': finish_time})
                    scheduled_in_this_step = True

        if not tasks_running:
            if tasks_to_do: return float('inf'), []
            else: break 

        if not scheduled_in_this_step or not tasks_to_do:
            next_event_time = min(tasks_running.values())
            if next_event_time > current_time:
                current_time = next_event_time
        
        finished_now = []
        for task_key, finish_time in list(tasks_running.items()):
            if finish_time <= current_time:
                finished_now.append(task_key)
        
        for task_key in finished_now:
            if task_key in tasks_running:
                del tasks_running[task_key]
                tasks_done.add(task_key)
                available_gpus += plan[task_key]
                if task_key[1] == 'VAE':
                    tasks_to_do.add((task_key[0], 'DiT'))
    
    final_makespan = max([task['end'] for task in schedule_details]) if schedule_details else 0
    return final_makespan, schedule_details

def assign_ranks_to_schedule(schedule, total_gpus):
    schedule.sort(key=lambda x: x['start'])
    gpu_availability_time = [0.0] * total_gpus
    plot_data = []
    
    for task in schedule:
        start_time = task['start']
        sp_needed = task['sp']
        available_ranks = [r for r, free_time in enumerate(gpu_availability_time) if free_time <= start_time]
        
        if len(available_ranks) < sp_needed:
            raise Exception("调度错误!")
            
        allocated_ranks = available_ranks[:sp_needed]
        
        for rank in allocated_ranks:
            gpu_availability_time[rank] = task['end']
            
        plot_data.append({'name': f"{task['name'][0]}_{task['name'][1]}", 'sp': task['sp'], 'start': task['start'], 'end': task['end'], 'ranks': allocated_ranks})
        
    return plot_data

def draw_gantt_chart(plot_data, total_gpus):
    fig, ax = plt.subplots(figsize=(20, 8))
    task_names = sorted(list(set([d['name'] for d in plot_data])))
    colors = plt.cm.get_cmap('tab20', len(task_names))
    color_map = {name: colors(i) for i, name in enumerate(task_names)}

    for task in plot_data:
        duration = task['end'] - task['start']
        for rank in task['ranks']:
            ax.add_patch(patches.Rectangle(xy=(task['start'], rank - 0.4), width=duration, height=0.8, facecolor=color_map[task['name']], edgecolor='black', linewidth=0.5))
            if duration > 5:
                 ax.text(task['start'] + duration / 2, rank, f"{task['name']}\nSP={task['sp']}", color='white', weight='bold', ha='center', va='center', fontsize=8)

    ax.set_yticks(range(total_gpus))
    ax.set_yticklabels([f'Rank {i}' for i in range(total_gpus)])
    ax.set_ylim(-0.5, total_gpus - 0.5)
    makespan = max([d['end'] for d in plot_data] or [0])
    ax.set_xlim(0, makespan * 1.05)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('GPU Ranks', fontsize=12)
    ax.set_title('Optimal Task Schedule Timeline (Gantt Chart)', fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('optimal_schedule_gantt_chart.png', dpi=300)
    print("\n✅ Optimal Gantt chart saved to 'optimal_schedule_gantt_chart.png'")
    plt.show()

# =============================================================================
# 3. 主程序: 查找最优解并绘图
# =============================================================================
if __name__ == "__main__":
    ordered_tasks = []
    for item in data_items:
        ordered_tasks.append((item, 'VAE'))
        ordered_tasks.append((item, 'DiT'))

    all_sp_options_for_tasks = [sp_choices[task] for task in ordered_tasks]
    all_plans_iterator = itertools.product(*all_sp_options_for_tasks)
    num_combinations = math.prod(len(options) for options in all_sp_options_for_tasks)

    print(f"--- Step 1: Starting brute-force search through {num_combinations} plans... ---")

    min_makespan = float('inf')
    best_plan = None
    best_schedule = None

    for sp_combination in tqdm(all_plans_iterator, total=num_combinations):
        current_plan = {task: sp for task, sp in zip(ordered_tasks, sp_combination)}
        makespan, schedule = simulate_plan_and_get_schedule(current_plan)
        if makespan < min_makespan:
            min_makespan = makespan
            best_plan = current_plan
            best_schedule = schedule

    print("\n--- Step 2: Brute-Force Search Finished ---")
    if best_plan:
        print(f"✅ Optimal Solution Found!")
        print(f"   Minimum Makespan: {min_makespan:.2f} seconds")
        
        print("\n--- Step 3: Generating Visualization for the Optimal Schedule... ---")
        print(f"best plan is {best_plan}, best_schedule = {best_schedule}")
        # 为找到的最优日程表分配具体GPU Rank
        final_plot_data = assign_ranks_to_schedule(best_schedule, N_gpus)
        
        # 绘制甘特图
        draw_gantt_chart(final_plot_data, N_gpus)
    else:
        print("⚠️ Could not find any valid solution.")