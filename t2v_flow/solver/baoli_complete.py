import itertools
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations
import json
# =============================================================================
# 1. 数据和配置
# =============================================================================
N_gpus = 16
data_items = ['D_101', 'D2_45', 'D_113', 'D1_113']
modules = ['VAE', 'DiT']

proc_times = {
    'D_101': {'VAE': {1: 7.387, 2: 3.87, 4: 2.27,5: 1.95, 8: 1.48, 10: 1.17 ,}, 'DiT': {4: 30.15, 5: 22.67, 8: 13.98, 10: 12.60}},
    'D2_45': {'VAE': {1: 5.71, 2: 2.96, 4: 1.77, 5: 1.51, 8: 1.13, 10: 0.88 ,}, 'DiT': {4: 16.38, 5: 11.74, 8:7.45, 10: 7.22}},
    'D_113': {'VAE': {1: 8.29, 2: 4.29, 4: 2.53, 5: 2.19, 8: 1.67, 10: 1.32}, 'DiT': {4: 35.69, 5: 28.47, 8: 17.26, 10: 14.96}},
    'D1_113': {'VAE': {1: 8.29, 2: 4.29, 4: 2.53, 5: 2.19, 8: 1.67, 10: 1.32}, 'DiT': {4: 35.69, 5: 28.47, 8: 17.26, 10: 14.96}}
}
sp_choices = {
    ('D_101', 'VAE'): [1, 2, 4, 5, 8, 10], ('D_101', 'DiT'): [ 4, 5, 8, 10],
    ('D2_45', 'VAE'): [1, 2, 4, 5, 8, 10], ('D2_45', 'DiT'): [4, 5, 8, 10],
    ('D_113', 'VAE'): [1, 2, 4, 5, 8, 10],    ('D_113', 'DiT'): [4, 5, 8, 10],
    ('D1_113', 'VAE'): [1, 2, 4, 5, 8, 10],    ('D1_113', 'DiT'): [4, 5, 8, 10]
}

# =============================================================================
# 2. 模拟器与绘图函数 (保持不变)
# =============================================================================


from collections import defaultdict
import networkx as nx

def generate_valid_orders(tasks):
    # 构造依赖图
    G = nx.DiGraph()
    G.add_nodes_from(tasks)
    
    # 添加约束：VAE_i → DiT_i
    for item in data_items:
        G.add_edge((item, 'VAE'), (item, 'DiT'))

    # 枚举所有拓扑排序
    return list(nx.all_topological_sorts(G))


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

def simulate_plan_and_get_schedule_with_order(plan, task_order):
    current_time = 0.0
    available_gpus = N_gpus
    tasks_running = {}
    tasks_to_do = set()
    tasks_done = set()
    schedule_details = []

    # 初始化VAE任务
    for item in data_items:
        tasks_to_do.add((item, 'VAE'))

    total_tasks = len(data_items) * 2

    while len(tasks_done) < total_tasks:
        scheduled_in_this_step = False

        for task_key in task_order:
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
            if tasks_to_do:
                return float('inf'), []
            else:
                break

        if not scheduled_in_this_step or not tasks_to_do:
            next_event_time = min(tasks_running.values())
            current_time = next_event_time

        finished_now = []
        for task_key, finish_time in list(tasks_running.items()):
            if finish_time <= current_time:
                finished_now.append(task_key)

        for task_key in finished_now:
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


def save_data_to_json(data, filename="schedule_data.json"):
    """
    将Python字典数据保存为JSON文件。

    Args:
        data (dict): 需要保存的数据。
        filename (str): 保存的文件名。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 使用 json.dump() 将数据写入文件
            # indent=4 让文件内容格式化，更易读
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到文件: {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

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
    # plt.show()

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
        # 🔁 尝试当前SP组合下的所有任务调度顺序
        valid_orders = generate_valid_orders(ordered_tasks)

        ordered_tasks_list = permutations(ordered_tasks)
    
        for task_order in valid_orders:
            makespan, schedule = simulate_plan_and_get_schedule_with_order(current_plan, task_order)
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
        save_data_to_json(final_plot_data)
        # 绘制甘特图
        draw_gantt_chart(final_plot_data, N_gpus)
    else:
        print("⚠️ Could not find any valid solution.")