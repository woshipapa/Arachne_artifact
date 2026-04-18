from itertools import permutations
from tqdm import tqdm
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

# 固定任务顺序
ordered_tasks = [
    ('D_97', 'VAE'),
    ('D_121', 'VAE'),
    ('D_241', 'VAE'),
    ('D_221', 'VAE'),
    ('D_97', 'DiT'),
    ('D_241', 'DiT'),
    ('D_221', 'DiT'),
    ('D_121', 'DiT'),
]

# 固定 SP 分配
plan = {
    ('D_97', 'VAE'): 1,
    ('D_121', 'VAE'): 8,
    ('D_241', 'VAE'): 4,
    ('D_221', 'VAE'): 2,
    ('D_97', 'DiT'): 4,
    ('D_241', 'DiT'): 4,
    ('D_221', 'DiT'): 4,
    ('D_121', 'DiT'): 1
}

# 遍历所有调度顺序，记录最优 makespan
min_makespan = float('inf')
best_schedule = None
best_order = None

for task_order in tqdm(permutations(ordered_tasks)):
    makespan, schedule = simulate_plan_and_get_schedule_with_order(plan, task_order)
    if makespan < min_makespan:
        min_makespan = makespan
        best_schedule = schedule
        best_order = task_order

print(f"\n✅ 最小makespan为 {min_makespan:.2f} 秒")
print(f"任务执行顺序为：")
for task in best_order:
    print(f"  {task[0]}_{task[1]}")

print(best_schedule)