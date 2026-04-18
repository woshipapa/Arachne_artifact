import heapq
import pandas as pd
from typing import List, Tuple, Dict
import time
import matplotlib.pyplot as plt
import pandas as pd
import random
import yaml
import os

SP_STRATEGIES = [1, 2, 4, 8]

DATASET_EXEC_TIMES_20 = {
    0: {
        "name": "D_97",
        "vae": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "dit": {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94},
    },
    1: {
        "name": "D_121",
        "vae": {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03},
        "dit": {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83},
    },
    2: {
        "name": "D_221",
        "vae": {2: 52.9, 4: 26.64, 8: 16.5},
        "dit": {2: 74.8, 4: 36.5, 8: 18.36},
    },
    3: {
        "name": "D_241",
        "vae": {2: 52.26, 4: 30.01, 8: 17.29},
        "dit": {2: 88.61, 4: 42.7, 8: 21.58},
    },
}


DATASET_EXEC_TIMES = {
    0: {
        "name": "D_97",
        "VAE": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "DiT": {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94},
    },
    1: {
        "name": "D_121",
        "VAE": {1: 52.76, 2: 28.04, 4: 15.7, 8: 10.03},
        "DiT": {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83},
    },
    2: {
        "name": "D_221",
        "VAE": {2: 52.9, 4: 31.60, 8: 16.5},
        "DiT": {2: 74.8, 4: 40.35, 8: 18.36},
    },
    3: {
        "name": "D_241",
        "VAE": {2: 63.77, 4: 30.01, 8: 17.29},
        "DiT": {2: 92.94, 4: 42.7, 8: 21.58},
    },
}
DATASET_EXEC_TIMES_120 = {
    0: {
        "name": "D_000",
        "vae": {1: 37.36, 2: 20.02, 4: 10.73, 8: 5.75},
        "dit": {1: 69.59, 2: 37.29, 4: 19.99, 8: 10.71},
    },
    1: {
        "name": "D_001",
        "vae": {1: 50.31, 2: 26.96, 4: 14.45, 8: 7.74},
        "dit": {1: 78.99, 2: 42.33, 4: 22.68, 8: 12.16},
    },
    2: {
        "name": "D_002",
        "vae": {1: 34.43, 2: 18.45, 4: 9.89, 8: 5.3},
        "dit": {1: 91.41, 2: 48.99, 4: 26.25, 8: 14.07},
    },
    3: {
        "name": "D_003",
        "vae": {1: 46.96, 2: 25.17, 4: 13.49, 8: 7.23},
        "dit": {1: 94.89, 2: 50.85, 4: 27.25, 8: 14.6},
    },
}


def generate_fake_dataset_exec_times(
    num_datasets: int = 4,
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """
    自动生成符合 DATASET_EXEC_TIMES 格式的伪数据。
    时间满足：SP 数越大，时间越短（模拟并行加速）。
    VAE 时间范围：[25, 120]，DIT 时间范围：[25, 241]
    """
    SP_STRATEGIES = [1, 2, 4, 8]
    fake_exec_times = {}

    for i in range(num_datasets):
        name = f"D_{i:03d}"

        # 先生成 base 时间（SP=1）
        base_vae = random.uniform(20, 60)  # 起点在 80~120
        base_dit = random.uniform(40, 100)  # 起点在 120~241

        # 每增加一倍卡数，执行时间减少一定比例
        vae = {}
        dit = {}
        for idx, sp in enumerate(SP_STRATEGIES):
            vae[sp] = round(base_vae / (sp**0.9), 2)
            dit[sp] = round(base_dit / (sp**0.9), 2)

        fake_exec_times[i] = {
            "name": name,
            "vae": vae,
            "dit": dit,
        }

    return fake_exec_times


# def simulate_tasks_variable(num_tasks: int) -> List[Dict[str, List[Tuple[int, float]]]]:
#     tasks = []
#     for i in range(num_tasks):
#         dataset_id = i % 4
#         fixed_times = DATASET_EXEC_TIMES[dataset_id]
#         dit_opts = list(fixed_times["dit"].items())
#         vae_opts = list(fixed_times["vae"].items())
#         tasks.append({"dit": dit_opts, "vae": vae_opts})
#     return tasks


def simulate_tasks_variable(num_tasks: int) -> List[Dict[str, List[Tuple[int, float]]]]:
    tasks = []
    for i in range(num_tasks):
        dataset_id = i % 4
        fixed_times = DATASET_EXEC_TIMES[dataset_id]
        name = fixed_times["name"]
        dit_opts = list(fixed_times["DiT"].items())
        vae_opts = list(fixed_times["VAE"].items())
        tasks.append(
            {
                "dit": dit_opts,
                "vae": vae_opts,
                "dataset_id": name,  # ✅ 加上 dataset 编号
            }
        )
    return tasks


def simulate_tasks_fixed(num_tasks: int) -> List[Dict[str, List[Tuple[int, float]]]]:
    fixed_times = {
        "vae": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "dit": {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17},
    }
    tasks = []
    for _ in range(num_tasks):
        dit_opts = [(k, fixed_times["dit"][k]) for k in SP_STRATEGIES]
        vae_opts = [(k, fixed_times["vae"][k]) for k in SP_STRATEGIES]
        tasks.append({"dit": dit_opts, "vae": vae_opts})
    return tasks


class State:
    def __init__(self, finished, gpu_timeline, total_time, trace, stage_end_time):
        self.finished = finished
        self.gpu_timeline = gpu_timeline
        self.total_time = total_time
        self.trace = trace
        self.stage_end_time = stage_end_time

    def __lt__(self, other):
        return self.total_time < other.total_time

def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    assert isinstance(gpu_timeline, list) and k > 0
    sorted_gpus = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
    selected = sorted_gpus[:k]
    start = max(gpu_timeline[i] for i in selected) if selected else float('inf')
    return selected, start




# def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
#     #print("find earliest gpus")
#     sorted_gpus = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
#     selected = sorted_gpus[:k]
#     start = max(gpu_timeline[i] for i in selected)
#     return selected, start


def find_buddy_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    """
    Buddy system GPU 分配：在 GPU timeline 中查找 k=2^n 个连续且按对齐块分配的 GPU。
    :param gpu_timeline: 每个 GPU 当前的可用时间（浮点数）
    :param k: 需要的 GPU 数量（必须为 2 的幂次）
    :return: 选中的 GPU 列表，以及其最早开始时间
    """
    assert (k & (k - 1)) == 0, "k 必须是 2 的幂次"

    n = len(gpu_timeline)
    block_size = k
    for i in range(0, n, block_size):
        if i + block_size > n:
            continue
        block = list(range(i, i + block_size))
        start_time = max(gpu_timeline[g] for g in block)
        yield block, start_time


# def heuristic(finished, tasks, total_gpus):
#     remain = 0
#     # print(f"{tasks}")
#     for i in range(len(tasks)):
#         if f"{i}_DIT" not in finished:
#             remain += min(t[1] for t in tasks[i]["dit"])
#         if f"{i}_VAE" not in finished:
#             remain += min(t[1] for t in tasks[i]["vae"])
#     return remain / total_gpus
def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
    """
    贪心调度：
      - VAE 阶段：使用 find_earliest_gpus，直接取 (gpus, start) 二元组
      - DiT 阶段：使用 buddy 分配（find_buddy_gpus），从候选块中选最早可行的，并满足 VAE->DiT 先后约束
    返回:
      trace: List[(task_id, stage, sp, dur, start, end, dataset_id, gpu_list)]
      makespan: float
    """
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time: Dict[str, float] = {}

    for i in range(len(tasks)):
        dataset_id = tasks[i]["dataset_id"]

        # --- VAE 阶段 ---
        selected = False
        # 按时长升序尝试不同 sp 挡位
        for k, t in sorted(tasks[i]["vae"], key=lambda x: x[1]):
            # ✅ 注意：find_earliest_gpus 返回 (gpus, start) 二元组，不能当候选列表再 min()
            gpus, start = find_earliest_gpus(gpu_timeline, k)
            end = start + t
            for g in gpus:
                gpu_timeline[g] = end
            trace.append((i, "VAE", k, t, start, end, dataset_id, gpus))
            stage_end_time[f"{i}_VAE"] = end
            selected = True
            break

        if not selected:
            raise RuntimeError(f"VAE阶段无法为任务 {i} 分配可用的 {k} 个 GPU")

        # --- DiT 阶段 ---
        selected = False
        for k, t in sorted(tasks[i]["dit"], key=lambda x: x[1]):
            # buddy 分配会 yield 多个候选块，这里组列表后选最早可行的
            candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
            if not candidate_blocks:
                continue
            # 先看 GPU 就绪时间，再与 VAE 完成时间取最大作为真正 start
            gpus, tentative_start = min(candidate_blocks, key=lambda x: x[1])
            start = max(tentative_start, stage_end_time[f"{i}_VAE"])
            end = start + t
            for g in gpus:
                gpu_timeline[g] = end
            trace.append((i, "DIT", k, t, start, end, dataset_id, gpus))
            stage_end_time[f"{i}_DIT"] = end
            selected = True
            break

        if not selected:
            raise RuntimeError(f"DIT阶段无法为任务 {i} 分配可用的 {k} 个 GPU")

    makespan = max(gpu_timeline) if gpu_timeline else 0.0
    return trace, makespan


# def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
#     gpu_timeline = [0.0] * total_gpus
#     trace = []
#     stage_end_time = {}

#     for i in range(len(tasks)):
#         dataset_id = tasks[i]["dataset_id"]

#         # --- VAE 阶段 ---
#         selected = False
#         for k, t in sorted(tasks[i]["vae"], key=lambda x: x[1]):
#             candidate_blocks = list(find_earliest_gpus(gpu_timeline, k))
#             if not candidate_blocks:
#                 continue
#             gpus, start = min(candidate_blocks, key=lambda x: x[1])
#             end = start + t
#             for g in gpus:
#                 gpu_timeline[g] = end
#             trace.append((i, "VAE", k, t, start, end, dataset_id, gpus))
#             stage_end_time[f"{i}_VAE"] = end
#             selected = True
#             break
#         if not selected:
#             raise RuntimeError(f"VAE阶段无法为任务 {i} 分配 {k} 个 GPU")

#         # --- DIT 阶段 ---
#         selected = False
#         for k, t in sorted(tasks[i]["dit"], key=lambda x: x[1]):
#             candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
#             if not candidate_blocks:
#                 continue
#             gpus, tentative_start = min(candidate_blocks, key=lambda x: x[1])
#             start = max(tentative_start, stage_end_time[f"{i}_VAE"])
#             end = start + t
#             for g in gpus:
#                 gpu_timeline[g] = end
#             trace.append((i, "DIT", k, t, start, end, dataset_id, gpus))
#             stage_end_time[f"{i}_DIT"] = end
#             selected = True
#             break
#         if not selected:
#             raise RuntimeError(f"DIT阶段无法为任务 {i} 分配 {k} 个 GPU")

#     makespan = max(gpu_timeline)
#     return trace, makespan


# def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
#     gpu_timeline = [0.0] * total_gpus
#     finished = set()
#     trace = []
#     stage_end_time = {}

#     for i in range(len(tasks)):
#         dataset_id = tasks[i]["dataset_id"]
#         # --- 1. VAE 阶段 ---
#         vae_opts = sorted(tasks[i]["vae"], key=lambda x: x[1])  # 按时间升序
#         for k, t in vae_opts:
#             candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
#             if not candidate_blocks:
#                 continue
#             gpus, start = min(candidate_blocks, key=lambda x: x[1])
#             end = start + t
#             for g in gpus:
#                 gpu_timeline[g] = end
#             trace.append((i, "VAE", k, t, start, end, dataset_id, gpus))
#             stage_end_time[f"{i}_VAE"] = end
#             break

#         # --- 2. DIT 阶段 ---
#         dit_opts = sorted(tasks[i]["dit"], key=lambda x: x[1])
#         for k, t in dit_opts:
#             candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
#             if not candidate_blocks:
#                 continue
#             vae_end = stage_end_time[f"{i}_VAE"]
#             gpus, start = min(candidate_blocks, key=lambda x: max(x[1], vae_end))
#             start = max(start, vae_end)
#             end = start + t
#             for g in gpus:
#                 gpu_timeline[g] = end
#             trace.append((i, "DIT", k, t, start, end, dataset_id, gpus))
#             break

#     makespan = max(gpu_timeline)
#     return trace, makespan


def a_star_schedule(tasks: List[Dict], total_gpus: int = 32):
    initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    heap = [(initial.total_time + heuristic(set(), tasks, total_gpus), initial)]

    while heap:
        _, state = heapq.heappop(heap)
        if len(state.finished) == 2 * len(tasks):
            return state

        #print("for loop")
        for i in range(len(tasks)):
            dit_key, vae_key = f"{i}_DIT", f"{i}_VAE"
            #print(f"dit_key: {dit_key}, vae_key: {vae_key}")
            # VAE 阶段未完成
            if vae_key not in state.finished:
                for k, t in tasks[i]["vae"]:
                    if total_gpus >= k:
                        gpus, start = find_earliest_gpus(state.gpu_timeline, k)

                        # candidate_blocks = list(
                        #     find_earliest_gpus(state.gpu_timeline, k)
                        # )
                        # if not candidate_blocks:
                        #     continue
                        # print(f"{candidate_blocks}")
                        # gpus, start = min(candidate_blocks, key=lambda x: x[1])

                        end = start + t
                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end
                        new_finished = state.finished | {vae_key}
                        new_trace = state.trace + [
                            (i, "VAE", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[vae_key] = end
                        new_state = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )
                        heapq.heappush(
                            heap,
                            (
                                new_state.total_time
                                + heuristic(new_finished, tasks, total_gpus),
                                new_state,
                            ),
                        )

            # VAE 完成但 DIT 未完成
            elif dit_key not in state.finished:
                for k, t in tasks[i]["dit"]:
                    if total_gpus >= k:
                        gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
                        vae_end = state.stage_end_time[vae_key]
                        start = max(gpu_ready, vae_end)
                        end = start + t
                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end
                        new_finished = state.finished | {dit_key}
                        new_trace = state.trace + [
                            (i, "DIT", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[dit_key] = end
                        new_state = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )
                        heapq.heappush(
                            heap,
                            (
                                new_state.total_time
                                + heuristic(new_finished, tasks, total_gpus),
                                new_state,
                            ),
                        )


# def a_star_schedule(tasks: List[Dict], total_gpus: int = 32):
#     initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
#     heap = [(initial.total_time + heuristic(set(), tasks, total_gpus), initial)]

#     while heap:
#         _, state = heapq.heappop(heap)
#         if len(state.finished) == 2 * len(tasks):
#             return state

#         for i in range(len(tasks)):
#             dit_key, vae_key = f"{i}_DIT", f"{i}_VAE"

#             # 如果该任务的 VAE 阶段尚未完成
#             if vae_key not in state.finished:
#                 for k, t in tasks[i]["vae"]:
#                     if total_gpus >= k:
#                         gpus, start = find_earliest_gpus(state.gpu_timeline, k)
#                         end = start + t
#                         new_gpu = state.gpu_timeline[:]
#                         for g in gpus:
#                             new_gpu[g] = end
#                         new_finished = state.finished | {vae_key}
#                         new_trace = state.trace + [
#                             (i, "VAE", k, t, start, end, tasks[i]["dataset_id"])
#                         ]
#                         new_end_times = state.stage_end_time.copy()
#                         new_end_times[vae_key] = end
#                         new_state = State(
#                             new_finished,
#                             new_gpu,
#                             max(end, state.total_time),
#                             new_trace,
#                             new_end_times,
#                         )
#                         heapq.heappush(
#                             heap,
#                             (
#                                 new_state.total_time
#                                 + heuristic(new_finished, tasks, total_gpus),
#                                 new_state,
#                             ),
#                         )

#             # 如果 VAE 完成了，但 DIT 尚未完成
#             elif dit_key not in state.finished:
#                 for k, t in tasks[i]["dit"]:
#                     if total_gpus >= k:
#                         gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
#                         vae_end = state.stage_end_time[vae_key]
#                         start = max(gpu_ready, vae_end)
#                         end = start + t
#                         new_gpu = state.gpu_timeline[:]
#                         for g in gpus:
#                             new_gpu[g] = end
#                         new_finished = state.finished | {dit_key}
#                         new_trace = state.trace + [
#                             (i, "DIT", k, t, start, end, tasks[i]["dataset_id"])
#                         ]
#                         new_end_times = state.stage_end_time.copy()
#                         new_end_times[dit_key] = end
#                         new_state = State(
#                             new_finished,
#                             new_gpu,
#                             max(end, state.total_time),
#                             new_trace,
#                             new_end_times,
#                         )
#                         heapq.heappush(
#                             heap,
#                             (
#                                 new_state.total_time
#                                 + heuristic(new_finished, tasks, total_gpus),
#                                 new_state,
#                             ),
#                         )


# def a_star_schedule(tasks: List[Dict], total_gpus: int = 32):
#     initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
#     heap = [(initial.total_time + heuristic(set(), tasks, total_gpus), initial)]

#     while heap:
#         _, state = heapq.heappop(heap)
#         if len(state.finished) == 2 * len(tasks):
#             return state

#         for i in range(len(tasks)):
#             dit_key, vae_key = f"{i}_DIT", f"{i}_VAE"

#             if vae_key not in state.finished:
#                 for k, t in tasks[i]["vae"]:
#                     if total_gpus >= k:
#                         gpus, start = find_earliest_gpus(state.gpu_timeline, k)
#                         end = start + t
#                         new_gpu = state.gpu_timeline[:]
#                         for g in gpus:
#                             new_gpu[g] = end
#                         new_finished = state.finished | {vae_key}
#                         new_trace = state.trace + [(i, "VAE", k, t, start, end)]
#                         new_end_times = state.stage_end_time.copy()
#                         new_end_times[vae_key] = end
#                         new_state = State(
#                             new_finished,
#                             new_gpu,
#                             max(end, state.total_time),
#                             new_trace,
#                             new_end_times,
#                         )
#                         heapq.heappush(
#                             heap,
#                             (
#                                 new_state.total_time
#                                 + heuristic(new_finished, tasks, total_gpus),
#                                 new_state,
#                             ),
#                         )
#                 break

#             if vae_key in state.finished and dit_key not in state.finished:
#                 for k, t in tasks[i]["dit"]:
#                     if total_gpus >= k:
#                         gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
#                         vae_end = state.stage_end_time[vae_key]
#                         start = max(gpu_ready, vae_end)
#                         end = start + t
#                         new_gpu = state.gpu_timeline[:]
#                         for g in gpus:
#                             new_gpu[g] = end
#                         new_finished = state.finished | {dit_key}
#                         new_trace = state.trace + [(i, "DIT", k, t, start, end)]
#                         new_end_times = state.stage_end_time.copy()
#                         new_end_times[dit_key] = end
#                         new_state = State(
#                             new_finished,
#                             new_gpu,
#                             max(end, state.total_time),
#                             new_trace,
#                             new_end_times,
#                         )
#                         heapq.heappush(
#                             heap,
#                             (
#                                 new_state.total_time
#                                 + heuristic(new_finished, tasks, total_gpus),
#                                 new_state,
#                             ),
#                         )
#                 break


def plot_schedule_timeline(trace, save_path="generated_schedule_timeline.png"):
    """
    根据调度 trace 生成任务时间线图。
    :param trace: 调度轨迹列表，元素为 (task_id, stage, gpus, duration, start, end, dataset_id)
    :param save_path: 若提供路径则保存图片，否则展示
    """
    df = pd.DataFrame(
        trace, columns=["Task", "Stage", "GPUs", "Time", "Start", "End", "DatasetID"]
    )

    # ✅ 添加任务标签 Task N [Ddataset_id]
    df["TaskLabel"] = df.apply(
        lambda row: f"Task {row['Task']} [D{row['DatasetID']}]", axis=1
    )
    unique_labels = df["TaskLabel"].unique()[::-1]

    plt.figure(figsize=(12, max(6, len(unique_labels) * 0.6)))
    colors = {"DIT": "skyblue", "VAE": "salmon"}

    for idx, row in df.iterrows():
        plt.barh(
            y=row["TaskLabel"],
            width=row["Time"],
            left=row["Start"],
            color=colors[row["Stage"]],
            edgecolor="black",
            label=row["Stage"] if idx < 2 else "",  # 避免重复图例
        )
        # 添加任务信息注释
        plt.text(
            row["Start"] + row["Time"] / 2,
            row["TaskLabel"],
            f'{row["Stage"]}-{row["GPUs"]}G',
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Task")
    plt.title("Task Scheduling Timeline (with Dataset ID)")
    plt.legend(loc="upper right")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📷 图像已保存至: {save_path}")
    else:
        plt.show()


def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
    """
    根据调度 trace 按 GPU rank 显示任务时间线（每个 GPU 一行）
    :param trace: 调度轨迹列表，元素为 (task_id, stage, gpus, duration, start, end, dataset_id, gpu_list)
    :param total_gpus: GPU 数量
    :param save_path: 若提供路径则保存图片，否则展示
    """
    records = []
    for task_id, stage, k, t, start, end, dataset_id, gpu_list in trace:
        for g in gpu_list:
            records.append(
                {
                    "GPU": g,
                    "Task": task_id,
                    "Stage": stage,
                    "Start": start,
                    "End": end,
                    "Time": t,
                    "Label": f"{stage}-T{task_id}[D{dataset_id}]",
                }
            )

    df = pd.DataFrame(records)
    df["GPU"] = df["GPU"].astype(int)

    plt.figure(figsize=(12, max(6, total_gpus * 0.5)))
    colors = {"DIT": "skyblue", "VAE": "salmon"}

    for _, row in df.iterrows():
        plt.barh(
            y=row["GPU"],
            width=row["Time"],
            left=row["Start"],
            color=colors.get(row["Stage"], "gray"),
            edgecolor="black",
        )
        plt.text(
            row["Start"] + row["Time"] / 2,
            row["GPU"],
            row["Label"],
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    plt.yticks(range(total_gpus), [f"GPU {i}" for i in range(total_gpus)])
    plt.xlabel("Time (s)")
    plt.ylabel("GPU ID")
    plt.title("GPU Usage Timeline by Rank")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📷 图像已保存至: {save_path}")
    else:
        plt.show()


# def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
#     """
#     根据调度 trace 按 GPU rank 显示任务时间线（每个 GPU 一行）
#     :param trace: 调度轨迹列表，元素为 (task_id, stage, gpus, duration, start, end, dataset_id)
#     :param total_gpus: GPU 数量
#     :param save_path: 若提供路径则保存图片，否则展示
#     """
#     # 展开每条任务中所有用到的 GPU
#     records = []
#     for task_id, stage, k, t, start, end, dataset_id in trace:
#         gpus, _ = find_earliest_gpus([0] * total_gpus, k)  # 仅为排序用
#         for g in gpus:
#             records.append(
#                 {
#                     "GPU": g,
#                     "Task": task_id,
#                     "Stage": stage,
#                     "Start": start,
#                     "End": end,
#                     "Time": t,
#                     "Label": f"{stage}-T{task_id}[D{dataset_id}]",
#                 }
#             )

#     df = pd.DataFrame(records)
#     df["GPU"] = df["GPU"].astype(int)

#     plt.figure(figsize=(12, max(6, total_gpus * 0.5)))
#     colors = {"DIT": "skyblue", "VAE": "salmon"}

#     for idx, row in df.iterrows():
#         plt.barh(
#             y=row["GPU"],
#             width=row["Time"],
#             left=row["Start"],
#             color=colors[row["Stage"]],
#             edgecolor="black",
#         )
#         # 添加文字注释
#         plt.text(
#             row["Start"] + row["Time"] / 2,
#             row["GPU"],
#             row["Label"],
#             ha="center",
#             va="center",
#             fontsize=8,
#             color="black",
#         )

#     plt.yticks(range(total_gpus), [f"GPU {i}" for i in range(total_gpus)])
#     plt.xlabel("Time (s)")
#     plt.ylabel("GPU ID")
#     plt.title("GPU Usage Timeline by Rank")
#     plt.grid(True, axis="x", linestyle="--", alpha=0.7)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path)
#         print(f"📷 图像已保存至: {save_path}")
#     else:
#         plt.show()


def schedule_random(num_runs=100, schedule_type="a_star"):
    total_gpus = 8
    makespans = []
    durations = []

    for i in range(num_runs):
        # 生成符合顺序递减要求的伪数据集
        dataset_exec_times = generate_fake_dataset_exec_times(num_datasets=4)

        # 替换全局变量（或你可以传入参数）
        global DATASET_EXEC_TIMES
        DATASET_EXEC_TIMES = dataset_exec_times

        print(DATASET_EXEC_TIMES)

        # 模拟任务
        tasks = simulate_tasks_variable(num_tasks=4)

        start = time.time()
        if schedule_type == "a_star":
            result = a_star_schedule(tasks, total_gpus=total_gpus)
            makespans.append(result.total_time)
        else:
            result = greedy_schedule(tasks, total_gpus=total_gpus)
            trace, makespan = greedy_schedule(tasks, total_gpus=8)
            makespans.append(makespan)
        end = time.time()

        if result:
            print(f"第 {i} 次模拟耗时：{end - start:.2f}s")

            durations.append(end - start)
        else:
            print(f"❌ 第 {i} 次模拟未找到调度解")
            makespans.append(None)
            durations.append(end - start)

    # 统计结果（剔除 None）
    valid_makespans = [m for m in makespans if m is not None]
    print("\n🔍 统计结果（基于成功调度）:")
    print(f"总次数: {num_runs}, 成功次数: {len(valid_makespans)}")
    print(f"最大 Makespan: {max(valid_makespans):.2f} s")
    # print(f"平均 Makespan: {np.mean(valid_makespans):.2f} s")
    # print(f"平均调度耗时: {np.mean(durations):.4f} s")

def beam_search_schedule(
    tasks: List[Dict],
    total_gpus: int = 8,
    beam_width: int = 32,
    tie_breaker_eps: float = 1e-6,
    diversity: bool = False,
):
    """
    束搜索：每一层（完成的阶段数相同）只保留评分最好的前 B 个状态。
    评分 = 当前代价 + 启发式估计（makespan + heuristic）

    返回：State（含 trace / total_time / stage_end_time）
    """
    # 初始层：空日程
    init = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    frontier = [init]  # 当前层
    visited_best = {}  # 去重/支配剪枝：signature -> best_cost

    total_stages = 2 * len(tasks)  # 每个任务 VAE+DIT

    def score(st: State) -> float:
        # f = 当前 makespan + 启发式
        return st.total_time + heuristic(st.finished, tasks, total_gpus)

    def signature(st: State):
        # 用“完成集合 + 量化后的 GPU 时间线”做状态签名，防止同一层重复
        # 量化避免浮点毛刺导致的哈希失效
        qt = tuple(round(t, 3) for t in st.gpu_timeline)
        return (tuple(sorted(st.finished)), qt)

    step = 0
    while frontier:
        # 终止条件：这一层有全完成的解
        done = [st for st in frontier if len(st.finished) == total_stages]
        if done:
            # 返回这一层里最优的
            return min(done, key=lambda s: s.total_time)

        # 扩展到下一层
        next_candidates = []

        for state in frontier:
            # 一次扩展：对所有任务尝试扩展下一个可用阶段
            for i in range(len(tasks)):
                vae_key, dit_key = f"{i}_VAE", f"{i}_DIT"

                # 1) 扩展 VAE（未完成）
                if vae_key not in state.finished:
                    for k, t in tasks[i]["vae"]:
                        if total_gpus < k:
                            continue
                        gpus, start = find_earliest_gpus(state.gpu_timeline, k)
                        end = start + t

                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end

                        new_finished = state.finished | {vae_key}
                        new_trace = state.trace + [
                            (i, "VAE", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[vae_key] = end

                        cand = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )

                        # 支配剪枝：同签名只保留更优的
                        sig = signature(cand)
                        val = cand.total_time
                        if (sig not in visited_best) or (val < visited_best[sig] - tie_breaker_eps):
                            visited_best[sig] = val
                            next_candidates.append(cand)

                # 2) 扩展 DIT（VAE 完成但 DIT 未完成）
                elif dit_key not in state.finished:
                    vae_end = state.stage_end_time[vae_key]
                    for k, t in tasks[i]["dit"]:
                        if total_gpus < k:
                            continue
                        gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
                        start = max(gpu_ready, vae_end)
                        end = start + t

                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end

                        new_finished = state.finished | {dit_key}
                        new_trace = state.trace + [
                            (i, "DIT", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[dit_key] = end

                        cand = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )

                        sig = signature(cand)
                        val = cand.total_time
                        if (sig not in visited_best) or (val < visited_best[sig] - tie_breaker_eps):
                            visited_best[sig] = val
                            next_candidates.append(cand)

        if not next_candidates:
            # 没法扩展，失败
            return None

        # 选择下一层的前 B：按 score 排序
        next_candidates.sort(key=score)

        if diversity:
            # 可选：简单多样性——去掉 GPU 时间线完全相同的近邻
            filtered = []
            seen_gpu = set()
            for st in next_candidates:
                key = tuple(round(t, 2) for t in st.gpu_timeline)
                if key in seen_gpu:
                    continue
                seen_gpu.add(key)
                filtered.append(st)
                if len(filtered) >= beam_width:
                    break
            frontier = filtered
        else:
            frontier = next_candidates[:beam_width]

        step += 1

    return None

def heuristic(finished, tasks, total_gpus, state=None):
    rem_ops = []
    for i in range(len(tasks)):
        if f"{i}_VAE" not in finished:
            rem_ops.append(("VAE", i, tasks[i]["vae"]))
        if f"{i}_DIT" not in finished:
            rem_ops.append(("DIT", i, tasks[i]["dit"]))

    # 没剩余任务时返回0
    if not rem_ops:
        return 0.0

    gpu_time = 0.0
    op_min_time = 0.0
    for _, _, opts in rem_ops:
        if not opts:  # 防止空列表
            continue
        min_t = min(t for _, t in opts)
        min_gpu_t = min(sp * t for sp, t in opts)
        gpu_time += min_gpu_t
        op_min_time = max(op_min_time, min_t)
    LB_gpu = gpu_time / total_gpus if total_gpus > 0 else 0.0
    LB_op  = op_min_time

    LB_cp = 0.0
    for i in range(len(tasks)):
        remain = 0.0
        if f"{i}_VAE" not in finished and tasks[i]["vae"]:
            remain += min(t for _, t in tasks[i]["vae"])
        if f"{i}_DIT" not in finished and tasks[i]["dit"]:
            remain += min(t for _, t in tasks[i]["dit"])
        LB_cp = max(LB_cp, remain)

    return float(max(LB_gpu, LB_op, LB_cp))


def pareto_filter(opts):
    """
    对 (sp, time) 列表做简单的 Pareto 预筛：
    - 去掉时间更慢且占用 GPU 不更少的选项；
    - 返回少量“非支配”档位，通常 2~3 个。
    """
    if not opts:
        return []
    # 先按 time 升序，其次按 sp 升序
    S = sorted(opts, key=lambda x: (x[1], x[0]))
    kept = []
    best_time = float('inf')
    best_sp = float('inf')
    for sp, t in S:
        # 若时间更短或相等但 GPU 更少，则保留
        if t < best_time or (t == best_time and sp < best_sp):
            kept.append((sp, t))
            best_time, best_sp = t, sp
    return kept


def a_star_nonop(tasks: List[Dict], total_gpus: int = 32,
                    w: float = 1.6,
                    time_budget: float = 2.0,
                    max_nodes: int = 200000,
                    k_tasks: int = 4, k_sp: int = 2):
    # 预筛 SP
    for i in range(len(tasks)):
        tasks[i]["vae"] = pareto_filter(tasks[i]["vae"])
        tasks[i]["dit"] = pareto_filter(tasks[i]["dit"])

    t0 = time.time()
    initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    heap = []
    best_state, best_cost = None, float('inf')

    # 初始上界（可选）
    try:
        gtrace, gms = greedy_schedule(tasks, total_gpus)
        best_cost = gms
    except Exception:
        pass

    def push(st):
        nonlocal best_cost
        f = st.total_time + w * heuristic(st.finished, tasks, total_gpus, st)
        if f >= best_cost:   # 上界剪枝（近似场景一样好用）
            return
        heapq.heappush(heap, (f, -st.total_time, st))

    push(initial)
    expanded = 0

    while heap and time.time()-t0 < time_budget and expanded < max_nodes:
        _, _, state = heapq.heappop(heap)
        expanded += 1

        if len(state.finished) == 2 * len(tasks):
            if state.total_time < best_cost:
                best_cost, best_state = state.total_time, state
            continue

        # 只扩就绪操作
        ready_ops = []
        for i in range(len(tasks)):
            if f"{i}_VAE" not in state.finished:
                ready_ops.append(("VAE", i, tasks[i]["vae"]))
            elif f"{i}_DIT" not in state.finished:
                ready_ops.append(("DIT", i, tasks[i]["dit"]))

        # 只取前K个“最有希望”的任务（最短min时间）
        ready_ops.sort(key=lambda x: min(t for _, t in x[2]))
        for stage, i, opts in ready_ops[:k_tasks]:
            # 每个任务取前M个最短时间的SP
            sp_opts = sorted(opts, key=lambda x: x[1])[:k_sp]
            for k, t in sp_opts:
                # 放置（保持你的一致性：VAE/DIT用相同的分配规则）
                if stage == "VAE":
                    gpus, start = find_earliest_gpus(state.gpu_timeline, k)
                else:
                    gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
                    vae_end = state.stage_end_time[f"{i}_VAE"]
                    start = max(gpu_ready, vae_end)

                end = start + t
                new_gpu = state.gpu_timeline[:]
                for g in gpus: new_gpu[g] = end
                new_finished = state.finished | {f"{i}_{stage}"}
                new_trace = state.trace + [(i, stage, k, t, start, end, tasks[i]["dataset_id"], gpus)]
                new_end_times = state.stage_end_time.copy()
                new_end_times[f"{i}_{stage}"] = end
                new_state = State(new_finished, new_gpu, max(end, state.total_time), new_trace, new_end_times)

                # 立即更新最好解（ Anytime 风格 ）
                if len(new_finished) == 2 * len(tasks) and new_state.total_time < best_cost:
                    best_cost, best_state = new_state.total_time, new_state

                push(new_state)

    return best_state if best_state else (heap and heap[0][2])


def schedule(yaml_path: str, type="nonop", tasks: List[Dict] = None, n_gpus=8, beam_width=10000):

    if tasks is None:
        tasks = simulate_tasks_variable(num_tasks=4)
    else:
        # 如果传入了 tasks，则不再模拟
        print("Using provided tasks for scheduling.")

    now = time.time()

    if type == "a_star":
        result = a_star_schedule(tasks, total_gpus=n_gpus)
    elif type == "nonop":
        result = a_star_nonop(tasks,total_gpus=n_gpus)
    elif type == "beam":
        result = beam_search_schedule(tasks, total_gpus=n_gpus, beam_width=beam_width, diversity=True)
    else:
        result = greedy_schedule(tasks, total_gpus=n_gpus)

    if result and result.trace:  # 确保 result 和 trace 都有效
        # ⭐ --- 新增代码：保存详细 Trace 到 CSV 文件 ---
        # 1. 定义CSV文件名 (与YAML文件同名，扩展名不同)
        trace_csv_path = yaml_path.replace(".yaml", "_trace.csv")

        # 2. 将 trace 转换为 Pandas DataFrame
        df = pd.DataFrame(
            result.trace,
            columns=[
                "Task_ID",  # 改为 Task_ID 以示区分
                "Stage",
                "GPUs_Count",  # 改为 GPUs_Count
                "Time",
                "Start",
                "End",
                "DatasetID",
                "GPU_List",
            ],
        )

        # 3. 添加 Makespan 和其他元数据
        df["Makespan"] = result.total_time

        # 4. 保存到 CSV 文件
        df.to_csv(trace_csv_path, index=False)
        print(f"📄 已将详细调度轨迹保存到 {trace_csv_path}")
        # --- 新增代码结束 ---

        print(df)
        print(f"\n✅ 最优 Makespan = {result.total_time:.2f}s")
        print(f"⏱️ 调度耗时: {time.time() - now:.2f}s")

        # 调用我们之前写好的函数来保存YAML
        save_yaml_plan_from_trace(result.trace, yaml_path)

        # 生成并保存GPU利用率图
        plot_schedule_by_gpu(
            result.trace,
            total_gpus=n_gpus,
            save_path=yaml_path.replace(".yaml", ".png"),
        )

    else:
        print("❌ 未找到调度方案")


def trace_to_yaml_with_correct_deps(trace: List[Tuple]) -> List[Dict]:
    """
    在保持原有命名格式和结构的前提下，正确生成包含所有资源依赖的YAML任务列表。

    处理步骤:
    1.  **预处理**: 遍历一次 trace，为每个任务的每个阶段预先生成其唯一的、符合你格式的名称，并存储在一个映射中。
    2.  **排序和依赖分析**:
        a.  按任务的实际开始时间 (`start`) 对 trace 进行排序，得到真实的执行顺序。
        b.  初始化一个 GPU 追踪器 `last_task_on_gpu`。
        c.  遍历排序后的 trace，对于每个任务：
            i.  查找它所使用的 GPU 在它之前被哪些任务占用，将这些任务加入依赖列表。
            ii. 如果是 DIT 任务，额外加入其对应的 VAE 任务作为依赖。
            iii.将包含完整依赖信息的任务字典存储起来。
    3.  **重组**: 按照原始 `trace` 的顺序，重新组织任务列表，以确保最终输出的YAML文件中的任务顺序与你之前的脚本尽可能一致。
    """
    # === 步骤 1: 预生成所有任务的名称，并建立映射 ===
    # task_key (e.g., "0_VAE") -> full_name (e.g., "D_97_VAE_g0123")
    task_name_map = {}

    conflict_tasks = find_conflict_task_names(trace)  # ✅ 获取有冲突的任务名集合
    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in trace:
        task_key = f"{task_id}_{stage}"
        gpu_str = "".join(str(g) for g in sorted(gpu_list))
        # 保持你原来的命名格式
        task_name = f"{dataset_id}_{stage}_g{gpu_str}"
        task_name_map[task_key] = task_name

    # === 步骤 2: 按时间顺序分析，构建包含完整依赖的任务字典 ===
    sorted_trace = sorted(trace, key=lambda x: x[4])  # x[4] is the start time

    # Key: full_task_name, Value: 完整的任务字典
    tasks_with_deps_dict = {}
    # Key: gpu_id, Value: full_task_name
    last_task_on_gpu = {}

    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in sorted_trace:
        current_task_key = f"{task_id}_{stage}"
        current_task_name = task_name_map[current_task_key]

        # 收集所有依赖项
        dependencies = set()

        # 2.1. 资源依赖：基于 GPU 的前后关系
        for gpu in gpu_list:
            if gpu in last_task_on_gpu:
                dependencies.add(last_task_on_gpu[gpu])

        # 2.2. 内部依赖：DIT 依赖 VAE
        if stage == "DIT":
            vae_key = f"{task_id}_VAE"
            if vae_key in task_name_map:
                dependencies.add(task_name_map[vae_key])

        # 构建任务字典，格式与你原来的一致
        task_dict = {
            "name": current_task_name,
            "gpus": sorted(gpu_list),
            "dependencies": sorted(list(dependencies)),  # 保证依赖列表顺序稳定
            "args": {
                "task_type": stage,
                "sp": sp,
            },
        }

        if stage == "DIT":
            task_dict["args"]["data_source_task"] = task_name_map.get(f"{task_id}_VAE")

        if task_name in conflict_tasks:
            task_dict["overlapped"] = True
        else:
            task_dict["overlapped"] = False

        tasks_with_deps_dict[current_task_name] = task_dict

        # 更新 GPU 追踪器
        for gpu in gpu_list:
            last_task_on_gpu[gpu] = current_task_name

    # === 步骤 3: 按照原始 trace 的顺序组装最终的列表 ===
    # 这确保了最终YAML文件中的任务顺序与你之前的脚本输出基本一致
    final_task_list = []
    # 使用一个集合来确保每个任务只被添加一次
    added_tasks = set()
    for task_id, stage, _, _, _, _, _, _ in trace:
        task_key = f"{task_id}_{stage}"
        task_name = task_name_map.get(task_key)
        if task_name and task_name not in added_tasks:
            final_task_list.append(tasks_with_deps_dict[task_name])
            added_tasks.add(task_name)

    return final_task_list


def find_conflict_task_names(trace: List[Tuple]) -> set:
    """
    返回发生 GPU 冲突的任务名称集合（即会写入 YAML 的 task.name）
    """
    conflicted_names = set()

    for i in range(len(trace)):
        id_i, _, _, _, start_i, end_i, dataset_i, gpus_i = trace[i]
        gpu_str_i = "".join(str(g) for g in sorted(gpus_i))
        name_i = f"{dataset_i}_{trace[i][1]}_g{gpu_str_i}"  # stage 是 trace[i][1]

        for j in range(i + 1, len(trace)):
            id_j, _, _, _, start_j, end_j, dataset_j, gpus_j = trace[j]
            gpu_str_j = "".join(str(g) for g in sorted(gpus_j))
            name_j = f"{dataset_j}_{trace[j][1]}_g{gpu_str_j}"

            if end_i > start_j and end_j > start_i:
                overlap = set(gpus_i) & set(gpus_j)
                if overlap:
                    conflicted_names.add(name_i)
                    conflicted_names.add(name_j)

    return conflicted_names


def save_yaml_plan_from_trace(
    trace: List[Tuple], yaml_path: str = "generated_plan.yaml"
):
    print(f"Trying to transform to yaml")
    tasks_yaml = trace_to_yaml_with_correct_deps(trace)

    tasks_yaml = {"tasks": tasks_yaml}

    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, yaml_path)

    print(f"Trying to save to: {save_path}")
    with open(save_path, "w") as f:
        yaml.dump(tasks_yaml, f, sort_keys=False)
    print(f"✅ 已将调度计划保存到 {save_path}")


def yaml_to_trace(yaml_path: str) -> Tuple[List[Tuple], Dict[str, List[str]]]:
    """
    从 YAML 文件中解析任务调度计划，恢复为 trace 列表。
    每项 trace 格式：
      (task_id, stage, sp, duration=0, start=0, end=0, dataset_id, gpu_list)
    """
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
        yaml_tasks = content["tasks"]

    trace = []
    dependency_map = {}

    for task in yaml_tasks:
        name = task["name"]  # 例: D_001_DIT_g0167
        stage = task["args"]["task_type"]
        sp = task["args"]["sp"]
        gpus = task["gpus"]
        dataset_id = "_".join(name.split("_")[:2])  # D_001
        deps = task.get("dependencies", [])

        trace.append((name, stage, sp, 0.0, 0.0, 0.0, dataset_id, gpus))
        dependency_map[name] = deps

    return trace, dependency_map


def topological_sort_trace(trace: List[Tuple], dependency_map: dict) -> List[Tuple]:
    name_to_trace = {t[0]: t for t in trace}
    visited = set()
    sorted_trace = []

    while len(visited) < len(trace):
        progress = False
        for name, deps in dependency_map.items():
            if name in visited:
                continue
            if all(d in visited for d in deps):
                sorted_trace.append(name_to_trace[name])
                visited.add(name)
                progress = True
        if not progress:
            raise RuntimeError("❌ 拓扑排序失败，可能存在循环依赖")
    return sorted_trace


def replay_schedule(
    trace: List[Tuple],
    dependency_map: dict,
    dataset_exec_times: dict,
    total_gpus: int = 8,
) -> Tuple[List[Tuple], float]:
    gpu_timeline = [0.0] * total_gpus
    stage_end_time = {}
    finished = set()
    replayed_trace = []

    for task_id, stage, sp, _, _, _, dataset_id, gpu_list in trace:
        for dep in dependency_map.get(task_id, []):
            if dep not in finished:
                raise RuntimeError(f"{task_id} 的依赖 {dep} 未完成")

        dataset = next(
            (v for v in dataset_exec_times.values() if v["name"] == dataset_id), None
        )
        if not dataset:
            raise KeyError(f"找不到数据集: {dataset_id}")
        duration = dataset[stage.lower()][sp]

        ready_time = max(gpu_timeline[g] for g in gpu_list)
        for dep in dependency_map.get(task_id, []):
            ready_time = max(ready_time, stage_end_time[dep])

        start = ready_time
        end = start + duration
        for g in gpu_list:
            gpu_timeline[g] = end

        stage_end_time[task_id] = end
        finished.add(task_id)

        replayed_trace.append(
            (task_id, stage, sp, duration, start, end, dataset_id, gpu_list)
        )

    makespan = max(gpu_timeline)
    print(f"✅ Replay Makespan: {makespan:.2f}s")
    return replayed_trace, makespan


# 得到调度图和生成yml
if __name__ == "__main__":
    schedule("generated_plan.yaml")


# 根据yml重放调度图
# if __name__ == "__main__":
#     yaml_path = "/Users/xander/Documents/git/flex/Megatron_VAST/t2v_flow/planner/generated_plan.yaml"

#     # 1. 加载 trace 和依赖关系
#     trace, dependency_map = yaml_to_trace(yaml_path)

#     # 2. 拓扑排序（确保依赖顺序）
#     sorted_trace = topological_sort_trace(trace, dependency_map)

#     # 3. 执行调度重放
#     replayed_trace, makespan = replay_schedule(
#         sorted_trace,
#         dependency_map=dependency_map,
#         dataset_exec_times=DATASET_EXEC_TIMES,  # 可切换为 DATASET_EXEC_TIMES_20 或 120
#         total_gpus=8,
#     )

#     # 4. 可视化
#     plot_schedule_by_gpu(replayed_trace, total_gpus=8)
