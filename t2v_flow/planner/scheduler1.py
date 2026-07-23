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
    SP_STRATEGIES = [1, 2, 4, 8]
    fake_exec_times = {}

    for i in range(num_datasets):
        name = f"D_{i:03d}"

        base_vae = random.uniform(20, 60)
        base_dit = random.uniform(40, 100)

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
                "dataset_id": name,
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
    assert (k & (k - 1)) == 0, "k must be a power of two"

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
      trace: List[(task_id, stage, sp, dur, start, end, dataset_id, gpu_list)]
      makespan: float
    """
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time: Dict[str, float] = {}

    for i in range(len(tasks)):
        dataset_id = tasks[i]["dataset_id"]

        selected = False
        for k, t in sorted(tasks[i]["vae"], key=lambda x: x[1]):
            gpus, start = find_earliest_gpus(gpu_timeline, k)
            end = start + t
            for g in gpus:
                gpu_timeline[g] = end
            trace.append((i, "VAE", k, t, start, end, dataset_id, gpus))
            stage_end_time[f"{i}_VAE"] = end
            selected = True
            break

        if not selected:
            raise RuntimeError(f"VAE stage: cannot allocate {k} available GPUs for task {i}")

        selected = False
        for k, t in sorted(tasks[i]["dit"], key=lambda x: x[1]):
            candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
            if not candidate_blocks:
                continue
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
            raise RuntimeError(f"DIT stage: cannot allocate {k} available GPUs for task {i}")

    makespan = max(gpu_timeline) if gpu_timeline else 0.0
    return trace, makespan


# def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
#     gpu_timeline = [0.0] * total_gpus
#     trace = []
#     stage_end_time = {}

#     for i in range(len(tasks)):
#         dataset_id = tasks[i]["dataset_id"]

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

#     makespan = max(gpu_timeline)
#     return trace, makespan


# def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
#     gpu_timeline = [0.0] * total_gpus
#     finished = set()
#     trace = []
#     stage_end_time = {}

#     for i in range(len(tasks)):
#         dataset_id = tasks[i]["dataset_id"]
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
    df = pd.DataFrame(
        trace, columns=["Task", "Stage", "GPUs", "Time", "Start", "End", "DatasetID"]
    )

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
            label=row["Stage"] if idx < 2 else "",
        )
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
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
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
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


# def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
#     """
#     """
#     records = []
#     for task_id, stage, k, t, start, end, dataset_id in trace:
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
#     else:
#         plt.show()


def schedule_random(num_runs=100, schedule_type="a_star"):
    total_gpus = 8
    makespans = []
    durations = []

    for i in range(num_runs):
        dataset_exec_times = generate_fake_dataset_exec_times(num_datasets=4)

        global DATASET_EXEC_TIMES
        DATASET_EXEC_TIMES = dataset_exec_times

        print(DATASET_EXEC_TIMES)

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
            print(f"Simulation {i} took {end - start:.2f}s")

            durations.append(end - start)
        else:
            print(f"[ERROR] simulation {i} found no schedule")
            makespans.append(None)
            durations.append(end - start)

    valid_makespans = [m for m in makespans if m is not None]
    print("\nStatistics (over the successful schedules):")
    print(f"Total runs: {num_runs}, successful: {len(valid_makespans)}")
    print(f"Max makespan: {max(valid_makespans):.2f} s")

def beam_search_schedule(
    tasks: List[Dict],
    total_gpus: int = 8,
    beam_width: int = 32,
    tie_breaker_eps: float = 1e-6,
    diversity: bool = False,
):
    init = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    frontier = [init]
    visited_best = {}

    total_stages = 2 * len(tasks)

    def score(st: State) -> float:
        return st.total_time + heuristic(st.finished, tasks, total_gpus)

    def signature(st: State):
        qt = tuple(round(t, 3) for t in st.gpu_timeline)
        return (tuple(sorted(st.finished)), qt)

    step = 0
    while frontier:
        done = [st for st in frontier if len(st.finished) == total_stages]
        if done:
            return min(done, key=lambda s: s.total_time)

        next_candidates = []

        for state in frontier:
            for i in range(len(tasks)):
                vae_key, dit_key = f"{i}_VAE", f"{i}_DIT"

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

                        sig = signature(cand)
                        val = cand.total_time
                        if (sig not in visited_best) or (val < visited_best[sig] - tie_breaker_eps):
                            visited_best[sig] = val
                            next_candidates.append(cand)

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
            return None

        next_candidates.sort(key=score)

        if diversity:
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

    if not rem_ops:
        return 0.0

    gpu_time = 0.0
    op_min_time = 0.0
    for _, _, opts in rem_ops:
        if not opts:
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
    if not opts:
        return []
    S = sorted(opts, key=lambda x: (x[1], x[0]))
    kept = []
    best_time = float('inf')
    best_sp = float('inf')
    for sp, t in S:
        if t < best_time or (t == best_time and sp < best_sp):
            kept.append((sp, t))
            best_time, best_sp = t, sp
    return kept


def a_star_nonop(tasks: List[Dict], total_gpus: int = 32,
                    w: float = 1.6,
                    time_budget: float = 2.0,
                    max_nodes: int = 200000,
                    k_tasks: int = 4, k_sp: int = 2):
    for i in range(len(tasks)):
        tasks[i]["vae"] = pareto_filter(tasks[i]["vae"])
        tasks[i]["dit"] = pareto_filter(tasks[i]["dit"])

    t0 = time.time()
    initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    heap = []
    best_state, best_cost = None, float('inf')

    try:
        gtrace, gms = greedy_schedule(tasks, total_gpus)
        best_cost = gms
    except Exception:
        pass

    def push(st):
        nonlocal best_cost
        f = st.total_time + w * heuristic(st.finished, tasks, total_gpus, st)
        if f >= best_cost:
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

        ready_ops = []
        for i in range(len(tasks)):
            if f"{i}_VAE" not in state.finished:
                ready_ops.append(("VAE", i, tasks[i]["vae"]))
            elif f"{i}_DIT" not in state.finished:
                ready_ops.append(("DIT", i, tasks[i]["dit"]))

        ready_ops.sort(key=lambda x: min(t for _, t in x[2]))
        for stage, i, opts in ready_ops[:k_tasks]:
            sp_opts = sorted(opts, key=lambda x: x[1])[:k_sp]
            for k, t in sp_opts:
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

                if len(new_finished) == 2 * len(tasks) and new_state.total_time < best_cost:
                    best_cost, best_state = new_state.total_time, new_state

                push(new_state)

    return best_state if best_state else (heap and heap[0][2])


def schedule(yaml_path: str, type="nonop", tasks: List[Dict] = None, n_gpus=8, beam_width=10000):

    if tasks is None:
        tasks = simulate_tasks_variable(num_tasks=4)
    else:
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

    if result and result.trace:
        trace_csv_path = yaml_path.replace(".yaml", "_trace.csv")

        df = pd.DataFrame(
            result.trace,
            columns=[
                "Task_ID",
                "Stage",
                "GPUs_Count",
                "Time",
                "Start",
                "End",
                "DatasetID",
                "GPU_List",
            ],
        )

        df["Makespan"] = result.total_time

        df.to_csv(trace_csv_path, index=False)
        print(f"Detailed schedule trace saved to {trace_csv_path}")

        print(df)
        print(f"\n[OK] best makespan = {result.total_time:.2f}s")
        print(f"Scheduling took: {time.time() - now:.2f}s")

        save_yaml_plan_from_trace(result.trace, yaml_path)

        plot_schedule_by_gpu(
            result.trace,
            total_gpus=n_gpus,
            save_path=yaml_path.replace(".yaml", ".png"),
        )

    else:
        print("[ERROR] no schedule found")


def trace_to_yaml_with_correct_deps(trace: List[Tuple]) -> List[Dict]:
    # task_key (e.g., "0_VAE") -> full_name (e.g., "D_97_VAE_g0123")
    task_name_map = {}

    conflict_tasks = find_conflict_task_names(trace)
    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in trace:
        task_key = f"{task_id}_{stage}"
        gpu_str = "".join(str(g) for g in sorted(gpu_list))
        task_name = f"{dataset_id}_{stage}_g{gpu_str}"
        task_name_map[task_key] = task_name

    sorted_trace = sorted(trace, key=lambda x: x[4])  # x[4] is the start time

    tasks_with_deps_dict = {}
    # Key: gpu_id, Value: full_task_name
    last_task_on_gpu = {}

    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in sorted_trace:
        current_task_key = f"{task_id}_{stage}"
        current_task_name = task_name_map[current_task_key]

        dependencies = set()

        for gpu in gpu_list:
            if gpu in last_task_on_gpu:
                dependencies.add(last_task_on_gpu[gpu])

        if stage == "DIT":
            vae_key = f"{task_id}_VAE"
            if vae_key in task_name_map:
                dependencies.add(task_name_map[vae_key])

        task_dict = {
            "name": current_task_name,
            "gpus": sorted(gpu_list),
            "dependencies": sorted(list(dependencies)),
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

        for gpu in gpu_list:
            last_task_on_gpu[gpu] = current_task_name

    final_task_list = []
    added_tasks = set()
    for task_id, stage, _, _, _, _, _, _ in trace:
        task_key = f"{task_id}_{stage}"
        task_name = task_name_map.get(task_key)
        if task_name and task_name not in added_tasks:
            final_task_list.append(tasks_with_deps_dict[task_name])
            added_tasks.add(task_name)

    return final_task_list


def find_conflict_task_names(trace: List[Tuple]) -> set:
    conflicted_names = set()

    for i in range(len(trace)):
        id_i, _, _, _, start_i, end_i, dataset_i, gpus_i = trace[i]
        gpu_str_i = "".join(str(g) for g in sorted(gpus_i))
        name_i = f"{dataset_i}_{trace[i][1]}_g{gpu_str_i}"

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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, yaml_path)

    print(f"Trying to save to: {save_path}")
    with open(save_path, "w") as f:
        yaml.dump(tasks_yaml, f, sort_keys=False)
    print(f"[OK] schedule saved to {save_path}")


def yaml_to_trace(yaml_path: str) -> Tuple[List[Tuple], Dict[str, List[str]]]:
    """
      (task_id, stage, sp, duration=0, start=0, end=0, dataset_id, gpu_list)
    """
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
        yaml_tasks = content["tasks"]

    trace = []
    dependency_map = {}

    for task in yaml_tasks:
        name = task["name"]
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
            raise RuntimeError("Topological sort failed; there may be a dependency cycle")
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
                raise RuntimeError(f"dependency {dep} of {task_id} is not complete")

        dataset = next(
            (v for v in dataset_exec_times.values() if v["name"] == dataset_id), None
        )
        if not dataset:
            raise KeyError(f"dataset not found: {dataset_id}")
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


if __name__ == "__main__":
    schedule("generated_plan.yaml")
