import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ==============================================================================
# 1. 通信模型 (与之前相同)
# ==============================================================================

class CommunicationModelConfig:
    GPUS_PER_NODE = 8
    ENABLE_INTRA_TASK_PENALTY = True
    INTRA_TASK_PENALTY_MAX_SP = 8
    INTRA_TASK_BASE_PENALTY = 2.5 
    INTRA_TASK_IMBALANCE_FACTOR = 1.5
    ENABLE_INTER_TASK_COST = True
    COST_BROADCAST = 3.0
    COST_P2P_BROADCAST = 5.0

def get_node_id(gpu_id: int) -> int:
    return gpu_id // CommunicationModelConfig.GPUS_PER_NODE

def calculate_intra_task_penalty(gpu_list: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTRA_TASK_PENALTY: return 0.0, "Intra-task penalty disabled."
    k = len(gpu_list)
    if k <= 1: return 0.0, "SP=1, no penalty."
    if k > CommunicationModelConfig.INTRA_TASK_PENALTY_MAX_SP: return 0.0, f"SP={k} > {CommunicationModelConfig.INTRA_TASK_PENALTY_MAX_SP}, penalty not applied."
    node_counts = {get_node_id(g): 0 for g in gpu_list}
    for g in gpu_list: node_counts[get_node_id(g)] += 1
    if len(node_counts) <= 1: return 0.0, f"All GPUs on a single node (Node {list(node_counts.keys())[0]})."
    counts = list(node_counts.values())
    base_penalty = CommunicationModelConfig.INTRA_TASK_BASE_PENALTY * (len(node_counts) - 1)
    imbalance_penalty = np.std(counts) * CommunicationModelConfig.INTRA_TASK_IMBALANCE_FACTOR
    total_penalty = base_penalty + imbalance_penalty
    reason = (f"Cross-node penalty ({node_counts}), Penalty: {total_penalty:.2f}s")
    return total_penalty, reason

def calculate_inter_task_communication_cost(vae_gpus: List[int], dit_gpus: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTER_TASK_COST: return 0.0, "Inter-task cost disabled."
    if not vae_gpus or not dit_gpus: return 0.0, "Invalid GPU list."
    vae_gpu_set, dit_gpu_set = set(vae_gpus), set(dit_gpus)
    if vae_gpu_set == dit_gpu_set: return 0.0, f"Full Overlap. Cost: 0.0s."
    cost = CommunicationModelConfig.COST_BROADCAST if vae_gpu_set.intersection(dit_gpu_set) else CommunicationModelConfig.COST_P2P_BROADCAST
    type_str = "Broadcast" if cost == CommunicationModelConfig.COST_BROADCAST else "P2P+Broadcast"
    reason = f"{type_str}: VAE GPUs {sorted(list(vae_gpu_set))} -> DiT GPUs {sorted(list(dit_gpu_set))}. Cost: {cost:.2f}s."
    return cost, reason

# ==============================================================================
# 2. 核心分析与重计算函数 (与之前相同)
# ==============================================================================

def analyze_and_correct_trace(original_trace_df: pd.DataFrame, total_gpus: int) -> Tuple[pd.DataFrame, float]:
    trace_df = original_trace_df.sort_values(by='Start').reset_index(drop=True)
    new_gpu_timeline = [0.0] * total_gpus
    new_task_end_times = {}
    task_gpu_map = { (row.Task_ID, row.Stage): ast.literal_eval(row.GPU_List) if isinstance(row.GPU_List, str) else row.GPU_List for _, row in trace_df.iterrows() }
    analysis_results = []

    print("Re-simulating trace with communication model...")
    for _, row in tqdm(trace_df.iterrows(), total=len(trace_df), desc="Analyzing Events"):
        task_key = (row.Task_ID, row.Stage)
        assigned_gpus = task_gpu_map[task_key]
        intra_penalty, intra_reason = calculate_intra_task_penalty(assigned_gpus)
        inter_cost, inter_reason, predecessor_new_end_time = 0.0, "N/A", 0.0
        
        if row.Stage == 'DIT':
            vae_key = (row.Task_ID, 'VAE')
            if vae_key in task_gpu_map:
                inter_cost, inter_reason = calculate_inter_task_communication_cost(task_gpu_map[vae_key], assigned_gpus)
                predecessor_new_end_time = new_task_end_times.get(vae_key, 0.0)

        dependency_ready_time = predecessor_new_end_time + inter_cost
        gpus_ready_time = max(new_gpu_timeline[g] for g in assigned_gpus) if assigned_gpus else 0
        new_start_time = max(dependency_ready_time, gpus_ready_time)
        new_duration = row.Time + intra_penalty
        new_end_time = new_start_time + new_duration
        
        for g in assigned_gpus: new_gpu_timeline[g] = new_end_time
        new_task_end_times[task_key] = new_end_time

        analysis_results.append({
            'Task_ID': row.Task_ID, 'Stage': row.Stage,
            'Original_Start': row.Start, 'Corrected_Start': new_start_time,
            'Original_End': row.End, 'Corrected_End': new_end_time,
            'Original_Time': row.Time, 'Corrected_Time': new_duration,
            'Inter_Comm_Cost': inter_cost, 'Intra_Penalty': intra_penalty,
            'Cost_Reason': f"{inter_reason} | {intra_reason}",
            'GPU_List': assigned_gpus
        })

    return pd.DataFrame(analysis_results), (max(new_gpu_timeline) if new_gpu_timeline else 0.0)

# ==============================================================================
# 3. 新增：可视化函数
# ==============================================================================

def visualize_schedule_comparison(corrected_df: pd.DataFrame, original_makespan: float, corrected_makespan: float, iter: int):
    """
    生成一个对比原始调度和修正后调度的Gantt图。
    """
    # 准备绘图数据
    tasks = [f"Task {r.Task_ID}-{r.Stage}" for _, r in corrected_df.iterrows()]
    colors = {'VAE': '#3498db', 'DIT': '#e74c3c'}
    penalty_color = '#f39c12'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    fig.suptitle('Schedule Comparison: Original vs. Corrected (Communication-Aware)', fontsize=18, weight='bold')

    # --- 绘制图1: 原始调度 ---
    ax1.set_title('Original Schedule (Communication-Unaware)', fontsize=14)
    for i, row in corrected_df.iterrows():
        task_label = f"Task {row.Task_ID}-{row.Stage}"
        ax1.barh(task_label, row.Original_Time, left=row.Original_Start, 
                 color=colors[row.Stage], edgecolor='black', linewidth=0.5)
    ax1.axvline(x=original_makespan, color='grey', linestyle='--', label=f'Original Makespan: {original_makespan:.2f}s')
    ax1.invert_yaxis()
    ax1.grid(axis='x', linestyle=':', color='gray')
    ax1.set_ylabel('Tasks')
    ax1.legend(loc='lower right')

    # --- 绘制图2: 修正后的调度 ---
    ax2.set_title('Corrected Schedule (Communication-Aware)', fontsize=14)
    for i, row in corrected_df.iterrows():
        task_label = f"Task {row.Task_ID}-{row.Stage}"
        # 绘制原始执行时间部分
        ax2.barh(task_label, row.Original_Time, left=row.Corrected_Start, 
                 color=colors[row.Stage], edgecolor='black', linewidth=0.5)
        # 绘制任务内惩罚部分（如果存在）
        if row.Intra_Penalty > 0:
            ax2.barh(task_label, row.Intra_Penalty, left=row.Corrected_Start + row.Original_Time,
                     color=penalty_color, edgecolor='black', linewidth=0.5, hatch='//')
    ax2.axvline(x=corrected_makespan, color='red', linestyle='--', label=f'Corrected Makespan: {corrected_makespan:.2f}s')
    ax2.invert_yaxis()
    ax2.grid(axis='x', linestyle=':', color='gray')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Tasks')

    # 创建图例
    legend_patches = [mpatches.Patch(color=colors['VAE'], label='VAE Task'),
                      mpatches.Patch(color=colors['DIT'], label='DIT Task'),
                      mpatches.Patch(color=penalty_color, hatch='//', label='Intra-Task Penalty (Asymmetry)')]
    ax2.legend(handles=legend_patches, loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 保存图像
    output_filename = f"schedule_comparison_{iter}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n📊 Visualization saved to '{output_filename}'")


# ==============================================================================
# 4. 主执行流程
# ==============================================================================

if __name__ == '__main__':
    # ... 配置区域 (与之前相同) ...
    iter = 4
    ORIGINAL_TRACE_CSV_PATH = f"generated_schedules/wan/schedule_{iter}_trace.csv"
    TOTAL_GPUS = 16
    
    # ... 打印配置信息 ...
    print("\n" + "="*80)
    print(" T R A C E   A N A L Y S I S   A N D   C O R R E C T I O N   T O O L ")
    print("="*80 + "\n")

    # 加载和执行分析
    try:
        original_df = pd.read_csv(ORIGINAL_TRACE_CSV_PATH)
        print(f"✅ Successfully loaded original trace from '{ORIGINAL_TRACE_CSV_PATH}'.\n")
    except Exception as e:
        print(f"❌ ERROR: {e}"); exit()

    corrected_df, corrected_makespan = analyze_and_correct_trace(original_df, TOTAL_GPUS)
    original_makespan = original_df['End'].max()

    # ... 打印文本结果 (与之前相同) ...
    print("\n--- Analysis Complete ---")
    pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 50); pd.set_option('display.width', 200)
    print(corrected_df.to_string())
    print("\n--- Summary ---")
    print(f"Original Makespan (Communication-Unaware): {original_makespan:.2f} s")
    print(f"Corrected Makespan (Communication-Aware):  {corrected_makespan:.2f} s")
    increase = corrected_makespan - original_makespan
    increase_percent = (increase / original_makespan * 100) if original_makespan > 0 else 0
    print(f"Makespan Increase: {increase:.2f} s ({increase_percent:.2f}%)")

    # 调用新的可视化函数
    visualize_schedule_comparison(corrected_df, original_makespan, corrected_makespan, iter)

    # ... 保存CSV文件 (与之前相同) ...
    output_path = ORIGINAL_TRACE_CSV_PATH.replace(".csv", "_corrected.csv")
    corrected_df.to_csv(output_path, index=False)
    print(f"\n✅ Detailed corrected analysis saved to '{output_path}'")