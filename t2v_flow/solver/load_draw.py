import json
import matplotlib

# ------------------- CRITICAL FIX -------------------
# Set the backend before importing matplotlib.pyplot.
# 'Agg' is a non-interactive backend for writing to files, it doesn't need a GUI.
matplotlib.use('Agg')
# ----------------------------------------------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def plot_rank_timeline_with_completion_time(prefix = "",data_file="schedule_data.json"):
    """
    Plots a rank allocation timeline, including the final completion time.
    """
    print("--- [Debug] 1. Function execution started ---")

    # 1. Load data from the JSON file
    try:
        data_file = prefix + data_file
        print(f"--- [Debug] 2. Attempting to open and read file: {data_file} ---")
        with open(data_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"--- [Debug] 3. File read successfully. Loaded {len(tasks)} tasks. ---")
    except Exception as e:
        print(f"--- [Error] An error occurred in step 2 or 3: {e} ---")
        return

    # ------------------- NEW STEP -------------------
    # Calculate the final completion time by finding the maximum 'end' time.
    if not tasks:
        final_completion_time = 0
    else:
        final_completion_time = max(task['end'] for task in tasks)
    print(f"--- [Debug] 3.5. Calculated final completion time: {final_completion_time:.2f}s ---")
    # ------------------------------------------------

    # 2. Prepare the plot
    try:
        print("--- [Debug] 4. Creating figure and axes (plt.subplots) ---")
        fig, ax = plt.subplots(figsize=(16, 8))
        print("--- [Debug] 5. Plot setup is ready. ---")
    except Exception as e:
        print(f"--- [Error] An error occurred in step 4 or 5: {e} ---")
        return

    # ... (Steps 6 to 9 for color mapping and drawing bars remain the same) ...
    print("--- [Debug] 6. Generating color map for tasks ---")
    task_names = sorted(list(set(item['name'] for item in tasks)))
    colors = cm.get_cmap('tab20', len(task_names))
    color_map = {name: colors(i) for i, name in enumerate(task_names)}
    print(f"--- [Debug] 7. Color map created for {len(task_names)} unique tasks. ---")
    print("--- [Debug] 8. Starting to loop and draw task bars... ---")
    for task in tasks:
        # ... (bar drawing logic is unchanged) ...
        ax.barh(y=task['ranks'], width=(task['end'] - task['start']), left=task['start'], height=0.6,
                align='center', color=color_map[task['name']], edgecolor='grey')
    print("--- [Debug] 9. All task bars have been drawn. ---")


    # 5. Set plot styles and labels
    print("--- [Debug] 10. Setting plot labels, title, and ticks... ---")
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_title('Task Allocation on Ranks Over Time', fontsize=16)
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(8))
    ax.invert_yaxis()

    # Create the legend
    print("--- [Debug] 11. Creating legend... ---")
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[name]) for name in task_names]
    ax.legend(legend_handles, task_names, title='Task Names', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # ------------------- NEW STEP -------------------
    # Add the final completion time text to the plot.
    # We place it in the top right area of the plot for visibility.
    print("--- [Debug] 11.5. Adding completion time text to the plot... ---")
    ax.text(final_completion_time, 0.5, # x, y coordinates
            f'Final Completion Time: {final_completion_time:.2f} s',
            fontsize=12,
            fontweight='bold',
            color='crimson',
            ha='right', # Horizontal alignment
            va='bottom') # Vertical alignment
    # ------------------------------------------------

    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    print("--- [Debug] 12. Adjusting layout (tight_layout)... ---")
    # Adjust the x-axis limit to make sure the new text is visible
    ax.set_xlim(right=ax.get_xlim()[1] * 1.05)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # 6. Save the plot to a file
    output_filename = prefix + "rank_timeline_final.png"
    print(f"--- [Debug] 13. Preparing to save the chart to file: {output_filename} ---")
    plt.savefig(output_filename)
    print(f"--- [Debug] 14. Chart successfully saved to: {output_filename} ---")
    
    print("--- [Debug] 15. plt.show() will be skipped or non-blocking in 'Agg' backend. ---")
    # plt.show()
    
    print("--- [Debug] 16. Script finished successfully. ---")


# --- Main execution block ---
if __name__ == "__main__":
    prefix = "2222_"
    plot_rank_timeline_with_completion_time(prefix= prefix,data_file="schedule_data.json")