# t2v_flow/planner

Planning and scheduling logic plus schedule generation artifacts.

## Core modules
- planner.py: Parse task YAML, build the task graph, and generate per-rank schedules.
- TopologyModel.py: Model GPU topology, NUMA distance, and overlap penalties for scoring schedules.
- TopologyModel.md: Design notes and assumptions for the topology model.
- scheduler.py: Main scheduler implementations (greedy, A*, genetic, trace export, plotting, and topology-aware selection).
- scheduler1.py: Earlier scheduler variant and baseline algorithms.
- schedule_pool.py: Batch scheduling pipeline that loads predictors and emits schedule_*.yaml and *_tasks.json.
- schedule_pool1.py: Earlier schedule_pool variant that uses scheduler1.
- analyze_schedule_trace.py: Replay trace CSVs, adjust timing, and visualize timelines.
- best_comm_aware_schedule.py: Communication-aware scheduling with NIC affinity and GA search.
- new_communication_aware_schedule.py: Updated communication-aware scheduler and evaluation flow.
- dag_assign_gpu.py: Topology-aware GPU assignment for SP tasks and GA search.
- evaluate_genetic_schedule.py: Evaluate GA schedule quality, critical path, and comm cost.
- debug_genetic_chromosome.py: Verbose debugging for a single GA chromosome schedule.
- flexsp_scheduler.py: Greedy FlexSP scheduler that chooses SP per task and emits traces.
- megatron_scheduler.py: Baseline Megatron-LM style uniform SP scheduling.
- generate_vae_yaml.py: Generate VAE-only task YAMLs for a given SP and GPU count.
- mock_utils.py: Helpers to create fixed SP task YAMLs for testing.
- old_find_gpu_schedule.py: Older GPU selection logic kept for reference.
- scheduler_pool.md: Notes for schedule pool usage.
- schedule_comparison.png: Schedule comparison plot output.
- schedule_comparison_0.png: Schedule comparison plot for variant 0.
- schedule_comparison_4.png: Schedule comparison plot for variant 4.
- __init__.py: Package marker.

## Subdirectories
- generated_schedules: Generated schedule outputs and visualization utilities.
- task_yamls: Task graphs and configuration YAMLs.
