import os
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity
from contextlib import contextmanager
import logging
import json
from typing import List, Union, Dict, Optional
import re
from collections import defaultdict
# It's good practice for a library module to not configure the root logger.
# It should use its own logger. The calling application can configure logging.
logger = logging.getLogger(__name__)

class FlexibleProfiler:
    """
    An internal class managing the profiling lifecycle and analysis.
    It's designed to be instantiated and used by the create_profiler_context factory.
    """
    def __init__(self,
                 is_enabled: bool,
                 output_dir: str,
                 logger_instance,
                 task_type: str ,
                 ops_to_analyze: Optional[Dict[str, str]] = None):
        
        self.is_enabled = is_enabled
        self.output_dir = output_dir
        self.logger = logger_instance
        self.ops_to_analyze = ops_to_analyze or {}
        self.profiler = None
        self.task_type = task_type

        if self.is_enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.logger.info(f"INFO: Profiler is ENABLED for Rank {self.rank}.")

# inside the FlexibleProfiler class
    def _trace_handler(self, p):
        """Saves the standard Chrome trace and summary table."""

        # --- conditional branch ---
        if self.task_type == "VAE":
            self.logger.info(f"Task type is VAE; skipping JSON trace export (rank {self.rank}).")
        else:
            # export the JSON trace only for non-VAE task types
            trace_file = os.path.join(self.output_dir, f"torch_prof_rank{self.rank}.json")
            p.export_chrome_trace(trace_file)
            self.logger.info(f"Profiler trace for rank {self.rank} saved to {trace_file}")

        # --- summary generation is unaffected ---
        summary_file = os.path.join(self.output_dir, f'prof_summary_rank{self.rank}.txt')
        with open(summary_file, 'w') as f:
            f.write(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
        self.logger.info(f"Profiler summary for rank {self.rank} saved to {summary_file}")

    def __enter__(self):
        if not self.is_enabled:
            return self

        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_enabled or self.profiler is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # --- NEW: Perform and persist detailed analysis upon exit ---
        if self.ops_to_analyze:
            self._analyze_and_persist_ops()

    def _get_op_metrics(self, operation_name: str, metric: str) -> List[Union[int, float]]:
        """Internal method to extract metrics for a single operation."""
        all_events = self.profiler.events()
        metric_values = [getattr(event, metric, 0) for event in all_events if operation_name in event.name]
        return metric_values
        
    def _analyze_and_persist_ops(self):
        """
        Analyse the configured operations, supporting regular expressions, and report
        """
        self.logger.info(f"Performing detailed analysis (grouped by unique name) for specified ops on Rank {self.rank}...")

        all_events = self.profiler.events()
        
        # =================================================================
        # Phase 1: collect every event matching a pattern, grouped by its full unique name
        # =================================================================
        # shape: {"tile_encoder_0": {"metric": "...", "events": [...]}, "tile_encoder_1": ...}
        events_grouped_by_name = defaultdict(lambda: {"metric": "", "events": []})

        # iterate the configured patterns (e.g. r"tile_encoder_\d+")
        for pattern, metric in self.ops_to_analyze.items():
            # iterate every profiler event
            for event in all_events:
                # if the event name matches the pattern
                if re.match(pattern, event.name):
                    full_event_name = event.name  # the full name, e.g. "tile_encoder_0"
                    # key the group by that full name
                    events_grouped_by_name[full_event_name]["metric"] = metric
                    events_grouped_by_name[full_event_name]["events"].append(event)

        if not events_grouped_by_name:
            self.logger.warning("No events found matching the specified patterns.")
            return

        # =================================================================
        # Phase 2: walk the groups and compute statistics for each one independently
        # =================================================================
        final_analysis = {}
        
        for full_name, group_data in events_grouped_by_name.items():
            metric_to_use = group_data["metric"]
            events_list = group_data["events"]
            
            all_values_us = [getattr(event, metric_to_use, 0) for event in events_list]
            non_zero_values_us = [v for v in all_values_us if v > 0]

            if non_zero_values_us:
                count = len(non_zero_values_us)
                total_us = sum(non_zero_values_us)
                average_us = total_us / count
                
                total_ms = total_us / 1000.0
                average_ms = average_us / 1000.0
                values_ms = [v / 1000.0 for v in non_zero_values_us]

                final_analysis[full_name] = {
                    "metric": metric_to_use,
                    "count": count,
                    "average_ms": average_ms,
                    "total_ms": total_ms,
                    "values_ms": values_ms
                }
                self.logger.info(f"  - Analyzed '{full_name}': Found {count} instances, Avg: {average_ms:.3f} ms.")
            else:
                self.logger.warning(f"  - For '{full_name}', all recorded instances had a zero value for metric '{metric_to_use}'.")
        
        # =================================================================
        # Phase 3: persist the result
        # =================================================================
        if final_analysis:
            output_file = os.path.join(self.output_dir, f"detailed_metrics_rank_{self.rank}.json")
            with open(output_file, 'w') as f:
                json.dump(final_analysis, f, indent=4, sort_keys=True) # sorting by key reads better
            self.logger.info(f"Detailed analysis saved to {output_file}")

@contextmanager
def create_profiler_context(current_task_type: str,
                            logger,
                            enabled_env_var: str = "PROFILE_TASK_TYPES",
                            base_output_dir: str = "prof_results",
                            ops_to_analyze: Optional[Dict[str, str]] = None):
    """
    Factory that builds a profiling context for a given task type.

    When the environment names the current task type, an active profiler is returned;
    on exit it saves the summary, the trace file and the per-operation analysis.

    Args:
        current_task_type (str): the task type being executed (e.g. "DIT", "VAE").
        logger: logger instance.
        enabled_env_var (str): environment variable naming the task types to profile.
                               Its value is comma separated, e.g. "DIT,VAE".
        base_output_dir (str): base directory for all profiling output.
        ops_to_analyze (dict): operations to analyse in detail.
    """
    # read the list of task types to profile from the environment
    profilable_tasks_str = os.environ.get(enabled_env_var, "")
    profilable_tasks = [t.strip() for t in profilable_tasks_str.split(',') if t.strip()]

    # is the current task type in that list?
    is_profiling_enabled = current_task_type in profilable_tasks

    if not is_profiling_enabled:
        # not enabled: return a context that does nothing
        yield
        return

    # give each task type its own output directory
    task_specific_output_dir = os.path.join(base_output_dir, current_task_type)

    # create and yield the profiler instance
    profiler_instance = FlexibleProfiler(
        is_enabled=True,
        output_dir=task_specific_output_dir, # task-specific directory
        logger_instance=logger,
        task_type = current_task_type,
        ops_to_analyze=ops_to_analyze
    )
    
    with profiler_instance:
        yield