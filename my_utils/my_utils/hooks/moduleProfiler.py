import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any

class ModuleProfiler:
    """
    A flexible hook-based performance profiler for PyTorch modules.

    Hook registration is decoupled from the model's actual execution.
    The class doubles as a context manager that registers and removes hooks.

    Usage:
    1. `with ModuleProfiler(model) as profiler:`
    2. In your training/inference loop:
       - `profiler.start()`
       - `model(inputs)`
       - `torch.cuda.synchronize()`
       - `profiler.stop()`
    3. After the loop, call `profiler.summary()` for the results.
    """
    def __init__(self, model: nn.Module):
        if not torch.cuda.is_available():
            raise RuntimeError("ModuleProfiler requires a CUDA-enabled GPU.")
        
        self.model = model
        self._module_events: Dict[str, Dict[str, torch.cuda.Event]] = {}
        self.module_timings: Dict[str, List[float]] = defaultdict(list)
        self._hook_handles: List[Any] = []
        self._is_profiling: bool = False
        
        self._register_hooks()

    def _register_hooks(self):
        """Register forward pre/post hooks on every leaf module."""
        for name, module in self.model.named_modules():
            if not list(module.children()): # only leaf modules
                self._module_events[name] = {
                    'start': torch.cuda.Event(enable_timing=True),
                    'end': torch.cuda.Event(enable_timing=True)
                }
                
                pre_hook_handle = module.register_forward_pre_hook(self._pre_hook_factory(name))
                post_hook_handle = module.register_forward_hook(self._post_hook_factory(name))
                self._hook_handles.extend([pre_hook_handle, post_hook_handle])

    def _pre_hook_factory(self, name: str):
        def pre_hook(module, input):
            if self._is_profiling:
                # print(f"{name} register pre_hook    record---------------------")
                self._module_events[name]['start'].record()
        return pre_hook

    def _post_hook_factory(self, name: str):
        def post_hook(module, input, output):
            if self._is_profiling:
                # print(f"{name} register post_hook   record---------------------")
                self._module_events[name]['end'].record()
        return post_hook

    def start(self):
        """Start a timing window. Call before the model's forward()."""
        self._is_profiling = True

    def stop(self):
        """
        End a timing window and record the measurement.
        Should be called after `torch.cuda.synchronize()`.
        """
        if not self._is_profiling:
            return
        torch.cuda.synchronize()
        for name, events in self._module_events.items():
            # only count entries where both start and end were recorded
            try:
                if events['start'].query() and events['end'].query():
                    elapsed_time_ms = events['start'].elapsed_time(events['end'])
                    self.module_timings[name].append(elapsed_time_ms)
            except Exception as e:
                print(f"module {name} meets {e}")        
        
        self._is_profiling = False

    def summary(self, output_path: str = None) -> pd.DataFrame:
        """Compute statistics once timing is finished and return a DataFrame."""
        if not self.module_timings:
            print("⚠️ Warning: No timing data was collected.")
            return pd.DataFrame()

        results = []
        total_model_time = sum(sum(times) for times in self.module_timings.values())

        if total_model_time == 0:
            print("⚠️ Warning: Total model time is zero. Cannot compute percentages.")

        for name, timings in self.module_timings.items():
            if not timings: continue
            
            total_time = sum(timings)
            percentage = (total_time / total_model_time * 100) if total_model_time > 0 else 0
            
            results.append({
                'module_name': name,
                'mean_ms': pd.Series(timings).mean(),
                'median_ms': pd.Series(timings).median(),
                'std_ms': pd.Series(timings).std(),
                'total_ms': total_time,
                'run_count': len(timings),
                'percentage': percentage
            })

        df = pd.DataFrame(results).sort_values(by='mean_ms', ascending=False).reset_index(drop=True)
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Summary saved to: {output_path}")
        
        return df
        
    def cleanup(self):
        """Remove every registered hook."""
        for handle in self._hook_handles:
            handle.remove()
        print("✅ All hooks have been removed.")

    def __enter__(self):
        """Context-manager entry; returns the profiler instance."""
        print("🚀 Profiler activated. Hooks are registered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context-manager exit; removes the hooks."""
        self.cleanup()