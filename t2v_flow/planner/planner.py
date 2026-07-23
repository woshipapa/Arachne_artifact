import yaml
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any, Set, Tuple
import torch.distributed as dist
@dataclass
class Task:
    """One task of a schedule plan; fields mirror the YAML entries."""
    name: str
    gpus: List[int]
    dependencies: List[str] = field(default_factory=list)
    args: Dict[str, Any] = field(default_factory=dict)

    task_type: Any = None
    data_source_task: Any = None
    has_overlapping_consumer: bool = False
    # no using
    overlapped: bool = False

    def __post_init__(self):
        """Promote the entries of `args` to instance attributes."""
        protected_attrs = {'name', 'gpus', 'dependencies', 'args'}
        for key, value in self.args.items():
            if key in protected_attrs:
                print(f"Warning: Argument key '{key}' conflicts with a core Task attribute. "
                      f"Access it via task.args['{key}'] instead.")
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        args_repr = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return (f"Task(name='{self.name}', gpus={self.gpus}, "
                f"dependencies={self.dependencies}, args={{ {args_repr} }}, "
                f"has_overlapping_consumer={self.has_overlapping_consumer})")

class Planner:
    """Builds a staged execution plan from the task DAG; constructed from
    task definitions or loaded from a YAML plan."""
    def __init__(self, task_definitions: List[Dict[str, Any]] = None):
        """Build the planner from a raw task-definition list."""
        # if not task_definitions:
        #     raise ValueError("Task definitions cannot be empty.")
        if task_definitions is not None:
            self._tasks: Dict[str, Task] = {
                t['name']: Task(**t) for t in task_definitions
            }
            self._staged_plan: List[List[Task]] = self._generate_staged_plan()
        else:
            self._tasks: Dict[str, Task] = None
            self._staged_plan: List[List[Task]] = None
        self._extra_required_groups = set()

    def set_extra_required_groups(self, groups):
        self._extra_required_groups = set(groups)

    def get_extra_required_groups(self):
        return set(self._extra_required_groups)

    def get_all_required_gpu_groups(self):
        groups = set()
        for t in self._tasks.values():
            groups.add(frozenset(t.gpus))
        groups |= set(self._extra_required_groups)
        return groups    
    

    def get_task_by_name(self, name: str) -> Task:
            """Look up a Task by name."""
            if name not in self._tasks:
                raise ValueError(f"Task with name '{name}' not found in planner.")
            return self._tasks[name]
    def _analyze_dependencies_and_enrich_tasks(self, task_definitions: List[Dict]):
        """Mark each VAE task whose consumer DIT shares GPUs with it
        (sets has_overlapping_consumer for the runtime hand-off)."""
        if not task_definitions:
            return

        task_def_map = {t['name']: t for t in task_definitions}
        consumer_map: Dict[str, List[str]] = {name: [] for name in task_def_map}

        for task_name, task_def in task_def_map.items():
            for dep_name in task_def.get('dependencies', []):
                if dep_name in consumer_map:
                    consumer_map[dep_name].append(task_name)

        for task_def in task_definitions:
            if task_def.get('args', {}).get('task_type') == 'VAE':
                vae_task_name = task_def['name']
                producer_gpus = set(task_def['gpus'])
                
                has_overlap = False
                for consumer_name in consumer_map.get(vae_task_name, []):
                    consumer_def = task_def_map[consumer_name]
                    if consumer_def.get('args', {}).get('data_source_task') == vae_task_name:
                        consumer_gpus = set(consumer_def['gpus'])
                        if not producer_gpus.isdisjoint(consumer_gpus):
                            has_overlap = True
                            break
                
                if 'args' not in task_def:
                    task_def['args'] = {}
                task_def['args']['has_overlapping_consumer'] = has_overlap


    def get_broker_communicators_for_iteration(self) -> Tuple[List[int], List[int]]:
        """Ranks that talk to the broker this iteration, split into
        (publishers, getters)."""
        if not self._tasks:
            return [], []

        publisher_ranks: List[int] = []
        getter_ranks: List[int] = []
        
        for task in self._tasks.values():
            if task.task_type == 'VAE' and not task.has_overlapping_consumer:
                producer_leader = min(task.gpus)
                publisher_ranks.append(producer_leader)

        for task in self._tasks.values():
            if task.task_type in ['DiT', 'DIT'] and task.data_source_task:
                if task.data_source_task not in self._tasks:
                    continue
                
                producer_task = self._tasks[task.data_source_task]
                if producer_task.task_type != 'VAE':
                    continue
                
                producer_gpus = set(producer_task.gpus)
                consumer_gpus = set(task.gpus)

                if producer_gpus.isdisjoint(consumer_gpus):
                    consumer_leader = min(task.gpus)
                    getter_ranks.append(consumer_leader)

        return sorted(publisher_ranks), sorted(getter_ranks)
    def update_staged_plan_per_iter(self, yml_path : str):
        task_definitions = self.from_yaml(yml_path)
        self._analyze_dependencies_and_enrich_tasks(task_definitions)
        self._tasks = {
                t['name']: Task(**t) for t in task_definitions
        }
        self._staged_plan = self._generate_staged_plan()

    
    def from_yaml(self, filepath: str) :

        print(f"Planner: Loading plan from file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Plan file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error: Failed to parse YAML file {filepath}. Reason: {e}")
            raise

        if not data or 'tasks' not in data or not isinstance(data['tasks'], list):
            print(f"data : {data}")
            raise ValueError(f"Invalid plan format in {filepath}. "
                             "It must contain a top-level 'tasks' key with a list of task definitions.")
        
        task_definitions = data['tasks']
        return task_definitions

    def _generate_staged_plan(self) -> List[List[Task]]:
        """Topological staging of the task DAG."""
        in_degree = {name: 0 for name in self._tasks}
        adj_list = {name: [] for name in self._tasks}
        for name, task in self._tasks.items():
            for dep in task.dependencies:
                if dep in self._tasks:
                    in_degree[name] += 1
                    adj_list[dep].append(name)
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        staged_plan = []
        while queue:
            current_stage_tasks = []
            for _ in range(len(queue)):
                task_name = queue.popleft()
                current_stage_tasks.append(self._tasks[task_name])
                for neighbor in adj_list[task_name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            staged_plan.append(current_stage_tasks)
        if len(self._tasks) != sum(len(stage) for stage in staged_plan):
            raise ValueError("Cycle detected in task dependencies! Plan cannot be generated.")
        return staged_plan
    
    def get_staged_plan(self) -> List[List[Task]]:
        return self._staged_plan

    # def get_all_required_gpu_groups(self) -> Set[frozenset]:
    #     unique_groups = set()
    #     for name, task in self._tasks.items():
    #         unique_groups.add(frozenset(task.gpus))
    #     return unique_groups
    
    def get_per_rank_schedules(self, world_size: int = None) -> Dict[int, List[Task]]:
        """Per-rank ordered task lists derived from the staged plan;
        returns Dict[rank, List[Task]]."""
        if world_size is None:
            # default pg 
            world_size = dist.get_world_size()
        schedules: Dict[int, List[Task]] = {rank: [] for rank in range(world_size)}

        staged_plan = self.get_staged_plan()
        for stage in staged_plan:
            for task in stage:
                for rank in task.gpus:
                    schedules[rank].append(task)
        
        return schedules
