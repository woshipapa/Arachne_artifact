import yaml
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any, Set, Tuple
import torch.distributed as dist
# Task 类保持不变
@dataclass
class Task:
    """
    一个用于存储和访问调度计划中单个任务信息的数据类。
    """
    # --- 核心属性，直接从YAML的顶层键映射 ---
    name: str
    gpus: List[int]
    dependencies: List[str] = field(default_factory=list)
    args: Dict[str, Any] = field(default_factory=dict)

    # --- 动态填充的属性 ---
    # 这两个属性将在 Planner 分析后被动态设置
    task_type: Any = None
    data_source_task: Any = None
    has_overlapping_consumer: bool = False
    # no using
    overlapped: bool = False

    def __post_init__(self):
        """
        在dataclass完成标准初始化后自动调用的方法。
        我们用它来动态地将 `args` 字典的内容设置为类的属性。
        """
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
    """
    根据任务的DAG（有向无环图）定义，生成一个分阶段的执行计划。
    可以通过传入任务定义列表来初始化，或直接从YAML文件加载。
    """
    def __init__(self, task_definitions: List[Dict[str, Any]] = None):
        """
        标准构造函数。
        Args:
            task_definitions (List[Dict]): 原始的任务定义列表。
        """
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
            """根据任务名查找并返回Task对象。"""
            if name not in self._tasks:
                raise ValueError(f"Task with name '{name}' not found in planner.")
            return self._tasks[name]
    def _analyze_dependencies_and_enrich_tasks(self, task_definitions: List[Dict]):
        """
        [新功能] 分析任务依赖，并为 VAE 任务自动填充 has_overlapping_consumer 属性。
        """
        if not task_definitions:
            return

        task_def_map = {t['name']: t for t in task_definitions}
        consumer_map: Dict[str, List[str]] = {name: [] for name in task_def_map}

        # 1. 构建消费者映射 (谁依赖于我)
        for task_name, task_def in task_def_map.items():
            for dep_name in task_def.get('dependencies', []):
                if dep_name in consumer_map:
                    consumer_map[dep_name].append(task_name)

        # 2. 遍历所有任务定义，为 VAE 任务计算 overlap 属性
        for task_def in task_definitions:
            if task_def.get('args', {}).get('task_type') == 'VAE':
                vae_task_name = task_def['name']
                producer_gpus = set(task_def['gpus'])
                
                has_overlap = False
                # 查找所有消费这个 VAE 结果的任务
                for consumer_name in consumer_map.get(vae_task_name, []):
                    consumer_def = task_def_map[consumer_name]
                    # 关键检查：只关心那些以此 VAE 为主要数据源的消费者
                    if consumer_def.get('args', {}).get('data_source_task') == vae_task_name:
                        consumer_gpus = set(consumer_def['gpus'])
                        if not producer_gpus.isdisjoint(consumer_gpus):
                            has_overlap = True
                            break # 找到一个重叠的就足够了
                
                # 动态地设置或覆盖 args 中的属性
                if 'args' not in task_def:
                    task_def['args'] = {}
                task_def['args']['has_overlapping_consumer'] = has_overlap


    def get_broker_communicators_for_iteration(self) -> Tuple[List[int], List[int]]:
        """
        [新功能] 获取在本轮迭代中所有会与 Broker 通信的 rank 列表，
        并将发布者和获取者分开返回。

        Returns:
            Tuple[List[int], List[int]]: 一个元组，包含两个列表。
                                         第一个列表是所有需要“发布”数据的 rank (publishers)。
                                         第二个列表是所有需要“获取”数据的 rank (getters)。
        """
        if not self._tasks:
            return [], []

        publisher_ranks: List[int] = []
        getter_ranks: List[int] = []
        
        # 1. 查找所有需要“发布”的 rank (生产者)
        for task in self._tasks.values():
            if task.task_type == 'VAE' and not task.has_overlapping_consumer:
                # 这是一个需要发布到 Broker 的 VAE 任务
                producer_leader = min(task.gpus)
                publisher_ranks.append(producer_leader)

        # 2. 查找所有需要“获取”的 rank (消费者)
        for task in self._tasks.values():
            if task.task_type in ['DiT', 'DIT'] and task.data_source_task:
                if task.data_source_task not in self._tasks:
                    continue
                
                producer_task = self._tasks[task.data_source_task]
                # 确保生产者是VAE类型
                if producer_task.task_type != 'VAE':
                    continue
                
                producer_gpus = set(producer_task.gpus)
                consumer_gpus = set(task.gpus)

                if producer_gpus.isdisjoint(consumer_gpus):
                    # 这是一个需要从 Broker 获取数据的 DIT 任务
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
        # 调用标准的__init__方法来完成实例的创建
        return task_definitions

    def _generate_staged_plan(self) -> List[List[Task]]:
        """拓扑排序算法，保持不变"""
        # ... (此函数的实现与上一回答完全相同)
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
    
    # get_staged_plan 和 get_all_required_gpu_groups 方法保持不变
    def get_staged_plan(self) -> List[List[Task]]:
        return self._staged_plan

    # def get_all_required_gpu_groups(self) -> Set[frozenset]:
    #     unique_groups = set()
    #     for name, task in self._tasks.items():
    #         unique_groups.add(frozenset(task.gpus))
    #     return unique_groups
    
    def get_per_rank_schedules(self, world_size: int = None) -> Dict[int, List[Task]]:
        """
        [新功能] 解析完整的阶段性计划，为每个rank生成其专属的任务执行序列。
        
        Args:
            world_size (int): 分布式环境中的总进程数。

        Returns:
            Dict[int, List[Task]]: 一个字典，键是rank号，值是该rank需要执行的任务列表。
        """
        if world_size is None:
            # default pg 
            world_size = dist.get_world_size()
        # 为每个rank初始化一个空的任务列表
        schedules: Dict[int, List[Task]] = {rank: [] for rank in range(world_size)}

        # 遍历已经拓扑排序好的阶段性计划
        staged_plan = self.get_staged_plan()
        for stage in staged_plan:
            for task in stage:
                # 将这个任务添加到其所有参与者的个人日程中
                for rank in task.gpus:
                    schedules[rank].append(task)
        
        return schedules
