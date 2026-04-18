import json
import math
from collections import defaultdict
import itertools # 我们将需要它来寻找组合
from functools import lru_cache
class TopologyModel:
    """
    封装了所有与GPU拓扑、通信重叠度和性能惩罚相关的计算。
    """
    def __init__(self, params_file: str , machine_size: int = 8, enable_topology: bool = True):
            if machine_size <= 0:
                raise ValueError(f"machine_size must be positive, got {machine_size}")
            self.enable_topology = enable_topology
            self.machine_size = machine_size

            # Extend NIC group mapping to cover all local GPU ids, not just 0..7.
            # This prevents GPUs with local_id >= 8 from being treated as "unknown".
            base_pattern = ["A", "A", "B", "B", "C", "C", "D", "D"]
            self.local_group = {i: base_pattern[i % len(base_pattern)] for i in range(self.machine_size)}
            # 新增：定义NIC组到NUMA节点的映射
            self.nic_to_numa = {
                'A': 0, 'B': 0, # A, B 在 NUMA 0
                'C': 1, 'D': 1  # C, D 在 NUMA 1
            }
            self.nic_bit = {
                'A': 1 << 0,
                'B': 1 << 1,
                'C': 1 << 2,
                'D': 1 << 3,
            }
            # 使用我们之前确认的默认参数
            self.default_params = {'A': 0.21138606119414527, 'B': 1.4790472808157773}
            self.fit_params = {8: self.default_params}
            print(f"✅ TopologyModel v2 initialized with NUMA awareness.")

    def _gpu_to_group_label(self, global_gpu: int) -> str:
        local_id = global_gpu % self.machine_size
        return self.local_group[local_id]
    def _gpu_nic_mask(self, global_gpu: int) -> int:
            """
            返回该 GPU 所属 NIC group 的 bitmask
            """
            local_id = global_gpu % self.machine_size
            group = self.local_group[local_id]
            return self.nic_bit[group]
    def _groups_from_gpu_list(self, gpu_list: list) -> list:
        return sorted({self._gpu_to_group_label(g) for g in gpu_list if self._gpu_to_group_label(g) is not None})

    def _overlap_score_sets(self, groups_a: list, groups_b: list) -> float:
        set_a, set_b = set(groups_a), set(groups_b)
        if not set_a or not set_b: return 0.0
        return float(len(set_a.intersection(set_b))) / float(max(len(set_a), len(set_b)))
    
    # def calculate_internal_overlap(self, gpu_list: list) -> float:
    #     """计算给定GPU列表的内部跨机NIC亲和性重叠度。"""
    #     if not gpu_list: return 1.0
    #     machine_gpus = defaultdict(list)
    #     for gpu_id in gpu_list:
    #         machine_gpus[gpu_id // self.machine_size].append(gpu_id)

    #     if len(machine_gpus) <= 1: return 1.0

    #     machine_nic_groups = [self._groups_from_gpu_list(gpus) for gpus in machine_gpus.values()]
        
    #     if len(machine_nic_groups) == 2:
    #         return self._overlap_score_sets(machine_nic_groups[0], machine_nic_groups[1])
    #     else:
    #         min_score = 1.0
    #         for g1, g2 in itertools.combinations(machine_nic_groups, 2):
    #             score = self._overlap_score_sets(g1, g2)
    #             if score < min_score: min_score = score
    #         return min_score



    def calculate_internal_overlap(self, gpu_list: list) -> float:
        """
        【已重构，支持N节点】计算给定GPU列表的内部跨机NIC亲和性重叠度。
        
        新逻辑 (总交集模型):
        1. 找出所有参与节点的NIC Group集合。
        2. 计算这些集合的“总交集”。
        3. 重叠度分数定义为：总交集的大小 / 所有参与节点中最大的NIC Group集合的大小。
           这个公式是双节点 overlap 计算逻辑的直接泛化。
        """
        if not gpu_list: return 1.0
        
        # 步骤 A: 按物理机对GPU进行分组
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)

        # 如果任务在单个节点内，则重叠度为完美
        if len(machine_gpus) <= 1: 
            return 1.0

        # 步骤 B: 获取每个节点的NIC Group集合
        machine_nic_groups = [
            set(self._groups_from_gpu_list(gpus)) 
            for gpus in machine_gpus.values()
        ]

        # 步骤 C: 计算所有节点的“总交集”
        # 从第一个节点的group集合开始
        global_intersection = machine_nic_groups[0].copy()
        # 依次与后续所有节点的group集合取交集
        for i in range(1, len(machine_nic_groups)):
            global_intersection.intersection_update(machine_nic_groups[i])
        
        # 步骤 D: 计算最终的Overlap Score
        # 找到所有节点中，所用NIC Group集合的最大尺寸
        max_group_size = max(len(g) for g in machine_nic_groups) if machine_nic_groups else 0
        
        if max_group_size == 0: 
            return 1.0  # 避免除零错误, 理论上不会发生

        # 公式: |A ∩ B ∩ C| / max(|A|, |B|, |C|)
        return float(len(global_intersection)) / float(max_group_size)
    
    @lru_cache(maxsize=2048)
    def predict_penalty(self, gpu_list: tuple) -> float:
        """
        【V3 - 最终修正版】根据给定的GPU列表，预测通信惩罚系数。
        
        修正逻辑:
        1. 首先判断GPU列表是否跨越多个物理机。
        2. 如果是节点内通信 (只涉及一台物理机)，则为真正基准，惩罚系数严格为 1.0。
        3. 如果是跨节点通信 (涉及多台物理机)，则无论 overlap 是多少，
        都必须应用指数惩罚模型。这确保了即便是完美对齐的跨机通信，
        其惩罚也大于1.0，以体现基础的跨网络开销。
        """
        if not self.enable_topology:
            return 1.0
        gpu_list = list(gpu_list)  # 转回列表以便处理
        n_gpus = len(gpu_list)
        if n_gpus <= 1:
            return 1.0
            
        # 1. 判断是否为节点内通信
        # 通过将GPU按物理机分组，检查涉及的物理机数量
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)
        
        # 如果只涉及一台物理机，这就是基准情况，直接返回无惩罚
        if len(machine_gpus) <= 1:
            return 1.0
            
        # 2. 如果是跨节点通信，则应用惩罚模型
        # 此时，我们不再需要单独检查 overlap == 1.0 的情况
        
        overlap = self.calculate_internal_overlap(gpu_list)
        
        # 使用 .get(key, default_value) 的方式获取参数
        params = self.fit_params.get(n_gpus, self.default_params)
        
        A, B = params['A'], params['B']
        
        penalty = 1.0 + A * math.exp(-B * overlap)
        
        return penalty
    

    def _get_sum_dist_to_intersection(self, nic: str, intersection: set) -> int:
        """
        【已修改】辅助函数：计算单个“游离”NIC到“核心”NIC集合(交集)的NUMA距离总和。
        """
        if not intersection:
            return float('inf')
            
        total_dist = 0
        for nic_in_intersection in intersection:
            # --- 核心逻辑修改：从 min 改为 sum ---
            # 累加当前游离NIC到每一个核心NIC的距离
            total_dist += 1 if self.nic_to_numa[nic] == self.nic_to_numa[nic_in_intersection] else 2
            
        return total_dist
    

    @lru_cache(maxsize=2048)
    def calculate_numa_distance(self, gpu_list: tuple) -> float:
        """
        【已重构，支持N节点】计算一个跨机GPU列表的NUMA距离。
        
        新逻辑:
        1. 找出所有参与节点的NIC Group的“总交集”。
        2. 如果总交集为空，意味着没有通用的最优通信路径，返回无穷大惩罚。
        3. 对每个节点，计算其“非交集”NIC到这个“总交集”的距离总和。
        4. 将所有节点的距离总和相加，作为最终结果。
        """
        # 步骤 1: 按物理机对GPU进行分组
        gpu_list = list(gpu_list)
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)
        
        # 如果任务在单个节点内或没有GPU，则没有跨机NUMA成本
        if len(machine_gpus) <= 1:
            return 0.0

        # 步骤 2: 获取每个节点的NIC Group集合
        machine_nic_groups = [
            set(self._groups_from_gpu_list(gpus)) 
            for gpus in machine_gpus.values()
        ]

        # 步骤 3: 计算所有节点的“总交集”
        # 从第一个节点的group集合开始
        global_intersection = machine_nic_groups[0].copy()
        # 依次与后续所有节点的group集合取交集
        for i in range(1, len(machine_nic_groups)):
            global_intersection.intersection_update(machine_nic_groups[i])
        
        # 步骤 4: 如果总交集为空，则这是一个非常差的分配，给予最大惩罚
        if not global_intersection:
            return float('inf')

        # 步骤 5: 计算每个节点非交集部分到总交集的距离，并累加
        total_distance = 0.0
        for node_groups in machine_nic_groups:
            # 找出当前节点独有的（非交集的）NICs
            non_intersection_nics = node_groups - global_intersection
            
            # 对每个独有的NIC，计算它到总交集的距离之和
            for nic in non_intersection_nics:
                total_distance += self._get_sum_dist_to_intersection(nic, global_intersection)
                
        return total_distance
    


    @lru_cache(maxsize=200_000)
    def score(self, gpu_tuple: tuple):
        """
        返回 (penalty, numa_distance)
        —— 合并 predict_penalty + calculate_numa_distance
        —— cache key = gpu_tuple
        """
        if not self.enable_topology:
            return 1.0, 0.0
        n_gpus = len(gpu_tuple)
        if n_gpus <= 1:
            return 1.0, 0.0

        # ---- 按物理机分组 ----
        machine_gpus = defaultdict(list)
        for g in gpu_tuple:
            machine_gpus[g // self.machine_size].append(g)

        # 单节点：基准
        if len(machine_gpus) <= 1:
            return 1.0, 0.0

        # ---- 每节点 NIC group ----
        machine_groups = [
            set(self._groups_from_gpu_list(gpus))
            for gpus in machine_gpus.values()
        ]

        # ---- 计算全局交集 ----
        global_intersection = machine_groups[0].copy()
        for gset in machine_groups[1:]:
            global_intersection.intersection_update(gset)

        # ---- overlap ----
        max_group_size = max(len(g) for g in machine_groups) or 1
        overlap = len(global_intersection) / max_group_size

        # ---- penalty ----
        params = self.fit_params.get(n_gpus, self.default_params)
        A, B = params['A'], params['B']
        penalty = 1.0 + A * math.exp(-B * overlap)

        # ---- NUMA distance ----
        if not global_intersection:
            numa_distance = float('inf')
        else:
            total_dist = 0.0
            for node_groups in machine_groups:
                for nic in (node_groups - global_intersection):
                    total_dist += self._get_sum_dist_to_intersection(
                        nic, global_intersection
                    )
            numa_distance = total_dist

        return penalty, numa_distance
    

    @lru_cache(maxsize=300_000)
    def score_bitmask(self, gpu_tuple: tuple):
        """
        Bitmask 加速版 topology score
        返回 (penalty, numa_distance)
        与 score() 语义完全一致，但无 set / dict / list
        """

        if not self.enable_topology:
            return 1.0, 0.0
        n_gpus = len(gpu_tuple)
        if n_gpus <= 1:
            return 1.0, 0.0

        # ---- Step 1: 每个 machine 的 NIC mask ----
        machine_mask = {}
        for g in gpu_tuple:
            mid = g // self.machine_size
            mask = self._gpu_nic_mask(g)
            machine_mask[mid] = machine_mask.get(mid, 0) | mask

        # 单节点：基准
        if len(machine_mask) <= 1:
            return 1.0, 0.0

        masks = list(machine_mask.values())

        # ---- Step 2: 全局交集 ----
        global_intersection = masks[0]
        for m in masks[1:]:
            global_intersection &= m

        # ---- Step 3: overlap ----
        max_group_size = max(m.bit_count() for m in masks) or 1
        overlap = global_intersection.bit_count() / max_group_size

        # ---- Step 4: penalty ----
        params = self.fit_params.get(n_gpus, self.default_params)
        A, B = params['A'], params['B']
        penalty = 1.0 + A * math.exp(-B * overlap)

        # ---- Step 5: NUMA distance ----
        if global_intersection == 0:
            numa_distance = float('inf')
        else:
            numa_distance = 0.0
            for mask in masks:
                diff = mask & ~global_intersection  # 非交集 NIC
                while diff:
                    bit = diff & -diff
                    diff -= bit

                    nic = next(k for k, v in self.nic_bit.items() if v == bit)
                    for inter_bit in self.nic_bit.values():
                        if global_intersection & inter_bit:
                            inter_nic = next(k for k, v in self.nic_bit.items() if v == inter_bit)
                            numa_distance += (
                                1 if self.nic_to_numa[nic] == self.nic_to_numa[inter_nic] else 2
                            )

        return penalty, numa_distance