import json
import math
from collections import defaultdict
import itertools
from functools import lru_cache
class TopologyModel:
    """GPU topology model: NIC-affinity overlap, communication penalty and
    NUMA distance for candidate GPU groups."""
    def __init__(self, params_file: str , machine_size: int = 8, enable_topology: bool = True):
            if machine_size <= 0:
                raise ValueError(f"machine_size must be positive, got {machine_size}")
            self.enable_topology = enable_topology
            self.machine_size = machine_size

            # Extend NIC group mapping to cover all local GPU ids, not just 0..7.
            # This prevents GPUs with local_id >= 8 from being treated as "unknown".
            base_pattern = ["A", "A", "B", "B", "C", "C", "D", "D"]
            self.local_group = {i: base_pattern[i % len(base_pattern)] for i in range(self.machine_size)}
            self.nic_to_numa = {
                'A': 0, 'B': 0,  # A, B on NUMA 0
                'C': 1, 'D': 1   # C, D on NUMA 1
            }
            self.nic_bit = {
                'A': 1 << 0,
                'B': 1 << 1,
                'C': 1 << 2,
                'D': 1 << 3,
            }
            self.default_params = {'A': 0.21138606119414527, 'B': 1.4790472808157773}
            self.fit_params = {8: self.default_params}
            print(f"✅ TopologyModel v2 initialized with NUMA awareness.")

    def _gpu_to_group_label(self, global_gpu: int) -> str:
        local_id = global_gpu % self.machine_size
        return self.local_group[local_id]
    def _gpu_nic_mask(self, global_gpu: int) -> int:
            """Bitmask of the GPU's NIC group."""
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
        """Cross-node NIC-affinity overlap of a GPU group (N-node generalization):
        overlap = |intersection of all nodes' NIC-group sets| /
                  max over nodes of |NIC-group set|;
        single-node groups score a perfect 1.0."""
        if not gpu_list: return 1.0
        
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)

        if len(machine_gpus) <= 1: 
            return 1.0

        machine_nic_groups = [
            set(self._groups_from_gpu_list(gpus)) 
            for gpus in machine_gpus.values()
        ]

        global_intersection = machine_nic_groups[0].copy()
        for i in range(1, len(machine_nic_groups)):
            global_intersection.intersection_update(machine_nic_groups[i])
        
        max_group_size = max(len(g) for g in machine_nic_groups) if machine_nic_groups else 0
        
        if max_group_size == 0: 
            return 1.0  # guard against div-by-zero

        return float(len(global_intersection)) / float(max_group_size)
    
    @lru_cache(maxsize=2048)
    def predict_penalty(self, gpu_list: tuple) -> float:
        """Communication-penalty factor for a GPU group:
        - intra-node group -> exactly 1.0 (the baseline);
        - cross-node group -> 1 + A * exp(-B * overlap), so even a
          perfectly NIC-aligned cross-node group pays the base network cost."""
        if not self.enable_topology:
            return 1.0
        gpu_list = list(gpu_list)
        n_gpus = len(gpu_list)
        if n_gpus <= 1:
            return 1.0
            
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)
        
        if len(machine_gpus) <= 1:
            return 1.0
            
        
        overlap = self.calculate_internal_overlap(gpu_list)
        
        params = self.fit_params.get(n_gpus, self.default_params)
        
        A, B = params['A'], params['B']
        
        penalty = 1.0 + A * math.exp(-B * overlap)
        
        return penalty
    

    def _get_sum_dist_to_intersection(self, nic: str, intersection: set) -> int:
        """Sum of NUMA distances from one non-intersection NIC to every NIC
        in the core (intersection) set."""
        if not intersection:
            return float('inf')
            
        total_dist = 0
        for nic_in_intersection in intersection:
            total_dist += 1 if self.nic_to_numa[nic] == self.nic_to_numa[nic_in_intersection] else 2
            
        return total_dist
    

    @lru_cache(maxsize=2048)
    def calculate_numa_distance(self, gpu_list: tuple) -> float:
        """NUMA distance of a cross-node GPU group: +inf when the global NIC
        intersection is empty (no common optimal path); otherwise the sum,
        over nodes, of each non-intersection NIC's distance to the
        intersection."""
        gpu_list = list(gpu_list)
        machine_gpus = defaultdict(list)
        for gpu_id in gpu_list:
            machine_gpus[gpu_id // self.machine_size].append(gpu_id)
        
        if len(machine_gpus) <= 1:
            return 0.0

        machine_nic_groups = [
            set(self._groups_from_gpu_list(gpus)) 
            for gpus in machine_gpus.values()
        ]

        global_intersection = machine_nic_groups[0].copy()
        for i in range(1, len(machine_nic_groups)):
            global_intersection.intersection_update(machine_nic_groups[i])
        
        if not global_intersection:
            return float('inf')

        total_distance = 0.0
        for node_groups in machine_nic_groups:
            non_intersection_nics = node_groups - global_intersection
            
            for nic in non_intersection_nics:
                total_distance += self._get_sum_dist_to_intersection(nic, global_intersection)
                
        return total_distance
    


    @lru_cache(maxsize=200_000)
    def score(self, gpu_tuple: tuple):
        """(penalty, numa_distance) computed in a single pass."""
        if not self.enable_topology:
            return 1.0, 0.0
        n_gpus = len(gpu_tuple)
        if n_gpus <= 1:
            return 1.0, 0.0

        machine_gpus = defaultdict(list)
        for g in gpu_tuple:
            machine_gpus[g // self.machine_size].append(g)

        if len(machine_gpus) <= 1:
            return 1.0, 0.0

        machine_groups = [
            set(self._groups_from_gpu_list(gpus))
            for gpus in machine_gpus.values()
        ]

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
        """Bitmask-accelerated score(): identical semantics, no set/dict
        allocations on the hot path."""

        if not self.enable_topology:
            return 1.0, 0.0
        n_gpus = len(gpu_tuple)
        if n_gpus <= 1:
            return 1.0, 0.0

        machine_mask = {}
        for g in gpu_tuple:
            mid = g // self.machine_size
            mask = self._gpu_nic_mask(g)
            machine_mask[mid] = machine_mask.get(mid, 0) | mask

        if len(machine_mask) <= 1:
            return 1.0, 0.0

        masks = list(machine_mask.values())

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
                diff = mask & ~global_intersection  # non-intersection NICs
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