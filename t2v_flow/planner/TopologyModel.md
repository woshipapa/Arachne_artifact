# TopologyModel 类详解

**类说明**： TopologyModel 封装了所有与 GPU 拓扑结构、跨机通信重叠度（Overlap）以及性能惩罚相关的计算逻辑。它主要用于评估分布式训练任务中，不同 GPU 组合在物理网络拓扑上的亲和性。

## 初始化方法

### `__init__(self, params_file: str, machine_size: int = 8)`
**作用**：初始化拓扑模型。

**主要逻辑**：
1. 设置单机 GPU 数量（默认为 8）。
2. 定义 GPU 到 NIC Group 的映射（如 GPU 0,1 属于 Group A）。
3. 定义 NIC Group 到 NUMA 节点 的映射（如 Group A, B 在 NUMA 0）。
4. 加载用于计算惩罚系数的拟合参数（`fit_params`）。

## 核心计算方法

### `calculate_internal_overlap(self, gpu_list: list) -> float`
**作用**：计算给定 GPU 列表在多节点间的 NIC 亲和性重叠度 (Overlap Score)。

**逻辑特点（支持 N 节点泛化）**：
1. 识别涉及的所有物理机及其对应的 NIC Group 集合。
2. 计算所有节点 NIC Group 集合的 **“总交集”**。
3. 分数计算公式为：**总交集大小 / 所有节点中最大的 NIC Group 集合大小**。
4. 能够衡量跨机通信时，不同机器间的网卡对齐程度。

### `predict_penalty(self, gpu_list: tuple) -> float`
**作用**：基于拓扑结构预测通信性能的 **惩罚系数 (Penalty)**。

**逻辑特点**：
1. **节点内通信**：如果 GPU 列表仅涉及一台物理机，直接返回 `1.0`（无惩罚基准）。
2. **跨节点通信**：应用**指数衰减模型** `1.0 + A * exp(-B * overlap)`。即便重叠度完美（Overlap=1.0），也会因为跨机通信产生基础开销（惩罚 > 1.0）。
3. 使用 **LRU 缓存** 以提升重复查询的效率。

### `calculate_numa_distance(self, gpu_list: tuple) -> float`
**作用**：计算跨机 GPU 分配方案的 NUMA 距离总和，用于衡量“游离” GPU 带来的额外总线开销。

**逻辑特点**：
1. 找出所有节点的 NIC Group 总交集（即核心通信路径）。
2. 如果总交集为空，返回**无穷大**（表示极差的拓扑分配）。
3. 遍历每个节点中 **不属于交集** 的“游离 NIC”，计算它们到总交集中所有核心 NIC 的 NUMA 距离并累加。
4. 同 NUMA 节点距离记为 `1`，跨 NUMA 节点距离记为 `2`。

## 内部辅助方法 (Internal Helpers)

### `_gpu_to_group_label(self, global_gpu: int) -> str`
**作用**：将全局 GPU ID 转换为其对应的本地 NIC Group 标签（如 `"A"`, `"B"`, `"C"`, `"D"`）。

### `_groups_from_gpu_list(self, gpu_list: list) -> list`
**作用**：输入一组 GPU ID，返回它们所属的去重且排序后的 NIC Group 标签列表。

### `_overlap_score_sets(self, groups_a: list, groups_b: list) -> float`
**作用**：计算两个 Group 集合的基础重叠分数（交集大小除以最大集合大小）。

### `_get_sum_dist_to_intersection(self, nic: str, intersection: set) -> int`
**作用**：辅助计算单个 NIC 到整个 NIC 交集集合的累积 NUMA 距离。