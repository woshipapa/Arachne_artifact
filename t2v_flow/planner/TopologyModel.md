# TopologyModel

**Overview**: TopologyModel holds every calculation concerning GPU topology,
cross-node communication overlap, and the resulting performance penalty. It is
used to judge how well a given combination of GPUs maps onto the physical
network topology in distributed training.

## Initialisation

### `__init__(self, params_file: str, machine_size: int = 8)`
**Purpose**: initialise the topology model.

**What it does**:
1. Sets the number of GPUs per machine (8 by default).
2. Defines the GPU to NIC-group mapping (e.g. GPUs 0 and 1 belong to group A).
3. Defines the NIC-group to NUMA-node mapping (e.g. groups A and B sit on NUMA 0).
4. Loads the fitted parameters used for the penalty coefficient (`fit_params`).

## Core calculations

### `calculate_internal_overlap(self, gpu_list: list) -> float`
**Purpose**: compute the NIC affinity overlap score of a GPU list across nodes.

**Behaviour (generalised to N nodes)**:
1. Identify every physical machine involved and its set of NIC groups.
2. Compute the **intersection of all** those NIC-group sets.
3. Score = **size of that intersection / size of the largest per-node NIC-group
   set**.
4. This measures how well the NICs of the participating machines line up for
   cross-node communication.

### `predict_penalty(self, gpu_list: tuple) -> float`
**Purpose**: predict the communication **penalty coefficient** from the topology.

**Behaviour**:
1. **Intra-node**: when the GPU list touches a single machine, return `1.0`
   (the no-penalty baseline).
2. **Cross-node**: apply the exponential-decay model
   `1.0 + A * exp(-B * overlap)`. Even at perfect overlap (1.0) the penalty
   stays above 1.0, reflecting the base cost of crossing machines.
3. Uses an **LRU cache** so repeated queries are cheap.

### `calculate_numa_distance(self, gpu_list: tuple) -> float`
**Purpose**: total NUMA distance of a cross-node GPU assignment, measuring the
extra bus cost contributed by "stray" GPUs.

**Behaviour**:
1. Find the intersection of all nodes' NIC groups (the core communication path).
2. If that intersection is empty, return **infinity** (a very poor assignment).
3. For every "stray" NIC in each node — one **outside** the intersection — sum
   its NUMA distance to each core NIC in the intersection.
4. Same NUMA node counts as `1`, crossing NUMA nodes counts as `2`.

## Internal helpers

### `_gpu_to_group_label(self, global_gpu: int) -> str`
Convert a global GPU id to its local NIC-group label (`"A"`, `"B"`, `"C"`, `"D"`).

### `_groups_from_gpu_list(self, gpu_list: list) -> list`
Given a set of GPU ids, return their NIC-group labels, deduplicated and sorted.

### `_overlap_score_sets(self, groups_a: list, groups_b: list) -> float`
Base overlap score of two group sets (intersection size / larger set size).

### `_get_sum_dist_to_intersection(self, nic: str, intersection: set) -> int`
Accumulated NUMA distance from one NIC to every NIC in the intersection.
