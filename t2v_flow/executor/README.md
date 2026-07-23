# t2v_flow/executor

Runtime execution and tensor transport helpers for schedule-driven runs.

## Files
- DynamicForwardStepHandler.py: Core distributed execution handler that wires the planner schedule to runtime steps, manages process groups, and exchanges tensors.
- tensor_utils.py: Utilities for packing nested tensor dicts into flat buffers and for send/recv or broadcast flows.
- old_broker_loop.py: Legacy broker loop left for reference.
- __init__.py: Exports DynamicForwardStepHandler and TensorUtils.
