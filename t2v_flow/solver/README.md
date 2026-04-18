# t2v_flow/solver

Scheduling solver experiments, formulations, and visualization helpers.

## Files
- TEST.py: Brute-force search over task orders for a fixed SP plan.
- D_97_test.py: OR-Tools CP-SAT debugging script for a minimal task set.
- cp-sat.py: CP-SAT scheduling model with SP choice and precedence constraints.
- baoli.py: Brute-force SP search with a list scheduler and Gantt plot output.
- baoli_complete.py: Extended brute-force with valid topological orders and JSON export.
- dis_time.py: Pyomo time-discretized MIP with linearized resource usage.
- hybrid_simu.py: Hybrid schedule simulator with static partitions and shared pool.
- load_draw.py: Plot rank timelines from schedule_data.json.
- solver_y_z.py: Pyomo model with y/z overlap constraints and debug checks.
- static_partition.py: Simulator for fixed GPU partitions per data item.
- generate_schedule_json.py: Build a schedule_data.json from a manual allocation plan.
- apt_install.sh: Helper for switching apt or yum repositories on Linux images.

## Subdirectories
- event_based: Event-based formulations and heuristic solvers.

## Notes
- Many scripts require external solvers or packages (ortools, pyomo, pulp, matplotlib).
