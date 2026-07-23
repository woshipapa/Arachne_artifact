This bundle contains the plotting scripts and extracted experiment data for the Arachne paper figures.

Layout:
- `scripts/`: figure scripts and helpers
- `baseline_log/`, `dynamic_flex_exp_log/`, `simulation_data/`, `t2v_flow/`: copied experiment data
- root JSON/TXT files: direct inputs used by some scripts

Run all commands from the bundle root so relative paths resolve correctly.

Usage:
```powershell
cd D:\Megatron_VAST_prof_results_bundle
python .\scripts\fig6.py
python .\scripts\fig7.py
python .\scripts\fig8.py
python .\scripts\fig10.py
python .\scripts\fig11.py
python .\scripts\fig12.py
python .\scripts\fig13.py
python .\scripts\fig14.py
```

Figure mapping:
- `fig6.py`: average iteration time / speedup comparison across stages, model sizes, and frame windows
- `fig7.py`: average GPU idle ratio comparison
- `fig8.py`: throughput comparison across model size, workload heterogeneity, and cluster size
- `fig10.py`: single-iteration per-rank FLOPs balance comparison
- `fig11.py`: CV stability over training iterations
- `fig12.py`: planning time vs training time
- `fig13.py`: normalized optimality gap vs number of cascades
- `fig14.py`: anytime Pareto / best-known planning curve

Generated outputs:
- Each script writes its figure files into the current bundle root.
- `fig6.py` also regenerates plot-data JSON files used for paper assets.

Notes:
- Some older helper scripts are still kept in `scripts/`, but the `fig*.py` files are the paper-facing entry points.
- If you rename figure scripts again, update this README and keep running from the bundle root.
