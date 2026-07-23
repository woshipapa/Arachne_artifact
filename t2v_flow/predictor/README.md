# t2v_flow/predictor

Runtime performance and memory predictors for DiT and VAE components.

## Files
- dit_fit.py: Fit polynomial regressions for DiT layer timing data.
- dit_fit_new.py: Alternative fitting workflow with multiple regressors and plots.
- dit_pre.py: Legacy DiT predictor that loads trained models based on an env var.
- dit_pre_evalution.py: Evaluate DiT predictor accuracy against CSVs.
- memory_model.py: DitMemoryPredictor that uses OOM thresholds to restrict SP choices.
- predict_backward_sp_specific.py: Load SP-specific backward models for prediction.
- predict_iteration_time.py: End-to-end iteration time prediction and comparison to actual CSVs.
- vae_pre.py: VAE predictor based on polynomial tile models and tiling logic.
- wan_dit_pre.py: DiT predictor using a hybrid performance model with extrapolation.
- wan_vae_pre.py: VAE system predictor using analytical and ML models.
- __init__.py: Exports predictor classes.

## Subdirectories
- dit_poly: Joblib model files for DiT predictors.
- oom_thresholds: OOM threshold JSONs for memory-based SP selection.
