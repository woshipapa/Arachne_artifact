import os
from wan_dit_pre import DitPredictor
from wan_vae_pre import VaeSystemPredictor
import argparse
import pandas as pd

import numpy as np


def find_actual_time(df: pd.DataFrame, case: dict) -> float:
    if df is None:
        return np.nan
    
    try:
        match = df[
            (df['bs'] == case['bs']) &
            (df['f'] == case['f']) &
            (df['h'] == case['h']) &
            (df['w'] == case['w'])
        ]
        
        if not match.empty:
            return match['time'].iloc[0] / 1000.0
            
    except KeyError as e:
        print(f"[Warning] a required column is missing from the CSV while looking up the measured time: {e}.")
        return np.nan
        
    return np.nan

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    MODEL_FILENAME_CONFIGS = {
        'wan': {
            'vae_base_name': 'tile_encoder_base_model_sp1',
            'vae_system_name': 'system_forward_model_multi_sp'
        },
        'hunyuan': {
            'vae_base_name': 'tile_encoder_base_model_sp1',
            'vae_system_name': 'system_forward_model_multi_sp'
        },
    }

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    test_cases = [
        {'bs': 1, 'f': 129, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 113, 'h': 1280, 'w': 720},
        {'bs': 4, 'f': 13,  'h': 1280, 'w': 720},
        {'bs': 3, 'f': 37,  'h': 1280, 'w': 720}
    ]

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Predict end-to-end DiT and VAE compute latency and compare against measurements.")
    parser.add_argument('--model_name', type=str, required=True, choices=MODEL_FILENAME_CONFIGS.keys(),
                        help='model name to test (e.g. wan, hunyuan)')
    parser.add_argument('--resolution', type=str, required=True,
                        help='resolution label to test (e.g. 720p, 1080p, 1024_1024)')
    parser.add_argument('--sp', type=int, required=True, help='sequence-parallel size to test')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"       End-to-End Latency Predictor vs. Actual Metrics")
    print(f"  Model: {args.model_name} | Resolution: {args.resolution} | SP: {args.sp}")
    print("="*80)

    try:
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        config = MODEL_FILENAME_CONFIGS[args.model_name]
        common_path_segment = os.path.join(args.model_name, args.resolution)
        
        model_dir = os.path.join('trained_models', common_path_segment)
        metrics_dir = os.path.join('sp_named_metrics', common_path_segment)
        
        print(f"[*] model path: {model_dir}")
        print(f"[*] metrics path: {metrics_dir}")

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        dit_fwd_actual_df, dit_bwd_actual_df, vae_fwd_actual_df = None, None, None
        try:
            dit_fwd_csv = os.path.join(metrics_dir, f'forward_sp{args.sp}.csv')
            dit_bwd_csv = os.path.join(metrics_dir, f'backward_sp{args.sp}.csv')
            vae_fwd_csv = os.path.join(metrics_dir, f'vae_forward_sp{args.sp}.csv') 
            
            dit_fwd_actual_df = pd.read_csv(dit_fwd_csv)
            print(f"✅ DiT Forward CSV:  '{dit_fwd_csv}'")
            dit_bwd_actual_df = pd.read_csv(dit_bwd_csv)
            print(f"✅ DiT Backward CSV: '{dit_bwd_csv}'")
            try:
                vae_fwd_actual_df = pd.read_csv(vae_fwd_csv)
                print(f"✅ VAE Forward CSV:  '{vae_fwd_csv}'")
            except FileNotFoundError:
                print(f"[Warning] VAE measurement file not found: '{vae_fwd_csv}'")

        except FileNotFoundError as e:
            print(f"[Warning] some or all DiT measurement files are missing. Error: {e}")
        
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        print("\n--- initialising the predictors ---")
        dit_predictor = DitPredictor(
            model_name=args.model_name,
            resolution=args.resolution,
            models_base_dir="trained_models"
        )
        vae_predictor = VaeSystemPredictor(
            model_dir=model_dir,
            base_model_name=config['vae_base_name'],
            system_model_name=config['vae_system_name']
        )
        print("[OK] all predictors initialised.")

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        results = []
        print("\n--- starting the batch test ---")
        for i, case in enumerate(test_cases):
            pass

        for i, case in enumerate(test_cases):
            print(f"  [Test {i+1}/{len(test_cases)}] Running test with: {case}")

            dit_data_df = pd.DataFrame([case])
            dit_fwd_s = dit_predictor.models['forward'][args.sp].predict(dit_data_df)[0] / 1000.0
            dit_bwd_s = dit_predictor.models['backward'][args.sp].predict(dit_data_df)[0] / 1000.0

            vae_fwd_s = vae_predictor.predict(**case, sp=args.sp)

            dit_fwd_actual_s = find_actual_time(dit_fwd_actual_df, case)
            dit_bwd_actual_s = find_actual_time(dit_bwd_actual_df, case)
            vae_fwd_actual_s = find_actual_time(vae_fwd_actual_df, case)

            dit_fwd_err = ((dit_fwd_s - dit_fwd_actual_s) / dit_fwd_actual_s * 100) if pd.notna(dit_fwd_actual_s) and dit_fwd_actual_s > 0 else np.nan
            dit_bwd_err = ((dit_bwd_s - dit_bwd_actual_s) / dit_bwd_actual_s * 100) if pd.notna(dit_bwd_actual_s) and dit_bwd_actual_s > 0 else np.nan
            vae_fwd_err = ((vae_fwd_s - vae_fwd_actual_s) / vae_fwd_actual_s * 100) if pd.notna(vae_fwd_actual_s) and vae_fwd_actual_s > 0 else np.nan
            
            grand_total_pred_s = dit_fwd_s + dit_bwd_s + vae_fwd_s
            total_actual_data_exists = all(pd.notna(t) for t in [dit_fwd_actual_s, dit_bwd_actual_s, vae_fwd_actual_s])
            grand_total_actual_s = (dit_fwd_actual_s + dit_bwd_actual_s + vae_fwd_actual_s) if total_actual_data_exists else np.nan
            grand_total_err = ((grand_total_pred_s - grand_total_actual_s) / grand_total_actual_s * 100) if pd.notna(grand_total_actual_s) and grand_total_actual_s > 0 else np.nan

            results.append({
                **case,
                'dit_fwd_pred_s': dit_fwd_s, 'dit_fwd_actual_s': dit_fwd_actual_s, 'dit_fwd_err_%': dit_fwd_err,
                'dit_bwd_pred_s': dit_bwd_s, 'dit_bwd_actual_s': dit_bwd_actual_s, 'dit_bwd_err_%': dit_bwd_err,
                'vae_fwd_pred_s': vae_fwd_s, 'vae_fwd_actual_s': vae_fwd_actual_s, 'vae_fwd_err_%': vae_fwd_err,
                'total_pred_s': grand_total_pred_s, 'total_actual_s': grand_total_actual_s, 'total_err_%': grand_total_err
            })


        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        if results:
            print("\n" + "-"*120)
            print("                                         --- batch test vs. measurements ---")
            print("-"*120)
            results_df = pd.DataFrame(results)
            column_order = [
                'bs', 'f', 'h', 'w', 
                'dit_fwd_pred_s', 'dit_fwd_actual_s', 'dit_fwd_err_%',
                'dit_bwd_pred_s', 'dit_bwd_actual_s', 'dit_bwd_err_%',
                'vae_fwd_pred_s', 'vae_fwd_actual_s', 'vae_fwd_err_%',
                'total_pred_s', 'total_actual_s', 'total_err_%'
            ]
            results_df = results_df.reindex(columns=column_order) # Use reindex to avoid KeyError if a column is missing
            
            formatters = {col: '{:+.2f}'.format for col in results_df.columns if col.endswith('_err_%')}
            for col in results_df.columns:
                if col.endswith('_s'):
                    formatters[col] = '{:.4f}'.format

            print(results_df.to_string(index=False, formatters=formatters, na_rep='N/A'))
            print("-"*120)
        else:
            print("\n--- no test results were produced ---")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"\n[ERROR] configuration or file error; test aborted: {e}")
        print("   Check the model name, resolution and SP value, and make sure every required model and CSV exists at the expected path.")
    except Exception as e:
        print(f"\n[ERROR] unexpected error; test aborted: {e}")