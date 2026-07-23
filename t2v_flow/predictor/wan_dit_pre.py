import os
import re
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import argparse
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
    except KeyError:
        return np.nan
        
    return np.nan
class HybridPerformanceModel:
    def __init__(self):
        self.analytical_model = LinearRegression(fit_intercept=False)
        self.k_coeffs = None
        self.ml_model = xgb.XGBRegressor(objective='reg:squarederror')

    @staticmethod
    def _calculate_sequence_length(df: pd.DataFrame) -> pd.Series:
        return ((df['f'] - 1) / 4 + 1) * (df['h'] / 16) * (df['w'] / 16)

    @staticmethod
    def _get_features(df: pd.DataFrame):
        B = df['bs']
        T = HybridPerformanceModel._calculate_sequence_length(df)
        h = df['h']
        w = df['w']
        X_analytical = pd.DataFrame({'BT': B * T, 'BT_sq': B * T**2})
        X_ml = pd.DataFrame({'bs': B, 'T': T, 'h': h, 'w': w})
        y = df['time'] if 'time' in df.columns else None
        return X_analytical, X_ml, y

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.k_coeffs is None:
            raise RuntimeError("Model not loaded or trained; cannot predict.")
        
        X_analytical, X_ml, _ = self._get_features(df.copy())
        
        y_analytical_pred = self.analytical_model.predict(X_analytical)
        residuals_pred = self.ml_model.predict(X_ml)
        
        return y_analytical_pred + residuals_pred

    @classmethod
    def load(cls, analytical_path: str, xgb_path: str):
        model = cls()
        model.analytical_model = joblib.load(analytical_path)
        model.k_coeffs = model.analytical_model.coef_
        model.ml_model.load_model(xgb_path)
        return model

class DitPredictor:
    def __init__(self, model_name: str, resolution :str,models_base_dir: str = "trained_models"):
        if not model_name:
            raise ValueError("A model_name must be provided.")
            
        self.model_name = model_name
        self.resolution = resolution
        self.models_base_dir = models_base_dir
        self.models = {'forward': {}, 'backward': {}}
        self.min_sp_for_extrapolation = 3 
        
        print(f"[OK] DitPredictor: initialised for model '{self.model_name}'.")
        self._load_all_models()

    def _load_all_models(self):
        model_specific_dir = os.path.join(self.models_base_dir, self.model_name, self.resolution)
        print(f"   -> loading weights from '{model_specific_dir}'...")

        if not os.path.isdir(model_specific_dir):
            print(f"[ERROR] model directory '{model_specific_dir}' does not exist; the predictor will not work.")
            return
            
        pattern = re.compile(r"(forward|backward)_pass_sp(\d+)_(analytical\.joblib|xgb\.json)")
        found_files = {}
        
        for filename in os.listdir(model_specific_dir):
            match = pattern.match(filename)
            if match:
                direction, sp_size_str, model_type = match.groups()
                sp_size = int(sp_size_str)
                key = (direction, sp_size)
                
                if key not in found_files:
                    found_files[key] = {}
                
                found_files[key][model_type] = os.path.join(model_specific_dir, filename)

        for (direction, sp_size), paths in found_files.items():
            if 'analytical.joblib' in paths and 'xgb.json' in paths:
                try:
                    model = HybridPerformanceModel.load(paths['analytical.joblib'], paths['xgb.json'])
                    self.models[direction][sp_size] = model
                    print(f"      - loaded model: {direction} sp={sp_size}")
                except Exception as e:
                    print(f"      - failed to load model: {direction} sp={sp_size}, error: {e}")
        
        if not self.models['forward'] and not self.models['backward']:
             print(f"   -> warning: no valid performance model could be loaded for '{self.model_name}'.")
        else:
             print("   -> DitPredictor: model loading complete.")


    def _predict_by_extrapolation(self, data_df: pd.DataFrame, direction: str, target_sp: int) -> float:
            available_sps = sorted(list(self.models[direction].keys()))
            print(f"   [extrapolation] no {direction} model for sp={target_sp}. Extrapolating from the available SPs {available_sps}...")

            if len(available_sps) < self.min_sp_for_extrapolation:
                raise RuntimeError(
                    f"Cannot extrapolate to sp={target_sp}: the number of available models ({len(available_sps)}) "
                    f"is below the required minimum ({self.min_sp_for_extrapolation})."
                )

            inverse_sp_values = np.array([1.0 / sp for sp in available_sps]).reshape(-1, 1)
            time_predictions_ms = []

            for sp in available_sps:
                model = self.models[direction][sp]
                pred_ms = model.predict(data_df)[0]
                time_predictions_ms.append(pred_ms)
            
            time_predictions_ms = np.array(time_predictions_ms)

            extrapolation_model = LinearRegression()
            extrapolation_model.fit(inverse_sp_values, time_predictions_ms)

            target_inverse_sp = np.array([[1.0 / target_sp]])
            extrapolated_time_ms = extrapolation_model.predict(target_inverse_sp)

            return extrapolated_time_ms[0]


    def predict(self, bs: int, frame: int, h: int, w: int, sp: int, layers: int) -> float:
            data_df = pd.DataFrame([{'bs': bs, 'f': frame, 'h': h, 'w': w}])
            
            # --- Forward Pass Prediction ---
            if sp in self.models['forward']:
                fwd_model = self.models['forward'][sp]
                fwd_time_ms = fwd_model.predict(data_df)
            else:
                fwd_time_ms = self._predict_by_extrapolation(data_df, 'forward', sp)

            # --- Backward Pass Prediction ---
            if sp in self.models['backward']:
                bwd_model = self.models['backward'][sp]
                bwd_time_ms = bwd_model.predict(data_df)
            else:
                bwd_time_ms = self._predict_by_extrapolation(data_df, 'backward', sp)
                
            total_time_ms = fwd_time_ms + bwd_time_ms
            total_time_s = total_time_ms / 1000.0
            
            return total_time_s if isinstance(total_time_s, float) else total_time_s[0]

    

if __name__ == "__main__":
    """
    Running this file directly performs a batch test and compares the predictions
    """
    test_cases = [
        {'bs': 1, 'f': 129, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 113, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 57, 'h': 1280, 'w': 720},
        {'bs': 4, 'f': 25, 'h': 1280, 'w': 720},
        # {'bs': 1, 'f': 60,  'h': 1024, 'w': 1024},
    ]

    parser = argparse.ArgumentParser(description="Batch-predict DiT compute latency and compare against measurements.")
    parser.add_argument('--model_name', type=str, default="hunyuan", help='model name to test')
    parser.add_argument('--resolution', type=str, default="720p", help='resolution directory of the model')
    parser.add_argument('--sp', type=int, default=4, help='sequence-parallel size to test')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"       DiT Latency Predictor vs. Actual Metrics (Batch Test Mode)")
    print(f"  Model: {args.model_name} | Resolution: {args.resolution} | SP: {args.sp}")
    print("="*80)

    fwd_actual_df, bwd_actual_df = None, None
    try:
        metrics_dir = os.path.join('sp_named_metrics', args.model_name, args.resolution)
        fwd_csv_path = os.path.join(metrics_dir, f'forward_sp{args.sp}.csv')
        bwd_csv_path = os.path.join(metrics_dir, f'backward_sp{args.sp}.csv')
        fwd_actual_df = pd.read_csv(fwd_csv_path)
        bwd_actual_df = pd.read_csv(bwd_csv_path)
        print(f"[OK] loaded the measurement files:\n   - {fwd_csv_path}\n   - {bwd_csv_path}")
    except FileNotFoundError:
        print(f"[Warning] no measurement CSV found in '{metrics_dir}'. Showing predictions only.")
    except Exception as e:
        print(f"[Warning] failed to load the measurement files: {e}. Showing predictions only.")

    try:
        predictor = DitPredictor(
            model_name=args.model_name,
            resolution=args.resolution,
            models_base_dir="trained_models"
        )
        results = []
        if args.sp not in predictor.models['forward'] or args.sp not in predictor.models['backward']:
            available_sp = sorted(list(predictor.models['forward'].keys()))
            raise ValueError(f"Could not load forward/backward models for sp={args.sp}. Available SPs: {available_sp}")

        print("\n--- starting the batch test ---")
        for i, case in enumerate(test_cases):
            print(f"  [Test {i+1}/{len(test_cases)}] Running test with bs={case['bs']}, f={case['f']}, h={case['h']}, w={case['w']}...")
            
            data_df = pd.DataFrame([case])
            fwd_model = predictor.models['forward'][args.sp]
            bwd_model = predictor.models['backward'][args.sp]

            fwd_pred_s = fwd_model.predict(data_df)[0] / 1000.0
            bwd_pred_s = bwd_model.predict(data_df)[0] / 1000.0
            total_pred_s = fwd_pred_s + bwd_pred_s
            
            fwd_actual_s = find_actual_time(fwd_actual_df, case)
            bwd_actual_s = find_actual_time(bwd_actual_df, case)
            total_actual_s = (fwd_actual_s + bwd_actual_s) if pd.notna(fwd_actual_s) and pd.notna(bwd_actual_s) else np.nan

            fwd_err = ((fwd_pred_s - fwd_actual_s) / fwd_actual_s * 100) if pd.notna(fwd_actual_s) and fwd_actual_s > 0 else np.nan
            bwd_err = ((bwd_pred_s - bwd_actual_s) / bwd_actual_s * 100) if pd.notna(bwd_actual_s) and bwd_actual_s > 0 else np.nan
            total_err = ((total_pred_s - total_actual_s) / total_actual_s * 100) if pd.notna(total_actual_s) and total_actual_s > 0 else np.nan

            results.append({
                'bs': case['bs'], 'f': case['f'], 'h': case['h'], 'w': case['w'],
                'fwd_pred_s': fwd_pred_s, 'fwd_actual_s': fwd_actual_s, 'fwd_err_%': fwd_err,
                'bwd_pred_s': bwd_pred_s, 'bwd_actual_s': bwd_actual_s, 'bwd_err_%': bwd_err,
                'total_pred_s': total_pred_s, 'total_actual_s': total_actual_s, 'total_err_%': total_err
            })

        if results:
            print("\n--- batch test vs. measurements ---")
            results_df = pd.DataFrame(results)
            column_order = [
                'bs', 'f', 'h', 'w', 
                'fwd_pred_s', 'fwd_actual_s', 'fwd_err_%',
                'bwd_pred_s', 'bwd_actual_s', 'bwd_err_%',
                'total_pred_s', 'total_actual_s', 'total_err_%'
            ]
            results_df = results_df[column_order]
            formatters = {
                'fwd_pred_s': '{:.4f}'.format, 'fwd_actual_s': '{:.4f}'.format, 'fwd_err_%': '{:+.2f}'.format,
                'bwd_pred_s': '{:.4f}'.format, 'bwd_actual_s': '{:.4f}'.format, 'bwd_err_%': '{:+.2f}'.format,
                'total_pred_s': '{:.4f}'.format, 'total_actual_s': '{:.4f}'.format, 'total_err_%': '{:+.2f}'.format,
            }
            print(results_df.to_string(index=False, formatters=formatters, na_rep='N/A'))
        else:
            print("\n--- no test results were produced ---")

        print("="*80)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\n[ERROR] fatal error; test aborted: {e}")
    except Exception as e:
        print(f"\n[ERROR] unexpected error; test aborted: {e}")