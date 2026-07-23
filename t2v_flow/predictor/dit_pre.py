
import os
import re
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression

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

    def fit(self, df: pd.DataFrame):
        X_analytical, X_ml, y = self._get_features(df)
        self.analytical_model.fit(X_analytical, y)
        self.k_coeffs = self.analytical_model.coef_
        y_analytical_pred = self.analytical_model.predict(X_analytical)
        residuals = y - y_analytical_pred
        self.ml_model.fit(X_ml, residuals)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.k_coeffs is None: raise RuntimeError("Model has not been trained!")
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
    def __init__(self, models_base_dir: str = "trained_models", trained_model_layers: int = 60):
        self.models_base_dir = models_base_dir
        self.models = {'forward': {}, 'backward': {}}
        self.trained_model_layers = trained_model_layers
        
        self.model_name = os.environ.get('model')
        if not self.model_name:
            print(f"[ERROR] the 'model' environment variable is not set; cannot tell which model's weights to load.")
            print(f"   Set it before running: export model=your_model_name (e.g. export model=wan)")
            return

        print(f"[OK] DitPredictor: model '{self.model_name}' selected. Loading weights from its directory.")
        if self.trained_model_layers <= 0:
            raise ValueError("trained_model_layers must be positive.")
            
        self._load_all_models()

    def _load_all_models(self):
        print(f"DitPredictor: loading every performance model for '{self.model_name}'...")

        model_specific_dir = os.path.join(self.models_base_dir, self.model_name)

        if not os.path.isdir(model_specific_dir):
            print(f"[Warning] model directory '{model_specific_dir}' does not exist; the predictor will not work.")
            return
            
        pattern = re.compile(r"(forward|backward)_pass_sp(\d+)_(analytical\.joblib|xgb\.json)")
        found_files = {}
        
        for filename in os.listdir(model_specific_dir):
            match = pattern.match(filename)
            if match:
                direction, sp_size_str, model_type = match.groups()
                sp_size = int(sp_size_str)
                key = (direction, sp_size)
                if key not in found_files: found_files[key] = {}
                found_files[key][model_type] = os.path.join(model_specific_dir, filename)

        for (direction, sp_size), paths in found_files.items():
            if 'analytical.joblib' in paths and 'xgb.json' in paths:
                try:
                    model = HybridPerformanceModel.load(paths['analytical.joblib'], paths['xgb.json'])
                    self.models[direction][sp_size] = model
                    print(f"  - loaded model: {direction} sp={sp_size}")
                except Exception as e:
                    print(f"  - failed to load model: {direction} sp={sp_size}, error: {e}")
        print("DitPredictor: model loading complete.")

    def predict(self, sp: int, bs: int, frame: int, h: int, w: int, layers: int) -> float:
        if not self.model_name:
             raise RuntimeError("DitPredictor was not initialised because the 'model' environment variable is missing.")

        if sp not in self.models['forward'] or sp not in self.models['backward']:
            available_sp = list(self.models['forward'].keys())
            raise ValueError(f"No model for sp={sp}. SP sizes available for '{self.model_name}': {available_sp}")
            
        data_df = pd.DataFrame([{'bs': bs, 'f': frame, 'h': h, 'w': w}])
        fwd_model = self.models['forward'][sp]
        bwd_model = self.models['backward'][sp]
        fwd_time_ms_base = fwd_model.predict(data_df)
        bwd_time_ms_base = bwd_model.predict(data_df)
        total_time_ms_base = fwd_time_ms_base + bwd_time_ms_base
        
        final_time_s = total_time_ms_base / 1000.0
        
        return final_time_s[0]

if __name__ == "__main__":
    """
    Running this file directly executes the test code below.
    """
    print("\n" + "="*50)
    print("                      DitPredictor test")
    print("="*50)

    MODEL_TO_TEST = 'wan'
    os.environ['model'] = MODEL_TO_TEST
    print(f"*** this test simulates setting the environment variable model='{MODEL_TO_TEST}' ***\n")

    try:
        predictor = DitPredictor(models_base_dir="trained_models")
    except Exception as e:
        print(f"Failed to initialise DitPredictor: {e}")
        exit()

    test_cases = [
        {'bs': 1, 'frame': 61, 'h': 1280, 'w': 720, 'sp': 4},
        {'bs': 1, 'frame': 49, 'h': 1280, 'w': 720, 'sp': 8},
        {'bs': 1, 'frame': 57, 'h': 1280, 'w': 720, 'sp': 4},
        {'bs': 1, 'frame': 69, 'h': 1280, 'w': 720, 'sp': 8},
        {'bs': 1, 'frame': 1, 'h': 1, 'w': 1, 'sp': 99},
    ]

    print("\n--- starting the batch prediction test ---")
    print(f"{'SP':<4} | {'BS':<4} | {'F':<5} | {'H':<5} | {'W':<5} | {'Predicted Time (s)':<20}")
    print("-"*60)

    for params in test_cases:
        try:
            predicted_time = predictor.predict(
                sp=params['sp'],
                bs=params['bs'],
                frame=params['frame'],
                h=params['h'],
                w=params['w'],
                layers=60 
            )
            param_str = f"{params['sp']:<4} | {params['bs']:<4} | {params['frame']:<5} | {params['h']:<5} | {params['w']:<5}"
            print(f"{param_str} | {predicted_time:<20.6f}")

        except (ValueError, RuntimeError) as e:
            param_str = f"{params['sp']:<4} | {params['bs']:<4} | {params['frame']:<5} | {params['h']:<5} | {params['w']:<5}"
            print(f"{param_str} | ERROR: {e}")
            
    print("\nTest complete.")