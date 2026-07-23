# predict_backward_sp_specific.py
import joblib
import os
import glob
import numpy as np

class BackwardSPPredictor:
    def __init__(self, model_dir: str = 'sp_specific_models'):
        self.models = {}
        print("--- initialising the per-SP backward predictor ---")
        model_files = glob.glob(os.path.join(model_dir, 'model_sp_*.joblib'))
        if not model_files:
            raise FileNotFoundError(f"Error: no 'model_sp_*.joblib' model file found in '{model_dir}'.")
        for f_path in model_files:
            try:
                filename = os.path.basename(f_path)
                sp_value = int(filename.split('_')[2])
                self.models[sp_value] = joblib.load(f_path)
                print(f"Loaded the expert model for SP={sp_value} from '{filename}'")
            except (IndexError, ValueError):
                print(f"Warning: could not parse an SP value from the filename '{filename}'; skipped.")
        print(f"--- backward predictor ready with {len(self.models)} model(s) ---")

    def predict(self, bs: int, f: int, h: int, w: int, sp: int) -> tuple[float, str]:
        s = ((f - 1) // 4 + 1) * (h // 16) * (w // 16)
        if sp in self.models:
            model = self.models[sp]
            X = np.array([bs, s]).reshape(1, -1)
            predicted_time_ms = model.predict(X)[0]
            model_used = f"Expert (SP={sp})"
        else:
            available_sps = sorted(list(self.models.keys()))
            raise KeyError(f"Error: no dedicated model for SP={sp}. Available SP values: {available_sps}")
        return max(0, predicted_time_ms), model_used