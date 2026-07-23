# predict_vae.py

import collections
from typing import Dict, Tuple
import json
import os
import numpy as np
import argparse
class _VaePredictionModel:
    def __init__(self, model_filepath: str):
        self.models: Dict[str, list] = self._load_models(model_filepath)
        if not self.models:
             raise FileNotFoundError(f"VAE model file missing or empty: '{model_filepath}'")
        print(f"[VAE Model Loader] loaded parameters for {len(self.models)} shape(s) from '{model_filepath}'.")

    def _load_models(self, model_filepath: str) -> Dict[str, list]:
        if not os.path.exists(model_filepath): return {}
        try:
            with open(model_filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e:
            print(f"Error loading or parsing the VAE model file '{model_filepath}': {e}"); return {}

    def predict_avg_time_per_tile(self, shape: Tuple[int, int, int], bs: int) -> float or None:
        if bs <= 0: return 0.0
        channels = 3; model_key = f"{channels}_{shape[0]}_{shape[1]}_{shape[2]}"
        if model_key not in self.models: return None
        polynomial_model = np.poly1d(self.models[model_key])
        predicted_ms = polynomial_model(bs)
        return predicted_ms if predicted_ms > 0 else 0.0

class VaePredictor:
    def __init__(self, model_file: str = 't2v_flow/predictor/tile_poly/merged_model_parameters_FINAL.json', 
                 tile_h: int = 256, h_stride: int = 192,
                 tile_w: int = 256, w_stride: int = 192,
                 tile_frame: int = 16, frame_stride: int = 12):
        self.model = _VaePredictionModel(model_file)
        self.tile_h, self.h_stride = tile_h, h_stride
        self.tile_w, self.w_stride = tile_w, w_stride
        self.tile_frame, self.frame_stride = tile_frame, frame_stride
        print(f"[VAE Predictor] model loaded; tiling parameters configured.")

    def predict(self, bs: int, frames: int, h: int, w: int, tile_parallel_size: int = 1) -> float:
        tile_counts = collections.defaultdict(int)
        full_tile_frame_count = self.tile_frame + 1
        if frames >= full_tile_frame_count:
            for f_start in range(0, frames, self.frame_stride):
                for h_start in range(0, h, self.h_stride):
                    for w_start in range(0, w, self.w_stride):
                        f_end = min(f_start + full_tile_frame_count, frames)
                        h_end = min(h_start + self.tile_h, h)
                        w_end = min(w_start + self.tile_w, w)
                        current_shape = (f_end - f_start, h_end - h_start, w_end - w_start)
                        tile_counts[current_shape] += 1
        else:
            for h_start in range(0, h, self.h_stride):
                for w_start in range(0, w, self.w_stride):
                    h_end=min(h_start + self.tile_h, h); w_end=min(w_start + self.tile_w, w)
                    current_shape = (frames, h_end - h_start, w_end - w_start)
                    tile_counts[current_shape] += 1
        
        total_predicted_time_serial_ms = 0.0
        for shape, count in tile_counts.items():
            avg_pred_time_ms = self.model.predict_avg_time_per_tile(shape=shape, bs=bs)
            if avg_pred_time_ms is not None:
                total_predicted_time_serial_ms += count * avg_pred_time_ms
        
        if tile_parallel_size <= 0: tile_parallel_size = 1
        final_predicted_time_ms = total_predicted_time_serial_ms / tile_parallel_size
        final_predicted_time_ms = final_predicted_time_ms / 1000
        if isinstance(final_predicted_time_ms, np.generic):
            final_predicted_time_ms = final_predicted_time_ms.item()


        return final_predicted_time_ms 

if __name__ == "__main__":
    MODEL_JSON_PATH = 't2v_flow/predictor/tile_poly/merged_model_parameters_FINAL.json'
    
    parser = argparse.ArgumentParser(description="Predict VAE total computation latency using a trained model.")
    parser.add_argument('--bs', type=int, default=1, help='Batch size.')
    parser.add_argument('--frame', type=int, default=69, help='Number of frames.')
    parser.add_argument('--h', type=int, default=1280, help='Frame height.')
    parser.add_argument('--w', type=int, default=720, help='Frame width.')
    parser.add_argument('--tile_parallel', type=int, default=8, help='Tile parallel size.')
    parser.add_argument('--model_file', type=str, default=MODEL_JSON_PATH, help=f"Path to the model file.")
    
    args = parser.parse_args()
    
    print("="*50)
    print("      VAE Total Latency Predictor (Test Mode)")
    print("="*50)
    
    try:
        vae_predictor = VaePredictor(args.model_file)
        
        total_time = vae_predictor.predict(args.bs, args.frame, args.h, args.w, args.tile_parallel)

        print(f"\nInput Parameters:")
        print(f"  - bs: {args.bs}\n  - frame: {args.frame}\n  - h: {args.h}\n  - w: {args.w}\n  - tile_parallel: {args.tile_parallel}")
        print("-" * 50)
        print(f"Predicted Total Latency for VAE:")
        print(f"  -> {total_time:.6f} seconds")
        print("="*50)
    
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")