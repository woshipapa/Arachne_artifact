import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from collections import defaultdict
import argparse

class _TileEncoderModelLoader:
    """
    内部辅助类，负责加载和使用已训练好的基础瓦片模型。
    对外部调用者隐藏实现细节。
    """
    def __init__(self):
        self.analytical_model = None
        self.ml_model = None
        self.ml_columns = None

    def load(self, directory: str, name: str) -> bool:
        """从指定目录加载所有模型组件。"""
        try:
            self.analytical_model = joblib.load(os.path.join(directory, f'{name}_analytical.joblib'))
            self.ml_model = xgb.XGBRegressor()
            self.ml_model.load_model(os.path.join(directory, f'{name}_xgb.json'))
            self.ml_columns = joblib.load(os.path.join(directory, f'{name}_ml_columns.joblib'))
            print(f"✅ [Predictor] 基础瓦片模型 '{name}' 加载成功。")
            return True
        except FileNotFoundError as e:
            print(f"❌ 错误: 加载基础瓦片模型失败。文件未找到: {e}")
            return False

    @staticmethod
    def _calculate_num_time_chunks(f_series: pd.Series) -> pd.Series:
        return (1 + np.ceil(np.maximum(0, f_series - 1) / 4)).astype(int)

    def _get_features(self, df_tile: pd.DataFrame):
        if self.ml_columns is None: raise RuntimeError("基础模型尚未加载。")
        num_chunks = self._calculate_num_time_chunks(df_tile['f'])
        spatial_workload = df_tile['bs'] * df_tile['h_tile'] * df_tile['w_tile']
        total_workload = num_chunks * spatial_workload
        X_analytical = pd.DataFrame({'total_workload': total_workload})
        X_ml = df_tile[['bs', 'f']].copy()
        X_ml['bs_f_interaction'] = X_ml['bs'] * X_ml['f']
        X_ml['tile_shape'] = df_tile['h_tile'].astype(str) + 'x' + df_tile['w_tile'].astype(str)
        shape_dummies = pd.get_dummies(X_ml['tile_shape'], prefix='shape')
        X_ml = pd.concat([X_ml, shape_dummies], axis=1)
        X_ml = X_ml.drop(['tile_shape'], axis=1)
        missing_cols = set(self.ml_columns) - set(X_ml.columns)
        for c in missing_cols: X_ml[c] = 0
        X_ml = X_ml[self.ml_columns]
        return X_analytical, X_ml

    def predict(self, df_tile: pd.DataFrame) -> np.ndarray:
        X_analytical, X_ml = self._get_features(df_tile)
        y_analytical_pred = self.analytical_model.predict(X_analytical)
        residuals_pred = self.ml_model.predict(X_ml)
        return np.maximum(0, y_analytical_pred + residuals_pred)

class VaeSystemPredictor:
    """
    一个完整的VAE系统级延迟预测器。
    它封装了分层仿真和双阶段混合建模的复杂逻辑。
    """
    def __init__(self, model_dir: str, base_model_name: str, system_model_name: str):
        """
        初始化预测器并加载所有必需的模型。
        """
        self.base_tile_model = _TileEncoderModelLoader()
        self.system_analytical_model = None
        self.system_ml_model = None
        self.tile_params = {'min_h': 256, 'min_w': 256, 'stride_h': 192, 'stride_w': 192}
        
        # 在初始化时直接加载所有模型
        self._load_models(model_dir,base_model_name, system_model_name)

    def _load_models(self, model_dir: str, base_model_name: str, system_model_name: str):
        """加载所有模型文件，如果失败则抛出异常。"""
        # 1. 加载基础瓦片模型
        if not self.base_tile_model.load(model_dir, base_model_name):
            raise FileNotFoundError("无法加载基础瓦片模型，预测器无法初始化。")

        # 2. 加载系统级模型
        try:
            self.system_analytical_model = joblib.load(os.path.join(model_dir, f'{system_model_name}_sys_analytical.joblib'))
            self.system_ml_model = xgb.XGBRegressor()
            self.system_ml_model.load_model(os.path.join(model_dir, f'{system_model_name}_sys_xgb.json'))
            print(f"✅ [Predictor] 系统级模型 '{system_model_name}' 加载成功。")
        except FileNotFoundError as e:
            print(f"❌ 错误: 加载系统级模型失败。文件未找到: {e}")
            raise e
            
    def _simulate_tile_distribution(self, H, W, world_size):
        tiles_by_shape = defaultdict(list)
        int_H, int_W, int_world_size = int(H), int(W), int(world_size)
        for i in range(0, int_H, self.tile_params['stride_h']):
            for j in range(0, int_W, self.tile_params['stride_w']):
                actual_h = min(i + self.tile_params['min_h'], int_H) - i
                actual_w = min(j + self.tile_params['min_w'], int_W) - j
                tiles_by_shape[(actual_h, actual_w)].append((i, j))
        rank_tiles = [[] for _ in range(int_world_size)]
        for shape, coords_list in sorted(tiles_by_shape.items()):
            for rank_idx in range(int_world_size):
                assigned_coords = coords_list[rank_idx::int_world_size]
                if assigned_coords:
                    rank_tiles[rank_idx].extend([shape] * len(assigned_coords))
        return rank_tiles

    def _predict_computation_bottleneck(self, df_row):
        bs, f, h, w, sp = df_row['bs'], df_row['f'], df_row['h'], df_row['w'], int(df_row['sp'])
        rank_tile_shapes = self._simulate_tile_distribution(h, w, sp)
        rank_times = []
        for tiles_for_rank in rank_tile_shapes:
            if not tiles_for_rank:
                rank_times.append(0)
                continue
            tile_df = pd.DataFrame(tiles_for_rank, columns=['h_tile', 'w_tile'])
            tile_df['bs'], tile_df['f'] = bs, f
            predicted_tile_times = self.base_tile_model.predict(tile_df)
            total_rank_time = np.sum(predicted_tile_times)
            rank_times.append(total_rank_time)
        return np.max(rank_times) if rank_times else 0

    def predict(self, bs: int, f: int, h: int, w: int, sp: int) -> float:
        """
        对给定的参数进行VAE总延迟预测。
        :return: 预测的总时间（单位：秒）。
        """
        # 1. 将输入参数构造成一个单行DataFrame
        input_data = pd.DataFrame([{'bs': bs, 'f': f, 'h': h, 'w': w, 'sp': sp}])

        # 2. 仿真计算瓶颈时间 (分析特征)
        bottleneck_time = self._predict_computation_bottleneck(input_data.iloc[0])
        X_analytical = pd.DataFrame({'bottleneck_time': [bottleneck_time]})

        # 3. 准备机器学习特征
        X_ml = input_data[['bs', 'f', 'h', 'w', 'sp']].copy()
        X_ml['num_total_tiles'] = sum(len(tiles) for tiles in self._simulate_tile_distribution(h, w, sp))

        # 4. 执行双阶段预测
        y_analytical_pred = self.system_analytical_model.predict(X_analytical)
        residuals_pred = self.system_ml_model.predict(X_ml)
        
        final_prediction = np.maximum(0, y_analytical_pred + residuals_pred)
        
        final_prediction_seconds = final_prediction / 1000.0
        # --- 修改结束 ---

        # 5. 返回单个浮点数值
        return final_prediction_seconds.item()

# --- 主程序入口，用于独立测试 ---
if __name__ == "__main__":
    # --- 模型路径配置 ---
    MODEL_DIR = os.path.join('trained_models', 'wan')
    BASE_MODEL_NAME = 'tile_encoder_base_model_sp1'
    SYSTEM_MODEL_NAME = 'system_forward_model_multi_sp' # 使用多SP训练的模型

    parser = argparse.ArgumentParser(description="使用分层仿真模型预测VAE的总计算延迟。")
    parser.add_argument('--bs', type=int, default=1, help='批大小 (Batch size)')
    parser.add_argument('--f', type=int, default=113, help='帧数 (Number of frames)')
    parser.add_argument('--h', type=int, default=1280, help='帧高度 (Frame height)')
    parser.add_argument('--w', type=int, default=720, help='帧宽度 (Frame width)')
    parser.add_argument('--sp', type=int, default=5, help='并行度 (Tile parallel size)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("      VAE System Latency Predictor (Test Mode)")
    print("="*50)
    
    try:
        # 1. 初始化预测器 (会自动加载模型)
        predictor = VaeSystemPredictor(MODEL_DIR, BASE_MODEL_NAME, SYSTEM_MODEL_NAME)
        
        # 2. 调用预测方法
        total_time = predictor.predict(args.bs, args.f, args.h, args.w, args.sp)

        # 3. 显示结果
        print("\n--- 输入参数 ---")
        print(f"  - bs: {args.bs}\n  - f: {args.f}\n  - h: {args.h}\n  - w: {args.w}\n  - sp: {args.sp}")
        print("-" * 50)
        print(f"预测的 VAE 总延迟:")
        print(f"  -> {total_time:.6f} 秒")
        print("="*50)
    
    except FileNotFoundError as e:
        print(f"\n错误：模型文件加载失败。请确保您已经成功训练了基础模型和系统模型，")
        print(f"并且它们位于 '{MODEL_DIR}' 目录下。")
    except Exception as e:
        print(f"发生意外错误: {e}")
