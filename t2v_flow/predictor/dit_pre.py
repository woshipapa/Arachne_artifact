# 文件: predictor.py

import os
import re
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression

class HybridPerformanceModel:
    """
    核心模型类，保持不变。
    """
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
        if self.k_coeffs is None: raise RuntimeError("模型尚未训练!")
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
    """
    适配 SchedulePool 的高性能DiT模型时间预测器。
    """
    def __init__(self, models_base_dir: str = "trained_models", trained_model_layers: int = 60):
        self.models_base_dir = models_base_dir
        self.models = {'forward': {}, 'backward': {}}
        self.trained_model_layers = trained_model_layers
        
        # <--- 修改点: 在初始化时从环境变量获取模型名称 ---
        self.model_name = os.environ.get('model')
        if not self.model_name:
            print(f"❌ 关键错误: 环境变量 'model' 未设置。无法确定要加载哪个模型的权重。")
            print(f"   请在运行前设置: export model=your_model_name (例如: export model=wan)")
            return # 停止初始化

        print(f"✅ DitPredictor: 已指定模型 '{self.model_name}'。准备从对应目录加载权重。")
        if self.trained_model_layers <= 0:
            raise ValueError("trained_model_layers 必须为正数。")
            
        self._load_all_models()

    def _load_all_models(self):
        print(f"DitPredictor: 正在为模型 '{self.model_name}' 加载所有性能模型...")

        # <--- 修改点: 构建指向特定模型子目录的路径 ---
        model_specific_dir = os.path.join(self.models_base_dir, self.model_name)

        if not os.path.isdir(model_specific_dir):
            print(f"❌ 警告: 模型目录 '{model_specific_dir}' 不存在。预测器将无法工作。")
            return
            
        pattern = re.compile(r"(forward|backward)_pass_sp(\d+)_(analytical\.joblib|xgb\.json)")
        found_files = {}
        
        # <--- 修改点: 遍历新的子目录 ---
        for filename in os.listdir(model_specific_dir):
            match = pattern.match(filename)
            if match:
                direction, sp_size_str, model_type = match.groups()
                sp_size = int(sp_size_str)
                key = (direction, sp_size)
                if key not in found_files: found_files[key] = {}
                # <--- 修改点: 使用新的子目录路径来组合完整路径 ---
                found_files[key][model_type] = os.path.join(model_specific_dir, filename)

        for (direction, sp_size), paths in found_files.items():
            if 'analytical.joblib' in paths and 'xgb.json' in paths:
                try:
                    model = HybridPerformanceModel.load(paths['analytical.joblib'], paths['xgb.json'])
                    self.models[direction][sp_size] = model
                    print(f"  - 成功加载模型: {direction} sp={sp_size}")
                except Exception as e:
                    print(f"  - 加载模型失败: {direction} sp={sp_size}，错误: {e}")
        print("DitPredictor: 模型加载完成。")

    def predict(self, sp: int, bs: int, frame: int, h: int, w: int, layers: int) -> float:
        if not self.model_name: # 检查模型名称是否已设置
             raise RuntimeError("DitPredictor 未成功初始化，因为 'model' 环境变量缺失。")

        if sp not in self.models['forward'] or sp not in self.models['backward']:
            available_sp = list(self.models['forward'].keys())
            raise ValueError(f"未找到 sp={sp} 的模型。'{self.model_name}' 模型可用的SP大小: {available_sp}")
            
        data_df = pd.DataFrame([{'bs': bs, 'f': frame, 'h': h, 'w': w}])
        fwd_model = self.models['forward'][sp]
        bwd_model = self.models['backward'][sp]
        fwd_time_ms_base = fwd_model.predict(data_df)
        bwd_time_ms_base = bwd_model.predict(data_df)
        total_time_ms_base = fwd_time_ms_base + bwd_time_ms_base
        
        final_time_s = total_time_ms_base / 1000.0
        
        return final_time_s[0]

# ✅ **核心修改: 添加 main 测试函数**
if __name__ == "__main__":
    """
    当此文件作为主程序直接运行时，执行此处的测试代码。
    """
    print("\n" + "="*50)
    print("                      执行 DitPredictor 测试")
    print("="*50)

    # <--- 修改点: 为了使测试脚本能独立运行，我们在这里模拟设置环境变量 ---
    # 在实际部署时，这个变量应该由您的启动脚本或外部环境来设置。
    MODEL_TO_TEST = 'wan'
    os.environ['model'] = MODEL_TO_TEST
    print(f"*** 本次测试模拟设置环境变量: model='{MODEL_TO_TEST}' ***\n")

    # 1. 初始化预测器
    # 它现在会自动查找 'trained_models/wan/' 目录
    try:
        predictor = DitPredictor(models_base_dir="trained_models")
    except Exception as e:
        print(f"初始化 DitPredictor 失败: {e}")
        exit()

    # 2. 定义一组测试用例
    test_cases = [
        {'bs': 1, 'frame': 61, 'h': 1280, 'w': 720, 'sp': 4},
        {'bs': 1, 'frame': 49, 'h': 1280, 'w': 720, 'sp': 8},
        {'bs': 1, 'frame': 57, 'h': 1280, 'w': 720, 'sp': 4},
        {'bs': 1, 'frame': 69, 'h': 1280, 'w': 720, 'sp': 8},
        {'bs': 1, 'frame': 1, 'h': 1, 'w': 1, 'sp': 99}, # 测试不存在的SP
    ]

    print("\n--- 开始批量预测测试 ---")
    print(f"{'SP':<4} | {'BS':<4} | {'F':<5} | {'H':<5} | {'W':<5} | {'Predicted Time (s)':<20}")
    print("-"*60)

    # 3. 循环执行并打印结果
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
            
    print("\n测试完成。")