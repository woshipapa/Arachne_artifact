import os
import re
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import argparse
def find_actual_time(df: pd.DataFrame, case: dict) -> float:
    """在一个DataFrame中根据bs, f, h, w查找匹配的真实时间。"""
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
            # 假设CSV中的time列单位是毫秒(ms)，将其转换为秒
            return match['time'].iloc[0] / 1000.0
    except KeyError:
        # 如果DataFrame中缺少 'bs', 'f' 等列，则返回NaN
        return np.nan
        
    return np.nan
class HybridPerformanceModel:
    """
    核心混合性能模型。
    这个类与您的训练脚本中的模型完全一致，确保了特征工程和预测逻辑的统一。
    """
    def __init__(self):
        self.analytical_model = LinearRegression(fit_intercept=False)
        self.k_coeffs = None
        # 初始化一个空的XGBoost回归器，之后会从文件加载具体参数
        self.ml_model = xgb.XGBRegressor(objective='reg:squarederror')

    @staticmethod
    def _calculate_sequence_length(df: pd.DataFrame) -> pd.Series:
        """计算DiT模型的序列长度 T。"""
        # 这个公式必须与训练时使用的公式完全相同
        return ((df['f'] - 1) / 4 + 1) * (df['h'] / 16) * (df['w'] / 16)

    @staticmethod
    def _get_features(df: pd.DataFrame):
        """为模型准备分析特征和机器学习特征。"""
        B = df['bs']
        T = HybridPerformanceModel._calculate_sequence_length(df)
        h = df['h']
        w = df['w']
        X_analytical = pd.DataFrame({'BT': B * T, 'BT_sq': B * T**2})
        X_ml = pd.DataFrame({'bs': B, 'T': T, 'h': h, 'w': w})
        # 预测时 'time' 列 (y) 不存在
        y = df['time'] if 'time' in df.columns else None
        return X_analytical, X_ml, y

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """执行双阶段预测。"""
        if self.k_coeffs is None:
            raise RuntimeError("模型尚未加载或训练，无法进行预测。")
        
        X_analytical, X_ml, _ = self._get_features(df.copy())
        
        # 阶段一：分析模型预测
        y_analytical_pred = self.analytical_model.predict(X_analytical)
        # 阶段二：ML模型修正残差
        residuals_pred = self.ml_model.predict(X_ml)
        
        # 最终结果 = 分析预测 + 残差修正
        return y_analytical_pred + residuals_pred

    @classmethod
    def load(cls, analytical_path: str, xgb_path: str):
        """从文件加载训练好的分析模型和XGBoost模型。"""
        model = cls()
        model.analytical_model = joblib.load(analytical_path)
        model.k_coeffs = model.analytical_model.coef_
        model.ml_model.load_model(xgb_path)
        return model

class DitPredictor:
    """
    一个通用的DiT模型时间预测器，可以通过模型名称加载不同的权重。
    """
    def __init__(self, model_name: str, resolution :str,models_base_dir: str = "trained_models"):
        """
        初始化预测器。
        :param model_name: 要加载的模型名称 (例如 'wan', 'hunyuan')。
        :param models_base_dir: 存放所有已训练模型的主目录。
        """
        if not model_name:
            raise ValueError("必须提供一个模型名称 (model_name)。")
            
        self.model_name = model_name
        self.resolution = resolution
        self.models_base_dir = models_base_dir
        self.models = {'forward': {}, 'backward': {}}
        self.min_sp_for_extrapolation = 3 
        
        print(f"✅ DitPredictor: 初始化完成，目标模型为 '{self.model_name}'。")
        self._load_all_models()

    def _load_all_models(self):
        """
        扫描指定模型的目录，加载所有可用的SP性能模型。
        """
        model_specific_dir = os.path.join(self.models_base_dir, self.model_name, self.resolution)
        print(f"   -> 正在从目录 '{model_specific_dir}' 加载权重...")

        if not os.path.isdir(model_specific_dir):
            print(f"❌ 警告: 模型目录 '{model_specific_dir}' 不存在。预测器将无法工作。")
            return
            
        # 正则表达式用于匹配 'forward_pass_sp4_analytical.joblib' 这样的文件名
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

        # 加载找到的成对的模型文件
        for (direction, sp_size), paths in found_files.items():
            if 'analytical.joblib' in paths and 'xgb.json' in paths:
                try:
                    model = HybridPerformanceModel.load(paths['analytical.joblib'], paths['xgb.json'])
                    self.models[direction][sp_size] = model
                    print(f"      - 成功加载模型: {direction} sp={sp_size}")
                except Exception as e:
                    print(f"      - ❗️ 加载模型失败: {direction} sp={sp_size}，错误: {e}")
        
        if not self.models['forward'] and not self.models['backward']:
             print(f"   -> 警告: 未能为模型 '{self.model_name}' 加载任何有效的性能模型。")
        else:
             print("   -> DitPredictor: 模型加载完成。")


    def _predict_by_extrapolation(self, data_df: pd.DataFrame, direction: str, target_sp: int) -> float:
            """
            【新功能】当没有直接可用的SP模型时，使用外推法进行预测。
            """
            available_sps = sorted(list(self.models[direction].keys()))
            print(f"   [外推警告] sp={target_sp} 的 {direction} 模型不存在。正在使用可用SP {available_sps} 进行外推...")

            if len(available_sps) < self.min_sp_for_extrapolation:
                raise RuntimeError(
                    f"无法为 sp={target_sp} 进行外推，因为可用的模型数量 ({len(available_sps)}) "
                    f"少于要求的最小数量 ({self.min_sp_for_extrapolation})。"
                )

            # 1. 对所有可用的sp进行预测，收集数据点
            # 我们使用 1/sp 作为特征
            inverse_sp_values = np.array([1.0 / sp for sp in available_sps]).reshape(-1, 1)
            time_predictions_ms = []

            for sp in available_sps:
                model = self.models[direction][sp]
                pred_ms = model.predict(data_df)[0]
                time_predictions_ms.append(pred_ms)
            
            time_predictions_ms = np.array(time_predictions_ms)

            # 2. 拟合线性模型: time = k1 * (1/sp) + k2
            extrapolation_model = LinearRegression()
            extrapolation_model.fit(inverse_sp_values, time_predictions_ms)

            # 3. 使用拟合好的模型来预测 target_sp 的时间
            target_inverse_sp = np.array([[1.0 / target_sp]])
            extrapolated_time_ms = extrapolation_model.predict(target_inverse_sp)

            return extrapolated_time_ms[0]


    def predict(self, bs: int, frame: int, h: int, w: int, sp: int, layers: int) -> float:
            """
            对给定的参数进行DiT总延迟预测 (forward + backward)。
            如果请求的sp不存在，则自动尝试使用外推法。
            """
            data_df = pd.DataFrame([{'bs': bs, 'f': frame, 'h': h, 'w': w}])
            
            # --- Forward Pass Prediction ---
            if sp in self.models['forward']:
                fwd_model = self.models['forward'][sp]
                fwd_time_ms = fwd_model.predict(data_df)
            else:
                # 如果模型不存在，则调用外推方法
                fwd_time_ms = self._predict_by_extrapolation(data_df, 'forward', sp)

            # --- Backward Pass Prediction ---
            if sp in self.models['backward']:
                bwd_model = self.models['backward'][sp]
                bwd_time_ms = bwd_model.predict(data_df)
            else:
                # 如果模型不存在，则调用外推方法
                bwd_time_ms = self._predict_by_extrapolation(data_df, 'backward', sp)
                
            # 计算总时间并转换为秒
            total_time_ms = fwd_time_ms + bwd_time_ms
            total_time_s = total_time_ms / 1000.0
            
            # 返回一个纯浮点数
            return total_time_s if isinstance(total_time_s, float) else total_time_s[0]

    

# --- 主程序入口，用于独立测试 (集成真实数据对比功能) ---
if __name__ == "__main__":
    """
    当此文件作为主程序直接运行时，执行批量测试，并与真实的指标数据进行对比。
    """
    # 1a. 定义您想要批量测试的组合
    test_cases = [
        {'bs': 1, 'f': 129, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 113, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 57, 'h': 1280, 'w': 720},
        {'bs': 4, 'f': 25, 'h': 1280, 'w': 720},
        # {'bs': 1, 'f': 60,  'h': 1024, 'w': 1024},
        # --- 在这里添加更多您想测试的组合 ---
    ]

    # 2. 保留命令行参数用于全局配置
    parser = argparse.ArgumentParser(description="批量预测DiT计算延迟，并与真实数据对比。")
    parser.add_argument('--model_name', type=str, default="hunyuan", help='要测试的模型名称')
    parser.add_argument('--resolution', type=str, default="720p", help='模型对应的分辨率目录')
    parser.add_argument('--sp', type=int, default=4, help='要测试的并行度 (SP size)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"       DiT Latency Predictor vs. Actual Metrics (Batch Test Mode)")
    print(f"  Model: {args.model_name} | Resolution: {args.resolution} | SP: {args.sp}")
    print("="*80)

    # 1b. (新功能) 加载真实的性能指标数据
    fwd_actual_df, bwd_actual_df = None, None
    try:
        metrics_dir = os.path.join('sp_named_metrics', args.model_name, args.resolution)
        fwd_csv_path = os.path.join(metrics_dir, f'forward_sp{args.sp}.csv')
        bwd_csv_path = os.path.join(metrics_dir, f'backward_sp{args.sp}.csv')
        fwd_actual_df = pd.read_csv(fwd_csv_path)
        bwd_actual_df = pd.read_csv(bwd_csv_path)
        print(f"✅ 成功加载真实指标文件:\n   - {fwd_csv_path}\n   - {bwd_csv_path}")
    except FileNotFoundError:
        print(f"⚠️ 警告: 未在 '{metrics_dir}' 找到真实指标CSV文件。将仅显示预测值。")
    except Exception as e:
        print(f"⚠️ 警告: 加载真实指标文件时出错: {e}。将仅显示预测值。")

    try:
        # 3. 初始化Predictor
        predictor = DitPredictor(
            model_name=args.model_name,
            resolution=args.resolution,
            models_base_dir="trained_models"
        )
        results = []
        if args.sp not in predictor.models['forward'] or args.sp not in predictor.models['backward']:
            available_sp = sorted(list(predictor.models['forward'].keys()))
            raise ValueError(f"未能为 sp={args.sp} 加载 forward/backward 模型。可用SP: {available_sp}")

        print("\n--- 开始批量测试 ---")
        # 4. 遍历测试用例，进行预测并对比
        for i, case in enumerate(test_cases):
            print(f"  [Test {i+1}/{len(test_cases)}] Running test with bs={case['bs']}, f={case['f']}, h={case['h']}, w={case['w']}...")
            
            data_df = pd.DataFrame([case])
            fwd_model = predictor.models['forward'][args.sp]
            bwd_model = predictor.models['backward'][args.sp]

            # 预测值 (单位: 秒)
            fwd_pred_s = fwd_model.predict(data_df)[0] / 1000.0
            bwd_pred_s = bwd_model.predict(data_df)[0] / 1000.0
            total_pred_s = fwd_pred_s + bwd_pred_s
            
            # (新功能) 真实值 (单位: 秒)
            fwd_actual_s = find_actual_time(fwd_actual_df, case)
            bwd_actual_s = find_actual_time(bwd_actual_df, case)
            total_actual_s = (fwd_actual_s + bwd_actual_s) if pd.notna(fwd_actual_s) and pd.notna(bwd_actual_s) else np.nan

            # (新功能) 误差计算 (%)
            fwd_err = ((fwd_pred_s - fwd_actual_s) / fwd_actual_s * 100) if pd.notna(fwd_actual_s) and fwd_actual_s > 0 else np.nan
            bwd_err = ((bwd_pred_s - bwd_actual_s) / bwd_actual_s * 100) if pd.notna(bwd_actual_s) and bwd_actual_s > 0 else np.nan
            total_err = ((total_pred_s - total_actual_s) / total_actual_s * 100) if pd.notna(total_actual_s) and total_actual_s > 0 else np.nan

            results.append({
                'bs': case['bs'], 'f': case['f'], 'h': case['h'], 'w': case['w'],
                'fwd_pred_s': fwd_pred_s, 'fwd_actual_s': fwd_actual_s, 'fwd_err_%': fwd_err,
                'bwd_pred_s': bwd_pred_s, 'bwd_actual_s': bwd_actual_s, 'bwd_err_%': bwd_err,
                'total_pred_s': total_pred_s, 'total_actual_s': total_actual_s, 'total_err_%': total_err
            })

        # 5. 使用Pandas格式化输出结果表格
        if results:
            print("\n--- 批量测试与真实值对比结果 ---")
            results_df = pd.DataFrame(results)
            column_order = [
                'bs', 'f', 'h', 'w', 
                'fwd_pred_s', 'fwd_actual_s', 'fwd_err_%',
                'bwd_pred_s', 'bwd_actual_s', 'bwd_err_%',
                'total_pred_s', 'total_actual_s', 'total_err_%'
            ]
            results_df = results_df[column_order]
            # 对不同列应用不同的格式化
            formatters = {
                'fwd_pred_s': '{:.4f}'.format, 'fwd_actual_s': '{:.4f}'.format, 'fwd_err_%': '{:+.2f}'.format,
                'bwd_pred_s': '{:.4f}'.format, 'bwd_actual_s': '{:.4f}'.format, 'bwd_err_%': '{:+.2f}'.format,
                'total_pred_s': '{:.4f}'.format, 'total_actual_s': '{:.4f}'.format, 'total_err_%': '{:+.2f}'.format,
            }
            print(results_df.to_string(index=False, formatters=formatters, na_rep='N/A'))
        else:
            print("\n--- 未生成任何测试结果 ---")

        print("="*80)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\n❌ 发生严重错误，测试中止: {e}")
    except Exception as e:
        print(f"\n❌ 发生意外错误，测试中止: {e}")