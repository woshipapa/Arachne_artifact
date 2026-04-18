# predict_backward_sp_specific.py
import joblib
import os
import glob
import numpy as np

class BackwardSPPredictor:
    def __init__(self, model_dir: str = 'sp_specific_models'):
        self.models = {}
        print("--- 正在初始化SP专属Backward预测器 ---")
        model_files = glob.glob(os.path.join(model_dir, 'model_sp_*.joblib'))
        if not model_files:
            raise FileNotFoundError(f"错误: 在目录 '{model_dir}' 中未找到任何 'model_sp_*.joblib' 模型文件。")
        for f_path in model_files:
            try:
                filename = os.path.basename(f_path)
                sp_value = int(filename.split('_')[2])
                self.models[sp_value] = joblib.load(f_path)
                print(f"成功加载专家模型 for SP={sp_value} from '{filename}'")
            except (IndexError, ValueError):
                print(f"警告: 无法从文件名 '{filename}' 解析SP值，已跳过。")
        print(f"--- 后向预测器已就绪，共加载 {len(self.models)} 个模型 ---")

    def predict(self, bs: int, f: int, h: int, w: int, sp: int) -> tuple[float, str]:
        s = ((f - 1) // 4 + 1) * (h // 16) * (w // 16)
        if sp in self.models:
            model = self.models[sp]
            X = np.array([bs, s]).reshape(1, -1)
            predicted_time_ms = model.predict(X)[0]
            model_used = f"Expert (SP={sp})"
        else:
            available_sps = sorted(list(self.models.keys()))
            raise KeyError(f"错误: 找不到SP={sp}的专属模型。当前可用的模型SP值为: {available_sps}")
        return max(0, predicted_time_ms), model_used