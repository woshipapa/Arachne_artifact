import os
from wan_dit_pre import DitPredictor
from wan_vae_pre import VaeSystemPredictor
import argparse
import pandas as pd

import numpy as np


def find_actual_time(df: pd.DataFrame, case: dict) -> float:
    """
    在一个DataFrame中根据给定的'case'字典（包含bs, f, h, w）查找匹配的真实时间。
    
    :param df: 从CSV加载的包含真实指标的Pandas DataFrame。
    :param case: 包含 'bs', 'f', 'h', 'w' 键的字典。
    :return: 匹配到的时间（单位：秒），如果找不到则返回np.nan。
    """
    # 如果DataFrame不存在（例如，CSV文件未加载），直接返回NaN
    if df is None:
        return np.nan
    
    try:
        # 使用多条件过滤来查找完全匹配的行
        match = df[
            (df['bs'] == case['bs']) &
            (df['f'] == case['f']) &
            (df['h'] == case['h']) &
            (df['w'] == case['w'])
        ]
        
        # 如果找到了匹配行
        if not match.empty:
            # 假设CSV中的'time'列单位是毫秒(ms)，我们取第一条匹配记录并将其转换为秒
            return match['time'].iloc[0] / 1000.0
            
    except KeyError as e:
        # 如果DataFrame中缺少 'bs', 'f', 'h', 'w', 'time' 等关键列，则打印警告并返回NaN
        print(f"⚠️  警告: 在查找真实时间时，CSV文件中缺少关键列: {e}。")
        return np.nan
        
    # 如果没有找到匹配行，返回NaN
    return np.nan

# --- 主程序入口，用于统一预测 DiT 和 VAE 的总延迟，并与真实数据对比 ---
if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 1. 简化的模型配置 (只包含模型特有的文件名，不含路径)
    # --------------------------------------------------------------------------
    # 路径将根据命令行参数动态生成。这里只存放那些可能因模型而异的“文件名”。
    MODEL_FILENAME_CONFIGS = {
        'wan': {
            'vae_base_name': 'tile_encoder_base_model_sp1',
            'vae_system_name': 'system_forward_model_multi_sp'
        },
        'hunyuan': {
            'vae_base_name': 'tile_encoder_base_model_sp1', # 假设的文件名
            'vae_system_name': 'system_forward_model_multi_sp'    # 假设的文件名
        },
        # 在此添加其他模型的特殊文件名配置
    }

    # --------------------------------------------------------------------------
    # 2. 定义批量测试用例
    # --------------------------------------------------------------------------
    test_cases = [
        {'bs': 1, 'f': 129, 'h': 1280, 'w': 720},
        {'bs': 1, 'f': 113, 'h': 1280, 'w': 720},
        {'bs': 4, 'f': 13,  'h': 1280, 'w': 720},
        {'bs': 3, 'f': 37,  'h': 1280, 'w': 720}
        # --- 在这里添加更多您想测试的组合 ---
    ]

    # --------------------------------------------------------------------------
    # 3. 命令行参数解析
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="统一预测DiT和VAE的端到端计算延迟，并与真实数据对比。")
    parser.add_argument('--model_name', type=str, required=True, choices=MODEL_FILENAME_CONFIGS.keys(),
                        help='要测试的模型名称 (例如: wan, hunyuan)')
    parser.add_argument('--resolution', type=str, required=True,
                        help='要测试的分辨率标签 (例如: 720p, 1080p, 1024_1024)')
    parser.add_argument('--sp', type=int, required=True, help='要测试的并行度 (SP size)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"       End-to-End Latency Predictor vs. Actual Metrics")
    print(f"  Model: {args.model_name} | Resolution: {args.resolution} | SP: {args.sp}")
    print("="*80)

    try:
        # ----------------------------------------------------------------------
        # 4. 从命令行参数动态构建所有路径
        # ----------------------------------------------------------------------
        config = MODEL_FILENAME_CONFIGS[args.model_name]
        common_path_segment = os.path.join(args.model_name, args.resolution)
        
        # 模型文件所在的目录
        model_dir = os.path.join('trained_models', common_path_segment)
        # 真实指标CSV文件所在的目录
        metrics_dir = os.path.join('sp_named_metrics', common_path_segment)
        
        print(f"[*] 模型路径: {model_dir}")
        print(f"[*] 指标路径: {metrics_dir}")

        # ----------------------------------------------------------------------
        # 5. 加载真实指标CSV文件
        # ----------------------------------------------------------------------
        dit_fwd_actual_df, dit_bwd_actual_df, vae_fwd_actual_df = None, None, None
        try:
            dit_fwd_csv = os.path.join(metrics_dir, f'forward_sp{args.sp}.csv')
            dit_bwd_csv = os.path.join(metrics_dir, f'backward_sp{args.sp}.csv')
            # 假设VAE的真实数据文件名，请根据实际情况调整
            vae_fwd_csv = os.path.join(metrics_dir, f'vae_forward_sp{args.sp}.csv') 
            
            dit_fwd_actual_df = pd.read_csv(dit_fwd_csv)
            print(f"✅ DiT Forward CSV:  '{dit_fwd_csv}'")
            dit_bwd_actual_df = pd.read_csv(dit_bwd_csv)
            print(f"✅ DiT Backward CSV: '{dit_bwd_csv}'")
            try:
                vae_fwd_actual_df = pd.read_csv(vae_fwd_csv)
                print(f"✅ VAE Forward CSV:  '{vae_fwd_csv}'")
            except FileNotFoundError:
                print(f"⚠️ 未找到VAE真实指标文件: '{vae_fwd_csv}'")

        except FileNotFoundError as e:
            print(f"⚠️ 警告: 未找到部分或全部DiT真实指标文件。错误: {e}")
        
        # ----------------------------------------------------------------------
        # 6. 初始化所有预测器
        # ----------------------------------------------------------------------
        print("\n--- 正在初始化预测器 ---")
        dit_predictor = DitPredictor(
            model_name=args.model_name,
            resolution=args.resolution,
            models_base_dir="trained_models" # DitPredictor内部会自动拼接
        )
        vae_predictor = VaeSystemPredictor(
            model_dir=model_dir, # VaePredictor需要完整的路径
            base_model_name=config['vae_base_name'],
            system_model_name=config['vae_system_name']
        )
        print("✅ 所有预测器初始化成功。")

        # ----------------------------------------------------------------------
        # 7. 执行批量预测与对比
        # ----------------------------------------------------------------------
        results = []
        print("\n--- 开始批量测试 ---")
        for i, case in enumerate(test_cases):
            # ... (此处省略了与上一版本完全相同的预测和计算逻辑)
            # 为了简洁，这里仅示意，请使用下面完整的循环代码
            pass # 占位符

        # 完整的循环代码
        for i, case in enumerate(test_cases):
            print(f"  [Test {i+1}/{len(test_cases)}] Running test with: {case}")

            # --- DiT 预测 ---
            dit_data_df = pd.DataFrame([case])
            dit_fwd_s = dit_predictor.models['forward'][args.sp].predict(dit_data_df)[0] / 1000.0
            dit_bwd_s = dit_predictor.models['backward'][args.sp].predict(dit_data_df)[0] / 1000.0

            # --- VAE 预测 ---
            vae_fwd_s = vae_predictor.predict(**case, sp=args.sp)

            # --- 查找真实值 ---
            dit_fwd_actual_s = find_actual_time(dit_fwd_actual_df, case)
            dit_bwd_actual_s = find_actual_time(dit_bwd_actual_df, case)
            vae_fwd_actual_s = find_actual_time(vae_fwd_actual_df, case)

            # --- 计算误差 ---
            dit_fwd_err = ((dit_fwd_s - dit_fwd_actual_s) / dit_fwd_actual_s * 100) if pd.notna(dit_fwd_actual_s) and dit_fwd_actual_s > 0 else np.nan
            dit_bwd_err = ((dit_bwd_s - dit_bwd_actual_s) / dit_bwd_actual_s * 100) if pd.notna(dit_bwd_actual_s) and dit_bwd_actual_s > 0 else np.nan
            vae_fwd_err = ((vae_fwd_s - vae_fwd_actual_s) / vae_fwd_actual_s * 100) if pd.notna(vae_fwd_actual_s) and vae_fwd_actual_s > 0 else np.nan
            
            # --- 计算总和 ---
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
        # 8. 格式化并显示结果
        # ----------------------------------------------------------------------
        if results:
            print("\n" + "-"*120)
            print("                                         --- 批量测试与真实值对比结果 ---")
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
            print("\n--- 未生成任何测试结果 ---")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"\n❌ 发生配置或文件错误，测试中止: {e}")
        print("   请检查模型名称、分辨率、SP值是否正确，并确保所有必需的模型和CSV文件都存在于规范的路径下。")
    except Exception as e:
        print(f"\n❌ 发生意外错误，测试中止: {e}")