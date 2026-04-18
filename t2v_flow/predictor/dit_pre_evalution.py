# evaluate_model.py (已更新，支持误差排序和top-n显示)

import pandas as pd
import numpy as np
import joblib
import argparse
import os

def load_model(filepath):
    """从.joblib文件加载训练好的模型。"""
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at '{filepath}'")
        print("Please run 'fit_from_csv.py' first to generate the model file.")
        exit(1)
    try:
        model = joblib.load(filepath)
        print(f"Successfully loaded model from '{filepath}'")
        return model
    except Exception as e:
        print(f"Error loading model file: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained latency model and show worst predictions first.")
    parser.add_argument('--model_file', type=str, required=True, help="Path to the trained model .joblib file.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing test data.')
    parser.add_argument('--top_n', type=int, default=None, help='Optional: Display only the top N worst predictions. Shows all by default.')
    
    args = parser.parse_args()

    # 1. 加载模型
    model = load_model(args.model_file)

    # 2. 加载并验证测试数据CSV
    try:
        df = pd.read_csv(args.csv_file)
        df.columns = df.columns.str.strip()
        print(f"Successfully loaded {len(df)} data points from '{args.csv_file}'")
    except FileNotFoundError:
        print(f"Error: Test data file not found at '{args.csv_file}'")
        exit(1)

    # 3. 数据和特征准备
    if 'median' in df.columns:
        time_col = 'median'
    else:
        print("Error: CSV file must contain a 'median' time column.")
        exit(1)
    
    if 's' not in df.columns:
        required_cols = ['f', 'h', 'w']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain either 's' or all of {required_cols} to calculate it.")
            exit(1)
        print("Info: 's' column not found. Calculating from 'f', 'h', 'w'...")
        df['s'] = ((df['f'] - 1) // 4 + 1) * (df['h'] // 16) * (df['w'] // 16)
    
    is_linear_model = hasattr(model, 'coef_')
    
    if is_linear_model:
        print("Linear model detected. Generating polynomial features...")
        df['s^2/sp'] = (df['s']**2) / df['sp']
        df['s/sp'] = df['s'] / df['sp']
        df['s*bs/sp'] = (df['s'] * df['bs']) / df['sp']
        df['s^2*bs/sp'] = (df['s']**2 * df['bs']) / df['sp']
        df['bs^2'] = df['bs']**2
        feature_order = ['s^2/sp', 's/sp', 'bs', 'bs^2', 's*bs/sp', 's^2*bs/sp']
    else:
        print("Tree-based model detected.")
        feature_order = ['bs', 's', 'sp']
    
    X_test = df[feature_order]

    # 4. 批量预测并计算误差
    print("Predicting latency for all data points...")
    df['predicted'] = model.predict(X_test)
    df['actual'] = df[time_col]
    df['difference'] = df['predicted'] - df['actual']
    df['percent_error'] = (df['difference'].abs() / df['actual']).replace(np.inf, 0) * 100

    # --- 5. 按误差百分比降序排序 ---
    df_sorted = df.sort_values(by='percent_error', ascending=False)
    
    # 根据 top_n 参数决定要显示的行数
    display_df = df_sorted.head(args.top_n) if args.top_n is not None else df_sorted

    # --- 6. 打印排序后的详细评估结果 ---
    print("\n" + "="*110)
    title = f"Sorted Evaluation Results (Worst {len(display_df)} Predictions)"
    print(title.center(110))
    print("="*110)
    
    # 更新表头以显示 f, h, w
    header = f"{'f':<5} | {'h':<5} | {'w':<5} | {'bs':<5} | {'sp':<5} | {'Actual (ms)':<15} | {'Predicted (ms)':<15} | {'Error':<15} | {'Error %':<12}"
    print(header)
    print("-" * 110)

    for _, row in display_df.iterrows():
        print(f"{row['f']:<5} | {row['h']:<5} | {row['w']:<5} | {row['bs']:<5} | {row['sp']:<5} | "
              f"{row['actual']:<15.4f} | {row['predicted']:<15.4f} | "
              f"{row['difference']:<15.4f} | {row['percent_error']:<11.2f}%")

    # --- 7. 计算并打印基于【全部数据】的总体性能指标 ---
    print("="*110)
    # 注意：性能指标始终在完整数据集上计算，不受top_n影响
    mae = df['difference'].abs().mean()
    mape = df['percent_error'].mean()
    
    print("\nOverall Model Performance Summary (on all test data):")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE):           {mae:.4f} ms")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()